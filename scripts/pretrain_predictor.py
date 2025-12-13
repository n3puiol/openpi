# """Pretraining script for Something-Something V2 dataset.

# This script pretrains the video prediction components (video encoder, diffusion transformer)
# on Something-Something V2 without action conditioning. The pretrained weights can then be
# fine-tuned on LIBERO with action conditioning.

# Usage:
#     python pretrain_predictor.py \
#         --checkpoint_dir /path/to/checkpoints \
#         --num_train_steps 100000 \
#         --batch_size 32 \
#         --resolution 224 \
#         --num_frames 16
# """

# import argparse
# import dataclasses
# import functools
# import logging
# import platform
# from typing import Iterator, Optional

# import etils.epath as epath
# import flax.nnx as nnx
# from flax.training import common_utils
# import jax
# import jax.numpy as jnp
# import numpy as np
# import optax
# import torch
# import torch.nn.functional as F
# from datasets import load_dataset
# import tqdm_loggable.auto as tqdm
# import wandb

# from openpi.models.pi0_predictor import Pi0Predictor, Pi0PredictorConfig
# import openpi.shared.array_typing as at
# import openpi.training.checkpoints as _checkpoints
# import openpi.training.optimizer as _optimizer
# import openpi.training.sharding as sharding
# import openpi.training.utils as training_utils


# # =============================================================================
# # Configuration
# # =============================================================================


# @dataclasses.dataclass(frozen=True)
# class PretrainConfig:
#     """Configuration for Something-Something V2 pretraining."""

#     # Model config (matches Pi0PredictorConfig)
#     in_channel: int = 4
#     hidden_size: int = 1024
#     num_heads: int = 8
#     num_layers: int = 12
#     freq_dim: int = 256
#     video_depth: int = 6
#     eps: float = 1e-5

#     # Data
#     resolution: int = 224
#     num_frames: int = 10
#     history_len: int = 5
#     horizon: int = 5

#     # Training
#     batch_size: int = 32
#     num_train_steps: int = 100000
#     seed: int = 42

#     # Optimizer
#     lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(
#         default_factory=lambda: _optimizer.CosineDecaySchedule(
#             warmup_steps=1000, peak_lr=1e-4, decay_steps=100000, decay_lr=1e-5
#         )
#     )
#     optimizer: _optimizer.OptimizerConfig = dataclasses.field(
#         default_factory=lambda: _optimizer.AdamW(weight_decay=0.01)
#     )
#     ema_decay: float | None = 0.9999

#     # Logging and checkpointing
#     checkpoint_dir: str = "./checkpoints/pretrain_ssv2"
#     log_interval: int = 100
#     save_interval: int = 5000
#     keep_period: int | None = 5000
#     wandb_enabled: bool = True
#     project_name: str = "world-model-pretrain"
#     exp_name: str = "ssv2-pretrain"

#     # Infrastructure
#     fsdp_devices: int = 1
#     num_workers: int = 4
#     overwrite: bool = False
#     resume: bool = True

#     def to_model_config(self) -> Pi0PredictorConfig:
#         """Convert to Pi0PredictorConfig for model initialization."""
#         return Pi0PredictorConfig(
#             in_channel=self.in_channel,
#             hidden_size=self.hidden_size,
#             num_heads=self.num_heads,
#             num_layers=self.num_layers,
#             freq_dim=self.freq_dim,
#             video_depth=self.video_depth,
#             eps=self.eps,
#             horizon=self.horizon,
#             history_len=self.history_len,
#         )

#     @property
#     def checkpoint_path(self) -> epath.Path:
#         return epath.Path(self.checkpoint_dir)


# # =============================================================================
# # Dataset and DataLoader
# # =============================================================================


# def resize_video_torch(
#     video_tensor: torch.Tensor, resolution: int = 224
# ) -> torch.Tensor:
#     """Center crop to square and resize video frames using PyTorch."""
#     if video_tensor.shape[-1] == 3:  # (T, H, W, C)
#         video_tensor = video_tensor.permute(0, 3, 1, 2)

#     T, C, H, W = video_tensor.shape

#     size = min(H, W)
#     top = (H - size) // 2
#     left = (W - size) // 2
#     video_tensor = video_tensor[:, :, top : top + size, left : left + size]

#     resized = F.interpolate(
#         video_tensor.float(),
#         size=(resolution, resolution),
#         mode="bilinear",
#         align_corners=False,
#     )

#     resized = resized.permute(0, 2, 3, 1) / 255.0
#     return resized


# class SomethingSomethingV2Dataset:
#     """Something-Something V2 dataset using HuggingFace datasets."""

#     def __init__(
#         self,
#         num_frames: int = 16,
#         resolution: int = 224,
#         streaming: bool = True,
#     ):
#         self.num_frames = num_frames
#         self.resolution = resolution
#         self.dataset = load_dataset(
#             "jxie/something_something_v2",
#             streaming=streaming,
#         )

#     def __iter__(self) -> Iterator[dict]:
#         for item in self.dataset["train"]:
#             try:
#                 video_data = self._process_video(item["video"])
#                 if video_data is None:
#                     continue
#                 yield {"video": video_data}
#             except Exception as e:
#                 logging.warning(f"Failed to process video: {e}")
#                 continue

#     def _process_video(self, video_reader) -> Optional[np.ndarray]:
#         """Process video from HuggingFace format."""
#         frames = [frame["data"] for frame in video_reader]

#         if len(frames) < self.num_frames:
#             return None

#         video_tensor = torch.stack(frames)
#         total_frames = len(video_tensor)
#         indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
#         video_tensor = video_tensor[indices]
#         video_tensor = resize_video_torch(video_tensor, self.resolution)

#         return video_tensor.numpy().astype(np.float32)


# class PretrainDataLoader:
#     """Data loader for pretraining on Something-Something V2."""

#     def __init__(
#         self,
#         config: PretrainConfig,
#         sharding: Optional[jax.sharding.Sharding] = None,
#     ):
#         self.config = config
#         self.sharding = sharding
#         self.dataset = SomethingSomethingV2Dataset(
#             num_frames=config.num_frames,
#             resolution=config.resolution,
#             streaming=True,
#         )

#     def __iter__(self) -> Iterator[dict]:
#         batch = []
#         for item in self.dataset:
#             batch.append(item)
#             if len(batch) >= self.config.batch_size:
#                 yield self._collate_and_shard(batch)
#                 batch = []

#     def _collate_and_shard(self, batch: list[dict]) -> dict:
#         """Collate batch and apply sharding."""
#         collated = {"video": np.stack([b["video"] for b in batch], axis=0)}

#         if self.sharding is not None:
#             collated = jax.tree.map(
#                 lambda x: jax.make_array_from_process_local_data(self.sharding, x),
#                 collated,
#             )
#         return collated


# # =============================================================================
# # Training State and Step (reusing training_utils.TrainState)
# # =============================================================================


# @at.typecheck
# def init_pretrain_state(
#     config: PretrainConfig,
#     init_rng: at.KeyArrayLike,
#     mesh: jax.sharding.Mesh,
#     *,
#     resume: bool,
# ) -> tuple[training_utils.TrainState, jax.sharding.NamedSharding]:
#     """Initialize pretraining state using the shared TrainState."""
#     tx = _optimizer.create_optimizer(
#         config.optimizer, config.lr_schedule, weight_decay_mask=None
#     )

#     def init(rng: at.KeyArrayLike) -> training_utils.TrainState:
#         model_config = config.to_model_config()
#         model = model_config.create(rng)
#         params = nnx.state(model)

#         return training_utils.TrainState(
#             step=0,
#             params=params,
#             model_def=nnx.graphdef(model),
#             tx=tx,
#             opt_state=tx.init(params),
#             ema_decay=config.ema_decay,
#             ema_params=params if config.ema_decay else None,
#         )

#     train_state_shape = jax.eval_shape(init, init_rng)
#     state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

#     if resume:
#         return train_state_shape, state_sharding

#     replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
#     train_state = jax.jit(init, out_shardings=state_sharding)(init_rng)

#     return train_state, state_sharding


# @at.typecheck
# def pretrain_step(
#     config: PretrainConfig,
#     rng: at.KeyArrayLike,
#     state: training_utils.TrainState,
#     batch: dict,
# ) -> tuple[training_utils.TrainState, dict[str, jnp.ndarray]]:
#     """Single pretraining step."""
#     model = nnx.merge(state.model_def, state.params)

#     def loss_fn(
#         model: Pi0Predictor,
#         rng: at.KeyArrayLike,
#         video: jnp.ndarray,
#     ) -> jnp.ndarray:
#         return model.compute_pretrain_loss(rng, video)

#     train_rng = jax.random.fold_in(rng, state.step)
#     video = batch["video"]

#     loss, grads = nnx.value_and_grad(loss_fn)(model, train_rng, video)

#     params = state.params
#     updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
#     new_params = optax.apply_updates(params, updates)

#     nnx.update(model, new_params)
#     new_params = nnx.state(model)

#     new_state = dataclasses.replace(
#         state, step=state.step + 1, params=new_params, opt_state=new_opt_state
#     )

#     if state.ema_decay is not None and state.ema_params is not None:
#         new_state = dataclasses.replace(
#             new_state,
#             ema_params=jax.tree.map(
#                 lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
#                 state.ema_params,
#                 new_params,
#             ),
#         )

#     info = {
#         "loss": loss,
#         "grad_norm": optax.global_norm(grads),
#         "param_norm": optax.global_norm(new_params),
#     }

#     return new_state, info


# # =============================================================================
# # Logging (reusing patterns from train_predictor)
# # =============================================================================


# def init_logging():
#     """Custom logging format for better readability."""
#     level_mapping = {
#         "DEBUG": "D",
#         "INFO": "I",
#         "WARNING": "W",
#         "ERROR": "E",
#         "CRITICAL": "C",
#     }

#     class CustomFormatter(logging.Formatter):
#         def format(self, record):
#             record.levelname = level_mapping.get(record.levelname, record.levelname)
#             return super().format(record)

#     formatter = CustomFormatter(
#         fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
#         datefmt="%H:%M:%S",
#     )

#     logger = logging.getLogger()
#     logger.setLevel(logging.INFO)
#     if logger.handlers:
#         logger.handlers[0].setFormatter(formatter)


# def init_wandb(config: PretrainConfig, *, resuming: bool = False):
#     """Initialize wandb logging."""
#     if not config.wandb_enabled:
#         wandb.init(mode="disabled")
#         return

#     ckpt_dir = config.checkpoint_path
#     ckpt_dir.mkdir(parents=True, exist_ok=True)

#     if resuming and (ckpt_dir / "wandb_id.txt").exists():
#         run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
#         wandb.init(id=run_id, resume="must", project=config.project_name)
#     else:
#         wandb.init(
#             name=config.exp_name,
#             config=dataclasses.asdict(config),
#             project=config.project_name,
#         )
#         (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


# # =============================================================================
# # Main Training Loop
# # =============================================================================


# def main(config: PretrainConfig):
#     """Main pretraining loop."""
#     init_logging()
#     logging.info(f"Running on: {platform.node()}")
#     logging.info(f"Config: {config}")

#     if config.batch_size % jax.device_count() != 0:
#         raise ValueError(
#             f"Batch size {config.batch_size} must be divisible by device count {jax.device_count()}"
#         )

#     jax.config.update(
#         "jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser())
#     )

#     rng = jax.random.key(config.seed)
#     train_rng, init_rng = jax.random.split(rng)

#     mesh = sharding.make_mesh(config.fsdp_devices)
#     data_sharding = jax.sharding.NamedSharding(
#         mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS)
#     )
#     replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

#     # Use existing checkpoint infrastructure
#     checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
#         config.checkpoint_path,
#         keep_period=config.keep_period,
#         overwrite=config.overwrite,
#         resume=config.resume,
#     )

#     init_wandb(config, resuming=resuming)

#     # Create data loader
#     logging.info("Creating data loader...")
#     data_loader = PretrainDataLoader(config, sharding=data_sharding)
#     data_iter = iter(data_loader)

#     logging.info("Loading first batch...")
#     batch = next(data_iter)
#     logging.info(f"Batch shapes: {jax.tree.map(lambda x: x.shape, batch)}")

#     # Log sample images to wandb
#     if config.wandb_enabled:
#         sample_video = jax.device_get(batch["video"][0])
#         num_preview_frames = min(8, sample_video.shape[0])
#         preview_frames = np.concatenate(
#             [
#                 (sample_video[i] * 255).astype(np.uint8)
#                 for i in range(num_preview_frames)
#             ],
#             axis=1,
#         )
#         wandb.log({"sample_video_frames": wandb.Image(preview_frames)}, step=0)

#     # Initialize training state
#     logging.info("Initializing training state...")
#     train_state, train_state_sharding = init_pretrain_state(
#         config, init_rng, mesh, resume=resuming
#     )
#     jax.block_until_ready(train_state)
#     logging.info(
#         f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}"
#     )

#     # Resume from checkpoint if available
#     if resuming:
#         logging.info("Resuming from checkpoint...")
#         train_state = _checkpoints.restore_state(
#             checkpoint_manager, train_state, data_loader
#         )
#         logging.info(f"Resumed from step {train_state.step}")

#     # JIT compile training step
#     logging.info("Compiling training step...")
#     ppretrain_step = jax.jit(
#         functools.partial(pretrain_step, config),
#         in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
#         out_shardings=(train_state_sharding, replicated_sharding),
#         donate_argnums=(1,),
#     )

#     # Warmup compilation
#     with sharding.set_mesh(mesh):
#         train_state, info = ppretrain_step(train_rng, train_state, batch)
#     jax.block_until_ready(train_state)
#     logging.info("Compilation complete")

#     # Training loop
#     start_step = int(train_state.step)
#     pbar = tqdm.tqdm(
#         range(start_step, config.num_train_steps),
#         initial=start_step,
#         total=config.num_train_steps,
#         dynamic_ncols=True,
#     )

#     infos = []
#     for step in pbar:
#         with sharding.set_mesh(mesh):
#             train_state, info = ppretrain_step(train_rng, train_state, batch)
#         infos.append(info)

#         # Logging
#         if step % config.log_interval == 0 and step > start_step:
#             stacked_infos = common_utils.stack_forest(infos)
#             reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
#             info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
#             pbar.write(f"Step {step}: {info_str}")
#             wandb.log(reduced_info, step=step)
#             infos = []

#         # Get next batch
#         try:
#             batch = next(data_iter)
#         except StopIteration:
#             logging.info("Dataset exhausted, restarting...")
#             data_iter = iter(data_loader)
#             batch = next(data_iter)

#         # Checkpointing using existing infrastructure
#         if (
#             step % config.save_interval == 0 and step > start_step
#         ) or step == config.num_train_steps - 1:
#             _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

#     logging.info("Waiting for checkpoint manager to finish")
#     checkpoint_manager.wait_until_finished()
#     logging.info("Pretraining complete!")


# def parse_args() -> PretrainConfig:
#     """Parse command line arguments."""
#     parser = argparse.ArgumentParser(description="Pretrain on Something-Something V2")

#     # Model
#     parser.add_argument("--hidden_size", type=int, default=256)
#     parser.add_argument("--num_heads", type=int, default=4)
#     parser.add_argument("--num_layers", type=int, default=4)
#     parser.add_argument("--video_depth", type=int, default=4)
#     parser.add_argument("--freq_dim", type=int, default=256)

#     # Data
#     parser.add_argument("--resolution", type=int, default=224)
#     parser.add_argument("--num_frames", type=int, default=10)
#     parser.add_argument("--history_len", type=int, default=5)
#     parser.add_argument("--horizon", type=int, default=5)

#     # Training
#     parser.add_argument("--batch_size", type=int, default=1)
#     parser.add_argument("--num_train_steps", type=int, default=100000)
#     parser.add_argument("--learning_rate", type=float, default=1e-4)
#     parser.add_argument("--warmup_steps", type=int, default=1000)
#     parser.add_argument("--weight_decay", type=float, default=0.01)
#     # parser.add_argument("--ema_decay", type=float, default=0.9999)
#     parser.add_argument("--ema_decay", type=float, default=None)
#     parser.add_argument("--seed", type=int, default=42)

#     # Logging
#     parser.add_argument(
#         "--checkpoint_dir", type=str, default="./checkpoints/pretrain_ssv2"
#     )
#     parser.add_argument("--log_interval", type=int, default=100)
#     parser.add_argument("--save_interval", type=int, default=5000)
#     parser.add_argument("--keep_period", type=int, default=5000)
#     parser.add_argument("--wandb_enabled", action="store_true")
#     parser.add_argument("--project_name", type=str, default="world-model-pretrain")
#     parser.add_argument("--exp_name", type=str, default="ssv2-pretrain")

#     # Infrastructure
#     parser.add_argument("--fsdp_devices", type=int, default=1)
#     parser.add_argument("--num_workers", type=int, default=4)
#     parser.add_argument("--overwrite", action="store_true")
#     parser.add_argument("--resume", action="store_true", default=True)
#     parser.add_argument("--no_resume", action="store_false", dest="resume")

#     args = parser.parse_args()

#     return PretrainConfig(
#         in_channel=4,
#         hidden_size=args.hidden_size,
#         num_heads=args.num_heads,
#         num_layers=args.num_layers,
#         freq_dim=args.freq_dim,
#         video_depth=args.video_depth,
#         eps=1e-5,
#         resolution=args.resolution,
#         num_frames=args.num_frames,
#         history_len=args.history_len,
#         horizon=args.horizon,
#         batch_size=args.batch_size,
#         num_train_steps=args.num_train_steps,
#         seed=args.seed,
#         lr_schedule=_optimizer.CosineDecaySchedule(
#             warmup_steps=args.warmup_steps,
#             peak_lr=args.learning_rate,
#             decay_steps=args.num_train_steps,
#             decay_lr=args.learning_rate * 0.1,
#         ),
#         optimizer=_optimizer.AdamW(weight_decay=args.weight_decay),
#         ema_decay=args.ema_decay,
#         checkpoint_dir=args.checkpoint_dir,
#         log_interval=args.log_interval,
#         save_interval=args.save_interval,
#         keep_period=args.keep_period,
#         wandb_enabled=args.wandb_enabled,
#         project_name=args.project_name,
#         exp_name=args.exp_name,
#         fsdp_devices=args.fsdp_devices,
#         num_workers=args.num_workers,
#         overwrite=args.overwrite,
#         resume=args.resume,
#     )


# if __name__ == "__main__":
#     config = parse_args()
#     main(config)
import dataclasses
import functools
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
from openpi.models.pi0_predictor import Pi0Predictor


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {
        "DEBUG": "D",
        "INFO": "I",
        "WARNING": "W",
        "ERROR": "E",
        "CRITICAL": "C",
    }

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(
    config: _config.TrainConfig,
    *,
    resuming: bool,
    log_code: bool = False,
    enabled: bool = True,
):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(
    loader: _weight_loaders.WeightLoader, params_shape: at.Params
) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    # at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {
            k: v
            for k, v in traverse_util.flatten_dict(loaded_params).items()
            if not isinstance(v, jax.ShapeDtypeStruct)
        }
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh,
    *,
    resume: bool,
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(
        config.optimizer, config.lr_schedule, weight_decay_mask=None
    )

    def init(
        rng: at.KeyArrayLike, partial_params: at.Params | None = None
    ) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(
            params,
            config.freeze_filter,
            lambda p: p.replace(p.value.astype(jnp.bfloat16)),
        )

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(
        config.weight_loader, train_state_shape.params.to_pure_dict()
    )
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(
        model, train_rng, observation, actions
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(
        state, step=state.step + 1, params=new_params, opt_state=new_opt_state
    )
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                state.ema_params,
                new_params,
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(
                nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")
            ),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def log_predicted_images(
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
    step: int,
):
    """Generate and log predicted future images to wandb."""
    # Use EMA params if available, otherwise use regular params
    params = state.ema_params if state.ema_params is not None else state.params
    model = nnx.merge(state.model_def, params)
    model.eval()

    observation, actions = batch

    # Call predict_future with decode_to_images=True
    pred_rng = jax.random.fold_in(rng, step)
    predicted_images = model.predict_future(
        pred_rng,
        observation,
        actions,
        decode_to_images=True,
    )

    # predicted_images shape: (B, horizon, H, W, C)
    predicted_images = jax.device_get(predicted_images)

    # Log predicted images to wandb
    images_to_log = []
    horizon_frames = predicted_images[0]  # (horizon, H, W, C)
    horizon_frames = (horizon_frames * 255).astype(np.uint8)
    concat_frames = np.concatenate(
        [horizon_frames[t] for t in range(horizon_frames.shape[0])], axis=1
    )
    images_to_log.append(
        wandb.Image(concat_frames, caption=f"Step {step}: Predicted future frames")
    )

    wandb.log({"predicted_future_images": images_to_log}, step=step)


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update(
        "jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser())
    )

    rng = jax.random.key(config.seed)
    train_rng, val_rng, init_rng = jax.random.split(rng, 3)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS)
    )
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_ssv2_dataloader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)

    batch = next(data_iter)
    logging.info(
        f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}"
    )

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(
            np.concatenate(
                [np.array(img[i]) for img in batch[0].images.values()], axis=1
            )
        )
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    # check min and max pixel values
    sample_image = jax.device_get(batch[0].images["base_0_rgb"][0])
    logging.info(f"Sample image pixel range: min={sample_image.min()}, max={sample_image.max()}")

    train_state, train_state_sharding = init_train_state(
        config, init_rng, mesh, resume=resuming
    )
    jax.block_until_ready(train_state)
    logging.info(
        f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}"
    )

    if resuming:
        train_state = _checkpoints.restore_state(
            checkpoint_manager, train_state, data_loader
        )

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []

        if step % 1000 == 0 and step > 0:
            try:
                pbar.write(f"Step {step}: Generating predicted future images...")
                with sharding.set_mesh(mesh):
                    log_predicted_images(val_rng, train_state, batch, step)
                pbar.write(f"Step {step}: Logged predicted future images to wandb")
            except Exception as e:
                pbar.write(f"Step {step}: Failed to generate predicted images: {e}")

        batch = next(data_iter)

        if (
            step % config.save_interval == 0 and step > start_step
        ) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
