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
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding
import openpi.training.checkpoints as _checkpoints
import openpi.training.optimizer as _optimizer
import openpi.training.utils as training_utils
from openpi.shared import array_typing as at
from predictor.dit_predictor import Predictor


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
        run = wandb.init(
            name=f"{config.exp_name}_predictor",
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        if run is not None:
            (ckpt_dir / "wandb_id.txt").write_text(run.id)

    if log_code and wandb.run is not None:
        wandb.run.log_code(str(epath.Path(__file__).parent.parent))


@at.typecheck
@dataclasses.dataclass
class PredictorTrainState:
    step: int
    predictor: Predictor
    optimizer: nnx.Optimizer


@at.typecheck
def init_predictor_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh
) -> tuple[PredictorTrainState, Any]:

    # Create policy first (this will be frozen)
    checkpoint_dir = download.maybe_download(
        "/scratch/s5649552/.cache/openpi/openpi-assets/checkpoints/pi0_fast_libero_predictor"
    )
    policy = _policy_config.create_trained_policy(config, checkpoint_dir)

    # Create the predictor
    predictor_rng, _ = jax.random.split(init_rng)
    predictor = Predictor(
        policy=policy, rngs=nnx.Rngs(predictor_rng, params=predictor_rng)
    )

    # Create optimizer for predictor only (policy is frozen)
    # Filter to only train diffusion transformer parameters
    trainable_filter = nnx.filterlib.PathContains("_diffusion_transformer")

    # Create optimizer with filtered parameters
    optimizer = nnx.Optimizer(
        predictor, optax.adam(learning_rate=5e-5), wrt=trainable_filter
    )

    train_state = PredictorTrainState(
        step=0,
        predictor=predictor,
        optimizer=optimizer,
    )

    # Determine sharding for train state
    train_state_shape = jax.eval_shape(lambda: train_state)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    return train_state, state_sharding


@at.typecheck
def predictor_train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: PredictorTrainState,
    batch: dict[str, Any],
) -> tuple[PredictorTrainState, dict[str, at.Array]]:

    def loss_fn(predictor: Predictor, batch: dict[str, Any]):
        # Extract images and actions from batch
        images = batch["image"]["base_0_rgb"]  # Shape: (B, T, H, W, C)
        actions = batch["actions"]  # Shape: (B, T, action_dim)

        loss = predictor.compute_loss(images, actions)
        return loss

    # Compute loss and gradients using the optimizer
    loss, grads = nnx.value_and_grad(loss_fn)(state.predictor, batch)

    # Update the predictor using the optimizer
    state.optimizer.update(grads)

    new_state = dataclasses.replace(
        state,
        step=state.step + 1,
    )

    # Compute metrics for trainable parameters only
    trainable_filter = nnx.filterlib.PathContains("_diffusion_transformer")
    trainable_params = nnx.state(state.predictor, nnx.All(nnx.Param, trainable_filter))

    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(trainable_params),
    }

    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running predictor training on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update(
        "jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser())
    )

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS)
    )
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize checkpoint directory for predictor
    predictor_checkpoint_dir = str(
        config.checkpoint_dir.parent / f"{config.exp_name}_predictor"
    )
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        predictor_checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # Create data loader
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(
        f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}"
    )

    # Log images from first batch to sanity check
    if "image" in batch and "base_0_rgb" in batch["image"]:
        images_to_log = [
            wandb.Image(
                np.array(batch["image"]["base_0_rgb"][i, 0])
            )  # Take first timestep
            for i in range(min(5, batch["image"]["base_0_rgb"].shape[0]))
        ]
        wandb.log({"camera_views": images_to_log}, step=0)

    # Initialize predictor train state
    train_state, train_state_sharding = init_predictor_train_state(
        config, init_rng, mesh
    )
    jax.block_until_ready(train_state)
    logging.info(f"Initialized predictor train state")

    # JIT compile the training step
    ptrain_step = jax.jit(
        functools.partial(predictor_train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = train_state.step
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

        batch = next(data_iter)

        if (
            step % config.save_interval == 0 and step > start_step
        ) or step == config.num_train_steps - 1:
            # Save predictor checkpoint (simplified for NNX)
            logging.info(f"Saving checkpoint at step {step}")
            # For now, we'll skip detailed checkpointing and just log

    logging.info("Training completed successfully")


if __name__ == "__main__":
    config = _config.get_config("pi0_fast_libero_predictor")
    main(config)
