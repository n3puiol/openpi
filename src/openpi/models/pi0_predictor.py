import dataclasses

import einops
from openpi.models.pi0 import Pi0, Pi0Config, make_attn_mask
import openpi.shared.nnx_utils as nnx_utils
from openpi.models import model as _model
from openpi.shared import array_typing as at
from typing_extensions import override

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models.dit import DiffusionTransformer


@dataclasses.dataclass(frozen=True)
class Pi0PredictorConfig(Pi0Config):
    in_channel: int = 2048
    hidden_size: int = 1024
    num_heads: int = 8
    num_layers: int = 8
    freq_dim: int = 256
    video_depth: int = 4
    eps: float = 1e-5
    image_key: str = "base_0_rgb"
    rollout_factor: float = 1.0
    horizon: int = 5
    history_len: int = 5

    # Reward estimation embeddings
    baseline_embedding_path: str = (
        "/scratch/s5649552/openpi/reward_estimation_embeddings/baseline_embedding_pi0_libero_predictor.npy"
    )
    goal_embedding_path: str = (
        "/scratch/s5649552/openpi/reward_estimation_embeddings/goal_embedding_pi0_libero_predictor.npy"
    )
    alpha: float = 0.5  # blending factor for regularized reward

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0_PREDICTOR

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0Predictor":
        # Construct Rngs with explicit param and dropout keys
        k_params, k_dropout = jax.random.split(rng)  # type: ignore[arg-type]
        rngs = nnx.Rngs(params=k_params, dropout=k_dropout)
        return Pi0Predictor(self, rngs=rngs)

    @override
    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        return nnx.Not(nnx_utils.PathRegex("_diffusion_transformer.*"))


class Pi0Predictor(Pi0):
    def __init__(self, config: Pi0PredictorConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)

        self._eps = config.eps
        self._image_key = config.image_key
        self._rollout_factor = config.rollout_factor
        self._horizon = config.horizon  # This is 'k' (prediction length)
        self._history_len = (
            config.history_len
        )  # This is 'l' (history length), fixed to 4 in LaDi-WM

        self.baseline_embedding = nnx.Variable(jnp.load(config.baseline_embedding_path))
        self.goal_embedding = nnx.Variable(jnp.load(config.goal_embedding_path))
        self._alpha = config.alpha

        self._diffusion_transformer = DiffusionTransformer(
            in_channel=config.in_channel,
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            n_layers=config.num_layers,
            freq_dim=config.freq_dim,
            video_depth=config.video_depth,
            epsilon=config.eps,
            rngs=rngs,
        )

    def add_noise(
        self, x: at.Array, noise: at.Array, timestep: at.Array, c: at.Array
    ) -> at.Array:
        time = timestep.reshape(c.shape[0], *((1,) * (len(c.shape) - 1)))
        x_noisy = x + c * time + time * noise
        return x_noisy

    def embed_inputs(
        self, observation: _model.Observation, train: bool, rng: at.KeyArrayLike
    ) -> at.Float[at.Array, "*b s emb"]:
        return self.PaliGemma.img(observation.images[self._image_key], train=False)[0]

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        b, t, _ = actions.shape
        h_len = self._history_len
        f_len = self._horizon

        max_rollout_steps = (t - h_len) // f_len

        if max_rollout_steps < 1:
            raise ValueError(
                f"Insufficient action length {t} for history {h_len} and horizon {f_len}."
            )

        observation = _model.preprocess_observation(
            rng, observation, train=train, image_keys=list(observation.images.keys())
        )

        image_embeddings = self.embed_inputs(observation, train=train, rng=rng)
        _, s, p = image_embeddings.shape
        image_embeddings = jnp.reshape(image_embeddings, (b, t, s, p))

        def compute_step_loss(lc_his, lc_next, a_future, rng):
            """Compute loss for a single prediction step."""
            x_prior = lc_his[:, -1:, :]
            action_tokens = self.action_in_proj(a_future)

            # Build Residual Target (Velocity)
            targets_concat = jnp.concatenate([x_prior, lc_next], axis=1)
            target_velocity = jnp.diff(targets_concat, axis=1)
            c_res = -target_velocity  # Drift term

            # Split RNG
            rng_t, rng_n = jax.random.split(rng)
            timestep = (
                jax.random.uniform(
                    rng_t, shape=(c_res.shape[0],), minval=0.0, maxval=1.0
                )
                * (1.0 - self._eps)
                + self._eps
            )
            noise = jax.random.normal(rng_n, shape=c_res.shape)

            # Add Noise
            x_noisy = self.add_noise(target_velocity, noise, timestep, c_res)

            # Forward Pass
            y_pred = self._diffusion_transformer(
                x_noisy, lc_his, action_tokens, timestep
            )

            # Reconstruction
            pred_cumulative_delta = jnp.cumsum(y_pred, axis=1)
            predicted_embeddings = x_prior + pred_cumulative_delta
            # emb_loss = jnp.mean((predicted_embeddings - lc_next) ** 2)
            # jax.debug.print("Embedding loss: {}", emb_loss)

            # Losses
            loss = jnp.mean((y_pred - c_res) ** 2)

            return loss, predicted_embeddings

        lc_his = image_embeddings[:, :h_len]
        lc_next = image_embeddings[:, h_len : h_len + f_len]
        a_future = actions[:, h_len : h_len + f_len]

        rng, step_rng = jax.random.split(rng)
        teacher_loss, predicted_embeddings = compute_step_loss(
            lc_his, lc_next, a_future, step_rng
        )

        def rollout_step(carry, step_idx):
            """Single rollout step using predicted embeddings as history."""
            predicted_emb, total_loss, rng = carry

            # Use predicted embeddings as new history
            if predicted_emb.shape[1] >= h_len:
                new_history = predicted_emb[:, -h_len:, :, :]
            else:
                needed = h_len - predicted_emb.shape[1]
                new_history = jnp.concatenate(
                    [image_embeddings[:, h_len - needed : h_len], predicted_emb], axis=1
                )

            # Ground truth target for this step
            start_idx = h_len + (step_idx + 1) * f_len
            start_indices = [0, start_idx, 0, 0]
            slice_sizes = [b, f_len, s, p]
            next_target = jax.lax.dynamic_slice(
                image_embeddings, start_indices, slice_sizes
            )

            # Actions - use dynamic_slice
            next_actions = jax.lax.dynamic_slice(
                actions, [0, start_idx, 0], [b, f_len, actions.shape[2]]
            )

            # Compute loss for this rollout step
            rng, step_rng = jax.random.split(rng)
            step_loss, new_predictions = compute_step_loss(
                new_history, next_target, next_actions, step_rng
            )

            return (new_predictions, total_loss + step_loss, rng), (step_loss)

        if max_rollout_steps > 1:
            init_carry = (predicted_embeddings, 0.0, rng)
            (_, rollout_loss_sum, _), (step_losses) = jax.lax.scan(
                rollout_step, init_carry, jnp.arange(max_rollout_steps - 1)
            )

            # Average rollout loss
            rollout_loss = rollout_loss_sum / (max_rollout_steps - 1)
        else:
            rollout_loss = 0.0

        total_loss = teacher_loss + rollout_loss

        return total_loss

    def get_fused_embedding(self, image_tokens, obs):
        input_mask = []
        ar_mask = []
        tokens = []

        tokens.append(image_tokens)
        input_mask.append(
            einops.repeat(
                obs.image_masks[self._image_key],
                "b -> b s",
                s=image_tokens.shape[1],
            )
        )
        ar_mask += [False] * image_tokens.shape[1]

        tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
        tokens.append(tokenized_inputs)
        input_mask.append(obs.tokenized_prompt_mask)
        ar_mask += [False] * tokenized_inputs.shape[1]

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)

        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        (fused_sequence_embeddings, _), _ = self.PaliGemma.llm(
            [tokens, None], mask=attn_mask, positions=positions
        )

        mask_expanded = jnp.expand_dims(input_mask, axis=-1)
        summed_embeddings = jnp.sum(fused_sequence_embeddings * mask_expanded, axis=1)
        num_valid_tokens = jnp.sum(input_mask, axis=1, keepdims=True)
        pooled_fused_embedding = summed_embeddings / jnp.maximum(num_valid_tokens, 1)
        return pooled_fused_embedding

    def predict_future(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ):
        b, t, _ = actions.shape
        h_len = self._history_len
        f_len = self._horizon

        # Calculate number of rollout steps possible
        max_rollout_steps = (t - h_len) // f_len

        if max_rollout_steps < 1:
            raise ValueError(
                f"Insufficient sequence length {t} for history {h_len} and horizon {f_len}."
            )

        observation = _model.preprocess_observation(
            rng, observation, train=train, image_keys=list(observation.images.keys())
        )
        image_embeddings = self.embed_inputs(observation, train=train, rng=rng)

        _, s, p = image_embeddings.shape
        image_embeddings = jnp.reshape(image_embeddings, (b, t, s, p))

        def predict_step(lc_his, a_future, rng):
            """Predict future embeddings for a single step."""
            x_prior = lc_his[:, -1:, :]  # (b, 1, s, p)
            action_tokens = self.action_in_proj(a_future)

            # Split RNG
            rng_t, rng_n = jax.random.split(rng)
            timestep = (
                jax.random.uniform(
                    rng_t, shape=(x_prior.shape[0],), minval=0.0, maxval=1.0
                )
                * (1.0 - self._eps)
                + self._eps
            )
            x_noisy = jax.random.normal(rng_n, shape=(b, f_len, s, p))

            # Forward through diffusion transformer
            y_pred = self._diffusion_transformer(
                x_noisy, lc_his, action_tokens, timestep
            )
            pred_cumulative_delta = jnp.cumsum(y_pred, axis=1)
            predicted_embedding = x_prior + pred_cumulative_delta

            return predicted_embedding

        # Split into history and future segments
        lc_his = image_embeddings[:, :h_len]
        # First prediction step
        a_future = actions[:, h_len : h_len + f_len]
        rng, step_rng = jax.random.split(rng)
        first_predictions = predict_step(lc_his, a_future, step_rng)

        all_predictions = [lc_his, first_predictions]

        def rollout_step(carry, step_idx):
            """Single rollout step using predicted embeddings as history."""
            predicted_emb, rng = carry

            # Get next actions
            start_idx = h_len + (step_idx + 1) * f_len
            next_actions = jax.lax.dynamic_slice(
                actions, [0, start_idx, 0], [b, f_len, actions.shape[2]]
            )

            # Predict next segment
            rng, step_rng = jax.random.split(rng)
            new_predictions = predict_step(predicted_emb, next_actions, step_rng)

            return (new_predictions, rng), new_predictions

        if max_rollout_steps > 1:
            init_carry = (first_predictions, rng)
            _, rollout_predictions = jax.lax.scan(
                rollout_step, init_carry, jnp.arange(max_rollout_steps - 1)
            )
            # rollout_predictions shape: (max_rollout_steps - 1, b, f_len, s, p)
            # Reshape to (b, (max_rollout_steps - 1) * f_len, s, p)
            rollout_predictions = jnp.reshape(
                rollout_predictions,
                (b, (max_rollout_steps - 1) * f_len, s, p),
            )
            all_predictions.append(rollout_predictions)

        all_predictions = jnp.concatenate(all_predictions, axis=1)

        def compute_fused_for_timestep(img_tokens):
            return self.get_fused_embedding(img_tokens, observation)

        # Use vmap to parallelize across time dimension
        batch_fused_fn = jax.vmap(compute_fused_for_timestep, in_axes=1, out_axes=1)

        past_fused_embeddings = batch_fused_fn(image_embeddings)  # (b, t, emb_dim)
        future_fused_embeddings = batch_fused_fn(all_predictions)  # (b, t, emb_dim)

        # emb_loss = jnp.mean((past_fused_embeddings - future_fused_embeddings) ** 2)
        # jax.debug.print("Fused embedding MSE loss: {}", emb_loss)

        return past_fused_embeddings, future_fused_embeddings

    def compute_regularized_reward(self, state_embedding: jnp.ndarray) -> jnp.ndarray:
        s = state_embedding
        g = self.goal_embedding.value
        b = self.baseline_embedding.value

        direction_vector = g - b
        direction_vector_norm_sq = jnp.sum(direction_vector**2)

        s_minus_b = s - b
        projection_scalar = jnp.dot(s_minus_b, direction_vector) / jnp.maximum(
            direction_vector_norm_sq, 1e-6
        )
        projected_s = b + projection_scalar * direction_vector

        blended_embedding = (1 - self._alpha) * s + self._alpha * projected_s

        reward = 1.0 - 0.5 * jnp.sum((blended_embedding - g) ** 2)
        return reward
