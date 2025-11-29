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
    num_layers: int = 12
    freq_dim: int = 256
    video_depth: int = 6
    eps: float = 1e-5
    image_key: str = "base_0_rgb"
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
        self._horizon = config.horizon  # This is 'k' (prediction length)
        self._history_len = config.history_len  # This is 'h' (history length)

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
        h_len, f_len = self._history_len, self._horizon

        # Calculate how many full windows fit in the sequence
        num_windows = (t - h_len) // f_len
        if num_windows < 1:
            raise ValueError(f"Insufficient sequence length {t} for H={h_len}, F={f_len}")

        # Embed observations (B, T, ...)
        obs_p = _model.preprocess_observation(rng, observation, train=train, image_keys=list(observation.images.keys()))
        embeddings = self.embed_inputs(obs_p, train=train, rng=rng) # (B, T, S, P)
        embeddings = jnp.reshape(embeddings, (b, t, -1, embeddings.shape[-1]))

        # We slice out the Initial History and the Future Targets/Actions
        # init_history: (B, h_len, N, C)
        init_history = embeddings[:, :h_len]

        # Reshape Targets and Actions into: (Num_Windows, B, f_len, ...)
        # This allows scan to iterate over the first dimension automatically
        valid_len = num_windows * f_len

        targets = embeddings[:, h_len : h_len + valid_len]
        targets = targets.reshape(num_windows, b, f_len, *targets.shape[2:])

        act_seq = actions[:, h_len : h_len + valid_len]
        act_seq = act_seq.reshape(num_windows, b, f_len, *act_seq.shape[2:])

        def scan_step(carry, inputs):
            # Carry holds the rolling history buffer; Inputs are the next window slices
            curr_history, rng = carry
            target_window, action_window = inputs

            B, T, N, C = target_window.shape

            # --- Flow Matching Math ---
            x_prior = curr_history[:, -1:, :, :] # Last frame of history is x_0

            # Target Velocity (Flow Matching)
            target_residual = target_window - x_prior

            # Action Projection & Dropout
            a_tokens = self.action_in_proj(action_window)
            if train:
                rng, r_drop = jax.random.split(rng)
                mask = jax.random.bernoulli(r_drop, p=0.12, shape=(B, 1, 1))
                a_tokens = jnp.where(mask, 0.0, a_tokens)

            # Noise & Timesteps (Diffusion Forcing)
            rng, r_time, r_noise = jax.random.split(rng, 3)
            timestep = jax.random.uniform(r_time, (B,), minval=0.001, maxval=0.999)
            t_bc = timestep[:, None, None, None]

            noise = jax.random.normal(r_noise, target_residual.shape)
            x_noisy = (1 - t_bc) * target_residual + t_bc * noise

            # Predict Velocity
            v_pred = self._diffusion_transformer(x_noisy, curr_history, a_tokens, timestep.flatten())

            # Loss: v_pred should match (epsilon - x_data)
            # Note: Your original target was `noise - target_residual`
            loss = jnp.mean((v_pred - (noise - target_residual)) ** 2)

            # Reconstruct Data for Next History (x_data = x_t - t * v)
            pred_data = x_noisy - (t_bc * v_pred)

            # --- Update History Buffer (Sliding Window) ---
            # Drop oldest f_len frames, append predicted data
            next_history = jnp.concatenate([curr_history[:, f_len:], pred_data], axis=1)
            next_history = next_history.astype(curr_history.dtype)

            return (next_history, rng), loss

        # Iterate over the reshaped windows
        init_carry = (init_history, rng)
        _, losses = jax.lax.scan(scan_step, init_carry, (targets, act_seq))

        return jnp.mean(losses)

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
        num_steps = (t - h_len) // f_len

        if num_steps < 1:
            raise ValueError(f"Insufficient sequence length {t} for history {h_len} and horizon {f_len}.")

        # Preprocess and embed observations
        observation = _model.preprocess_observation(
            rng, observation, train=train, image_keys=list(observation.images.keys())
        )
        image_embeddings = self.embed_inputs(observation, train=train, rng=rng)
        image_embeddings = jnp.reshape(image_embeddings, (b, t, -1, image_embeddings.shape[-1]))

        def predict_step(history, future_actions, rng):
            """Predict next embeddings from history and actions."""
            x_prior = history[:, -1:, :, :]
            B, T, N, C = history.shape
            
            action_tokens = self.action_in_proj(future_actions)
            rng_t, rng_n = jax.random.split(rng)
            timestep = jax.random.uniform(rng_t, (B,), minval=0.02, maxval=0.98)
            noise = jax.random.normal(rng_n, (B, T, N, C))
            x_noisy = noise * timestep[:, None, None, None]
            
            y_pred = self._diffusion_transformer(x_noisy, history, action_tokens, timestep)
            predicted =  x_prior - y_pred
            return predicted.astype(history.dtype)

        def scan_step(carry, step_idx):
            """Rollout step for scan."""
            history, rng = carry
            rng, step_rng = jax.random.split(rng)
            
            start_idx = h_len + step_idx * f_len
            step_actions = jax.lax.dynamic_slice(actions, [0, start_idx, 0], [b, f_len, actions.shape[2]])
            
            predictions = predict_step(history, step_actions, step_rng)
            return (predictions, rng), predictions

        # Run all prediction steps via scan
        init_history = image_embeddings[:, :h_len]
        _, all_preds = jax.lax.scan(scan_step, (init_history, rng), jnp.arange(num_steps))
        
        # Reshape: (num_steps, b, f_len, s, p) -> (b, num_steps * f_len, s, p)
        all_preds = einops.rearrange(all_preds, "n b t s p -> b (n t) s p")
        all_predictions = jnp.concatenate([init_history, all_preds], axis=1)

        # Compute fused embeddings for all timesteps
        def compute_fused_for_timestep(img_tokens):
            return self.get_fused_embedding(img_tokens, observation)

        batch_fused_fn = jax.vmap(compute_fused_for_timestep, in_axes=1, out_axes=1)
        
        past_fused_embeddings = batch_fused_fn(image_embeddings)
        future_fused_embeddings = batch_fused_fn(all_predictions)

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
