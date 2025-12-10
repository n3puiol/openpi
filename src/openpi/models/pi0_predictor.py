import dataclasses

import numpy as np
from openpi.models.pi0 import Pi0, Pi0Config, make_attn_mask
import openpi.shared.nnx_utils as nnx_utils
from openpi.models import model as _model
from openpi.shared import array_typing as at
from typing_extensions import override

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models.dit import DiffusionTransformer
from diffusers import FlaxAutoencoderKL


@dataclasses.dataclass(frozen=True)
class Pi0PredictorConfig(Pi0Config):
    in_channel: int = 4
    hidden_size: int = 1024
    num_heads: int = 8
    num_layers: int = 12
    freq_dim: int = 256
    video_depth: int = 6
    eps: float = 1e-5
    image_key: str = "base_0_rgb"
    horizon: int = 5
    history_len: int = 5
    num_denoise_steps: int = 6

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
        # This will automatically freeze the VAE params as they are not matched by this regex
        return nnx.Not(nnx_utils.PathRegex("_diffusion_transformer.*"))


class Pi0Predictor(Pi0):
    def __init__(self, config: Pi0PredictorConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)

        self._eps = config.eps
        self._image_key = config.image_key
        self._horizon = config.horizon
        self._history_len = config.history_len
        self._num_denoise_steps = config.num_denoise_steps
        self._baseline_embedding = nnx.Variable(
            jnp.load(config.baseline_embedding_path)
        )
        self._goal_embedding = nnx.Variable(jnp.load(config.goal_embedding_path))
        self._alpha = config.alpha

        vae, vae_params = FlaxAutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse", from_pt=True, dtype=jnp.float32
        )
        self._vae = vae
        self._vae_params = jax.tree_util.tree_map(lambda x: nnx.Param(x), vae_params)
        self._vae_scaling_factor = vae.config.scaling_factor

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

    def _encode_with_vae(
        self, images: jnp.ndarray, rng: at.KeyArrayLike
    ) -> jnp.ndarray:
        """Encode images using the VAE encoder."""
        images_trans = jnp.transpose(images, (0, 3, 1, 2))

        vae_params_raw = jax.tree_util.tree_map(lambda x: x.value, self._vae_params)

        posterior = self._vae.apply(
            {"params": vae_params_raw}, images_trans, method=self._vae.encode
        )

        latents = posterior.latent_dist.sample(rng)
        latents = latents * self._vae_scaling_factor
        return latents

    def embed_inputs(
        self, observation: _model.Observation, train: bool, rng: at.KeyArrayLike
    ) -> at.Float[at.Array, "*b s emb"]:
        """Encode images using SD VAE instead of PaliGemma."""
        images = observation.images[self._image_key]
        return self._encode_with_vae(images, rng)

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

        num_windows = (t - h_len) // f_len
        if num_windows < 1:
            raise ValueError(
                f"Insufficient sequence length {t} for H={h_len}, F={f_len}"
            )

        # Preprocess and embed observations
        rng, rng_preprocess, rng_embed = jax.random.split(rng, 3)
        obs_p = _model.preprocess_observation(
            rng_preprocess,
            observation,
            train=train,
            image_keys=list(observation.images.keys()),
        )
        embeddings = self.embed_inputs(obs_p, train=train, rng=rng_embed)
        embeddings = jnp.reshape(embeddings, (b, t, -1, 4))

        # Extract initial history
        init_history = embeddings[:, :h_len]

        # Prepare windows for scan:  (num_windows, B, f_len, ...)
        valid_len = num_windows * f_len
        targets = embeddings[:, h_len : h_len + valid_len]
        targets = targets.reshape(b, num_windows, f_len, *targets.shape[2:])
        targets = jnp.transpose(
            targets, (1, 0, 2, 3, 4)
        )  # (num_windows, B, f_len, N, C)

        act_seq = actions[:, h_len : h_len + valid_len]
        act_seq = act_seq.reshape(b, num_windows, f_len, -1)
        act_seq = jnp.transpose(
            act_seq, (1, 0, 2, 3)
        )  # (num_windows, B, f_len, action_dim)

        def scan_step(carry, inputs):
            curr_history, step_rng = carry
            target_window, action_window = inputs
            batch_size = target_window.shape[0]

            # Split RNG properly for noise and timestep
            step_rng, rng_noise, rng_time = jax.random.split(step_rng, 3)

            # Sample noise and timestep
            noise = jax.random.normal(rng_noise, target_window.shape)
            timestep = jax.random.uniform(
                rng_time, (batch_size,), minval=0.02, maxval=0.98
            )

            # Flow matching interpolation:  x_t = (1-t)*x_0 + t*x_1
            # where x_0 = noise, x_1 = target
            t_broadcast = timestep[:, None, None, None]
            x_t = (1.0 - t_broadcast) * noise + t_broadcast * target_window

            # Target velocity: v = x_1 - x_0 = target - noise
            target_velocity = target_window - noise

            # Predict velocity
            v_pred = self._diffusion_transformer(
                x_t, curr_history, action_window, timestep
            )

            # MSE loss
            loss = jnp.mean(jnp.square(v_pred - target_velocity))

            # Note: History stays fixed (parallel window training, not autoregressive)
            # For autoregressive:  next_history = concat(curr_history[:, f_len:], target_window)
            return (curr_history, step_rng), loss

        init_carry = (init_history, rng)
        _, losses = jax.lax.scan(scan_step, init_carry, (targets, act_seq))

        return jnp.mean(losses)

    def _decode_with_vae(self, latents: jnp.ndarray) -> jnp.ndarray:
        """Decode latents back to images using the VAE decoder.

        Args:
            latents: Latent representations of shape (B, T, N, C) where N = H' * W' (flattened spatial),
                     or (B, T, H', W', C) if already in spatial format.

        Returns:
            Decoded images of shape (B, T, H, W, C) normalized to [0, 1].
        """
        # Shape: (B, T, N, C) where N = H' * W' (flattened spatial dims)
        b, t, n, c = latents.shape
        # Infer spatial dimensions (assuming square latent space)
        h_latent = int(np.sqrt(n))
        w_latent = h_latent
        if h_latent * w_latent != n:
            raise ValueError(
                f"Cannot reshape flattened spatial dim {n} to square. "
                f"Expected perfect square."
            )
        # Reshape to (B, T, H', W', C)
        latents = latents.reshape(b, t, h_latent, w_latent, c)

        # Now latents is (B, T, H', W', C)
        b, t = latents.shape[:2]

        # Flatten batch and time: (B, T, H', W', C) -> (B*T, H', W', C)
        latents_flat = latents.reshape(-1, *latents.shape[2:])

        # Scale latents back (inverse of encoding scaling)
        latents_flat = latents_flat / self._vae_scaling_factor

        # Convert to NCHW format for VAE decoder: (N, H', W', C) -> (N, C, H', W')
        latents_flat = jnp.transpose(latents_flat, (0, 3, 1, 2))

        # Unwrap VAE params
        vae_params_raw = jax.tree_util.tree_map(lambda x: x.value, self._vae_params)

        # Decode using VAE
        decoded = self._vae.apply(
            {"params": vae_params_raw}, latents_flat, method=self._vae.decode
        ).sample

        # Convert back to NHWC format: (N, C, H, W) -> (N, H, W, C)
        if decoded.shape[1] == 3:
            decoded = jnp.transpose(decoded, (0, 2, 3, 1))

        # Normalize from [-1, 1] to [0, 1]
        decoded = (decoded + 1.0) / 2.0
        decoded = jnp.clip(decoded, 0.0, 1.0)

        # Reshape back to include time dimension
        decoded = decoded.reshape(b, t, *decoded.shape[1:])

        return decoded


    def predict_future(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        decode_to_images: bool = False,
    ) -> jnp.ndarray:
        """Predict future states using flow matching inference."""
        b, t, _ = actions.shape

        # Preprocess and encode observations
        obs_p = _model.preprocess_observation(
            rng, observation, train=False, image_keys=list(observation.images.keys())
        )
        rng, rng_embed = jax.random.split(rng)
        embeddings = self.embed_inputs(obs_p, train=False, rng=rng_embed)
        embeddings = jnp.reshape(embeddings, (b, t, -1, 4))

        _, _, N, C = embeddings.shape
        history = embeddings[:, : self._history_len]
        action_window = actions[:, self._history_len : self._history_len + self._horizon]

        # Initialize at t=0 with pure noise
        rng, rng_init = jax.random.split(rng)
        x_t = jax.random.normal(rng_init, (b, self._horizon, N, C))

        # Integrate from t=0 to t=1 (noise → data)
        num_steps = self._num_denoise_steps
        step_size = 1.0 / num_steps
        timesteps = jnp.linspace(0.0, 1.0, num_steps, endpoint=False)

        def denoise_step(x_curr, timestep):
            t_batch = jnp.full((b,), timestep)
            v_pred = self._diffusion_transformer(x_curr, history, action_window, t_batch)
            x_next = x_curr + v_pred * step_size  # Euler step:  x += v * dt
            return x_next, None

        predicted_latents, _ = jax.lax.scan(denoise_step, x_t, timesteps)

        if decode_to_images:
            return self._decode_with_vae(predicted_latents)
        return predicted_latents

    def compute_regularized_reward(self, state_embedding: jnp.ndarray) -> jnp.ndarray:
        s = state_embedding
        g = self._goal_embedding.value
        b = self._baseline_embedding.value

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
