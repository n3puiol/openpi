import dataclasses

import numpy as np
import openpi.shared.nnx_utils as nnx_utils
from openpi.models import model as _model
from openpi.shared import array_typing as at
from typing_extensions import override

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models.dit import DiffusionTransformer, MultiViewVideoTransformer
from diffusers import FlaxAutoencoderKL


# class Pi0PredictorConfig(Pi0Config):
@dataclasses.dataclass(frozen=True)
class Pi0PredictorConfig(_model.BaseModelConfig):
    in_channel: int = 4
    hidden_size: int = 1024
    num_heads: int = 8
    num_layers: int = 12
    freq_dim: int = 256
    video_depth: int = 6
    eps: float = 1e-5
    # image_key: str = "base_0_rgb"
    horizon: int = 5
    history_len: int = 5
    num_denoise_steps: int = 12
    pretrain: bool = True
    num_cross_view_layers: int = 2  # Number of cross-view attention layers
    primary_image_key: str = "base_0_rgb"  # Primary view for loss computation
    ignore_image_keys: list[str] = dataclasses.field(default_factory=list)

    action_dim: int = 32
    action_horizon: int = 10
    max_token_len: int = 48

    # Reward estimation embeddings
    # baseline_embedding_path: str = (
    #     "/scratch/s5649552/openpi/reward_estimation_embeddings/baseline_embedding_pi0_libero_predictor.npy"
    # )
    # goal_embedding_path: str = (
    #     "/scratch/s5649552/openpi/reward_estimation_embeddings/goal_embedding_pi0_libero_predictor.npy"
    # )
    alpha: float = 0.5  # blending factor for regularized reward

    @override
    def inputs_spec(
        self, *, batch_size: int = 1
    ) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct(
            [batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32
        )
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct(
                    [batch_size, self.max_token_len], jnp.int32
                ),
                tokenized_prompt_mask=jax.ShapeDtypeStruct(
                    [batch_size, self.max_token_len], bool
                ),
            )
        action_spec = jax.ShapeDtypeStruct(
            [batch_size, self.action_horizon, self.action_dim], jnp.float32
        )

        return observation_spec, action_spec

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

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns a filter selecting params to freeze.

        We want to TRAIN only:
          - _diffusion_transformer
          - _action_embedder
          - _video_encoder

        So we FREEZE everything else: Not(Any(trainable_filters)).
        """
        dit_filter = nnx_utils.PathRegex("_diffusion_transformer.*")
        ae_filter = nnx_utils.PathRegex("_action_embedder.*")
        ve_filter = nnx_utils.PathRegex("_video_encoder.*")
        trainable = nnx.Any(dit_filter, ae_filter, ve_filter)
        return nnx.Not(trainable)


# class Pi0Predictor(Pi0):
class Pi0Predictor(_model.BaseModel):
    def __init__(self, config: Pi0PredictorConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self._eps = config.eps
        self._primary_image_key = config.primary_image_key
        self._ignore_image_keys = config.ignore_image_keys

        self._horizon = config.horizon
        self._history_len = config.history_len
        self._num_denoise_steps = config.num_denoise_steps
        # self._baseline_embedding = nnx.Variable(
        #     jnp.load(config.baseline_embedding_path)
        # )
        # self._goal_embedding = nnx.Variable(jnp.load(config.goal_embedding_path))
        self._alpha = config.alpha

        self._pretrain = config.pretrain

        vae, vae_params = FlaxAutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse", from_pt=True, dtype=jnp.float32
        )
        self._vae = vae
        self._vae_params = jax.tree_util.tree_map(lambda x: nnx.Param(x), vae_params)
        self._vae_scaling_factor = vae.config.scaling_factor

        # self._video_encoder = VideoTransformer(
        #     in_channel=config.in_channel,
        #     dim=config.hidden_size,
        #     depth=config.video_depth,
        #     num_heads=config.num_heads,
        #     rngs=rngs,
        # )
        self._video_encoder = MultiViewVideoTransformer(
            in_channel=config.in_channel,
            dim=config.hidden_size,
            depth=config.video_depth,
            num_heads=config.num_heads,
            num_cross_attn_layers=config.num_cross_view_layers,
            rngs=rngs,
        )
        self._action_embedder = nnx.Linear(32, config.hidden_size, rngs=rngs)
        self._diffusion_transformer = DiffusionTransformer(
            in_channel=config.in_channel,
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            n_layers=config.num_layers,
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

    def embed_inputs_multi_view(
        self, observation: _model.Observation, train: bool, rng: at.KeyArrayLike
    ) -> tuple[list[jnp.ndarray], jnp.ndarray]:
        """
        Encode all available camera views and return embeddings.

        Returns:
            Tuple of:
                - List of embeddings for all views, each [B, T, N, C]
                - Primary view embedding for loss computation [B, T, N, C]
        """
        all_embeddings = []
        primary_embedding = None

        # Sort keys to ensure consistent ordering (primary key first)
        image_keys = sorted(
            observation.images.keys(), key=lambda k: (k != self._primary_image_key, k)
        )

        for key in image_keys:
            if key in self._ignore_image_keys:
                continue
            images = observation.images[key]
            rng, rng_vae = jax.random.split(rng)
            embedding = self._encode_with_vae(images, rng_vae)

            # Reshape:  [B, C, H', W'] -> [B, N, C] where N = H' * W'
            b = embedding.shape[0]
            embedding = embedding.reshape(b, -1, embedding.shape[-1])  # [B, N, C]

            all_embeddings.append(embedding)

            if key == self._primary_image_key:
                primary_embedding = embedding

        if primary_embedding is None:
            primary_embedding = all_embeddings[0]

        return all_embeddings, primary_embedding

    # def embed_inputs(
    #     self, observation: _model.Observation, train: bool, rng: at.KeyArrayLike
    # ) -> at.Float[at.Array, "*b s emb"]:
    #     """Encode images using SD VAE instead of PaliGemma."""
    #     images = observation.images[self._image_key]
    #     return self._encode_with_vae(images, rng)

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b"]:
        b, t, _ = actions.shape

        # Preprocess and embed observations
        rng, rng_preprocess, rng_embed = jax.random.split(rng, 3)
        obs_p = _model.preprocess_observation(
            rng_preprocess,
            observation,
            train=train,
            image_keys=list(observation.images.keys()),
        )
        # Multi-view path:  encode all views
        all_view_embeddings, primary_embedding = self.embed_inputs_multi_view(
            obs_p, train=train, rng=rng_embed
        )

        # Reshape embeddings to [B, T, N, C] format
        # Assuming temporal dimension is handled in observation structure
        # Each embedding is [B*T, N, C], reshape to [B, T, N, C]
        view_embeddings_reshaped = []
        for emb in all_view_embeddings:
            emb_reshaped = jnp.reshape(emb, (b, t, -1, 4))
            view_embeddings_reshaped.append(emb_reshaped)

        primary_embedding = jnp.reshape(primary_embedding, (b, t, -1, 4))

        # Extract history portion for each view
        history_views = [v[:, : self._history_len] for v in view_embeddings_reshaped]

        # Get history features using multi-view encoder
        history_features = self._video_encoder(history_views)

        # Future is only from primary view (for loss computation)
        future = primary_embedding[:, self._history_len :]

        # embeddings = self.embed_inputs(obs_p, train=train, rng=rng_embed)
        # embeddings = jnp.reshape(embeddings, (b, t, -1, 4))

        # history = embeddings[:, : self._history_len]
        # future = embeddings[:, self._history_len :]
        # history_features = self._video_encoder(history)

        if not self._pretrain:
            action_window = actions[:, self._history_len :]
            action_features = self._action_embedder(action_window)
        else:
            action_features = None

        rng, rng_noise, rng_t = jax.random.split(rng, 3)
        noise = jax.random.normal(rng_noise, future.shape)
        timestep = jax.random.uniform(rng_t, (b,), minval=0.02, maxval=0.98)

        t = timestep[:, None, None, None]
        x_t = (1 - t) * noise + t * future
        target_velocity = future - noise

        v_pred = self._diffusion_transformer(
            x_t,
            history_features,
            action_tokens=action_features,
            timestep=timestep,
        )

        loss = jnp.mean(jnp.square(v_pred - target_velocity))
        return loss

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> _model.Actions:
        raise NotImplementedError("Action sampling not implemented for Pi0Predictor.")

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
        """Predict future states using flow matching inference.

        Uses Heun's method when num_denoise_steps >= 10, otherwise Euler.
        """
        b, t, _ = actions.shape

        obs_p = _model.preprocess_observation(
            rng, observation, train=False, image_keys=list(observation.images.keys())
        )
        rng, rng_embed = jax.random.split(rng)

        all_view_embeddings, primary_embedding = self.embed_inputs_multi_view(
            obs_p, train=False, rng=rng_embed
        )

        view_embeddings_reshaped = []
        for emb in all_view_embeddings:
            emb_reshaped = jnp.reshape(emb, (b, t, -1, 4))
            view_embeddings_reshaped.append(emb_reshaped)

        primary_embedding = jnp.reshape(primary_embedding, (b, t, -1, 4))
        _, _, N, C = primary_embedding.shape

        history_views = [v[:, : self._history_len] for v in view_embeddings_reshaped]
        history_features = self._video_encoder(history_views)

        # embeddings = self.embed_inputs(obs_p, train=False, rng=rng_embed)
        # embeddings = jnp.reshape(embeddings, (b, t, -1, 4))

        # _, _, N, C = embeddings.shape
        # history = embeddings[:, : self._history_len]
        # history_features = self._video_encoder(history)

        if not self._pretrain:
            action_window = actions[
                :, self._history_len : self._history_len + self._horizon
            ]
            action_features = self._action_embedder(action_window)
        else:
            action_features = None

        rng, rng_init = jax.random.split(rng)
        x_t = jax.random.normal(rng_init, (b, self._horizon, N, C))

        num_steps = self._num_denoise_steps
        step_size = 1.0 / num_steps
        timesteps = jnp.linspace(0.0, 1.0, num_steps, endpoint=False)

        # Choose solver based on num_denoise_steps
        use_heun = num_steps >= 10

        def euler_step(x_curr, timestep):
            """First-order Euler integration."""
            t_batch = jnp.full((b,), timestep)
            v_pred = self._diffusion_transformer(
                x_curr,
                history_features,
                action_tokens=action_features,
                timestep=t_batch,
            )
            x_next = x_curr + v_pred * step_size
            return x_next, None

        def heun_step(x_curr, timestep):
            """Second-order Heun (predictor-corrector) integration."""
            t_batch = jnp.full((b,), timestep)
            t_next_batch = jnp.full((b,), jnp.minimum(timestep + step_size, 1.0))

            # Predictor:  Euler step
            k1 = self._diffusion_transformer(
                x_curr,
                history_features,
                action_tokens=action_features,
                timestep=t_batch,
            )
            x_pred = x_curr + k1 * step_size

            # Corrector: average slopes
            k2 = self._diffusion_transformer(
                x_pred,
                history_features,
                action_tokens=action_features,
                timestep=t_next_batch,
            )
            x_next = x_curr + (k1 + k2) * (step_size / 2.0)

            return x_next, None

        # Select solver
        step_fn = heun_step if use_heun else euler_step

        predicted_latents, _ = jax.lax.scan(step_fn, x_t, timesteps)

        if decode_to_images:
            return self._decode_with_vae(predicted_latents)
        return predicted_latents

    # def compute_regularized_reward(self, state_embedding: jnp.ndarray) -> jnp.ndarray:
    #     s = state_embedding
    #     g = self._goal_embedding.value
    #     b = self._baseline_embedding.value

    #     direction_vector = g - b
    #     direction_vector_norm_sq = jnp.sum(direction_vector**2)

    #     s_minus_b = s - b
    #     projection_scalar = jnp.dot(s_minus_b, direction_vector) / jnp.maximum(
    #         direction_vector_norm_sq, 1e-6
    #     )
    #     projected_s = b + projection_scalar * direction_vector

    #     blended_embedding = (1 - self._alpha) * s + self._alpha * projected_s

    #     reward = 1.0 - 0.5 * jnp.sum((blended_embedding - g) ** 2)
    #     return reward
