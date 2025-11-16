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

# TODO (Training): Train in similar fashion as V-Jepa2 , ie. train to predict 1 future embedding block as well as predict from predicted embedding to future embedding (like masked modeling)


@dataclasses.dataclass(frozen=True)
class Pi0PredictorConfig(Pi0Config):
    in_channel: int = 2048
    hidden_size: int = 1024
    num_heads: int = 8
    num_layers: int = 8
    eps: float = 1e-5
    image_key: str = "base_0_rgb"
    rollout_factor: float = 1.0
    training_steps: int = 6 # number of steps for training (teacher forcing (2) + rollout steps (steps - 2))
    
    # Reward estimation embeddings
    baseline_embedding_path: str = (
        "reward_estimation_embeddings/baseline_embedding_pi0_libero_predictor.npy"
    )
    goal_embedding_path: str = (
        "reward_estimation_embeddings/goal_embedding_pi0_libero_predictor.npy"
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
        self._training_steps = config.training_steps
        # self._baseline_embedding_path = jnp.load(config.baseline_embedding_path)
        # self._goal_embedding_path = jnp.load(config.goal_embedding_path)
        # self._alpha = config.alpha

        self._diffusion_transformer = DiffusionTransformer(
            in_channel=config.in_channel,
            dim=config.hidden_size,
            num_heads=config.num_heads,
            n_layers=config.num_layers,
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
        observation = _model.preprocess_observation(
            rng, observation, train=train, image_keys=list(observation.images.keys())
        )
        return self.PaliGemma.img(observation.images[self._image_key], train=False)[0]
    
    # @override
    # def compute_loss(
    #     self,
    #     rng: at.KeyArrayLike,
    #     observation: _model.Observation,
    #     actions: _model.Actions,
    #     *,
    #     train: bool = False,
    # ) -> at.Float[at.Array, "*b ah"]:
    #     # horizon is 10, so we use first 5 for teacher forcing, last 5 for rollout
    #     b, t, _ = actions.shape
    #     # horizon = t // 2
    #     steps = 6
    #     horizon = t // steps
    #     image_embeddings = self.embed_inputs(observation, train=train, rng=rng)
    #     _, s, p = image_embeddings.shape
    #     image_embeddings = jnp.reshape(image_embeddings, (b, t, s, p))

    #     # teacher forcing loss
        
    #     # Split into history and future segments
    #     lc_his = image_embeddings[:, :horizon]  # (b, horizon, 256, 2048)
    #     print("lc_his shape:", lc_his.shape)
    #     x_prior = lc_his[:, -1:, :]  # (b, 1, 256, 2048)
    #     lc_next = image_embeddings[:, horizon:horizon*2]  # (b, horizon, 256, 2048)
    #     a_next = actions[:, horizon:horizon*2]  # (b, horizon, 7)

    #     # Build residual target and drift term
    #     res = jnp.concatenate([x_prior, lc_next], axis=1)
    #     res = jnp.diff(res, axis=1)
    #     c_res = -res  # Multiply by drift term to guide diffusion process

    #     # Split RNG to avoid correlation between timestep sampling and noise
    #     rng_t, rng_n = jax.random.split(rng)
    #     timestep = (
    #         jax.random.uniform(rng_t, shape=(c_res.shape[0],), minval=0.0, maxval=1.0)
    #         * (1.0 - self._eps)
    #         + self._eps
    #     )
    #     noise = jax.random.normal(rng_n, shape=c_res.shape)

    #     # Forward through diffusion transformer
    #     x_noisy = self.add_noise(res, noise, timestep, c_res)
    #     y_pred, y_pred_tmp = self._diffusion_transformer(
    #         x_noisy, lc_his, a_next, timestep
    #     )
    #     print("y_pred shape:", y_pred.shape)
        
    #     # Losses
    #     loss = jnp.mean((y_pred - c_res) ** 2)
    #     aux_loss = jnp.mean((y_pred_tmp - c_res) ** 2)
    #     teacher_forcing_loss = jnp.mean(loss + 0.1 * aux_loss)
        
    #     # rollout loss
    #     rollout_loss = 0.0
    #     for i in range(2, steps):
    #         lc_his = y_pred
    #         x_prior = lc_his[:, -1:, :]
    #         lc_next = image_embeddings[:, horizon*i:horizon*(i+1)]
    #         a_next = actions[:, horizon*i:horizon*(i+1)]
            
    #         # Build residual target and drift term
    #         res = jnp.concatenate([x_prior, lc_next], axis=1)
    #         res = jnp.diff(res, axis=1)
    #         c_res = -res  # Multiply by drift term to guide diffusion process

    #         # Split RNG to avoid correlation between timestep sampling and noise
    #         rng_t, rng_n = jax.random.split(rng)
    #         timestep = (
    #             jax.random.uniform(rng_t, shape=(c_res.shape[0],), minval=0.0, maxval=1.0)
    #             * (1.0 - self._eps)
    #             + self._eps
    #         )
    #         noise = jax.random.normal(rng_n, shape=c_res.shape)

    #         # Forward through diffusion transformer
    #         x_noisy = self.add_noise(res, noise, timestep, c_res)
    #         y_pred, y_pred_tmp = self._diffusion_transformer(
    #             x_noisy, lc_his, a_next, timestep
    #         )
            
    #         # Losses
    #         loss = jnp.mean((y_pred - c_res) ** 2)
    #         aux_loss = jnp.mean((y_pred_tmp - c_res) ** 2)
    #         rollout_loss += jnp.mean(loss + 0.1 * aux_loss)
    #     rollout_loss = rollout_loss / (steps - 2)
        
    #     return teacher_forcing_loss + rollout_loss
    
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
        steps = self._training_steps
        horizon = t // steps
        
        image_embeddings = self.embed_inputs(observation, train=train, rng=rng)
        _, s, p = image_embeddings.shape
        image_embeddings = jnp.reshape(image_embeddings, (b, t, s, p))

        def compute_step_loss(lc_his, lc_next, a_next, rng):
            """Compute loss for a single prediction step."""
            x_prior = lc_his[:, -1:, :]  # (b, 1, 256, 2048)
            
            # Build residual target and drift term
            res = jnp.concatenate([x_prior, lc_next], axis=1)
            res = jnp.diff(res, axis=1)
            c_res = -res
            
            # Split RNG for timestep and noise
            rng_t, rng_n = jax.random.split(rng)
            timestep = (
                jax.random.uniform(rng_t, shape=(c_res.shape[0],), minval=0.0, maxval=1.0)
                * (1.0 - self._eps)
                + self._eps
            )
            noise = jax.random.normal(rng_n, shape=c_res.shape)
            
            # Forward through diffusion transformer
            x_noisy = self.add_noise(res, noise, timestep, c_res)
            y_pred, y_pred_tmp = self._diffusion_transformer(
                x_noisy, lc_his, a_next, timestep
            )
            
            # Compute losses
            loss = jnp.mean((y_pred - c_res) ** 2)
            aux_loss = jnp.mean((y_pred_tmp - c_res) ** 2)
            total_loss = loss + 0.1 * aux_loss
            
            return total_loss, y_pred

        # Teacher forcing loss
        lc_his = image_embeddings[:, :horizon]
        lc_next = image_embeddings[:, horizon:horizon*2]
        a_next = actions[:, horizon:horizon*2]
        
        teacher_forcing_loss, y_pred = compute_step_loss(lc_his, lc_next, a_next, rng)
        
        # Rollout loss
        rollout_loss = 0.0
        for i in range(2, steps):
            lc_his = y_pred
            lc_next = image_embeddings[:, horizon*i:horizon*(i+1)]
            a_next = actions[:, horizon*i:horizon*(i+1)]
            
            step_loss, y_pred = compute_step_loss(lc_his, lc_next, a_next, rng)
            rollout_loss += step_loss
        
        rollout_loss = rollout_loss / (steps - 2)
        
        return teacher_forcing_loss + self._rollout_factor * rollout_loss

    # @override
    # def sample_actions(
    #     self,
    #     rng: at.KeyArrayLike,
    #     observation: _model.Observation,
    #     *,
    #     num_steps: int | at.Int[at.Array, ""] = 10,
    #     energy_fn: (
    #         Callable[[jnp.ndarray], float] | None
    #     ) = None,  # NEW: e.g., lambda A: -reward_model(A)
    #     guidance_scale: float = 1.0,  # NEW: s >1 for stronger guidance
    #     guidance_cov_scale: float = 5.0,  # NEW: sigma for lambda_t = guidance_cov_scale**2 * (1 - time)
    # ) -> _model.Actions:
    #     observation = _model.preprocess_observation(None, observation, train=False)
    #     # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
    #     # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
    #     dt = -1.0 / num_steps
    #     batch_size = observation.state.shape[0]
    #     noise = jax.random.normal(
    #         rng, (batch_size, self.action_horizon, self.action_dim)
    #     )

    #     # first fill KV cache with a forward pass of the prefix
    #     prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    #     prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    #     positions = jnp.cumsum(prefix_mask, axis=1) - 1
    #     _, kv_cache = self.PaliGemma.llm(
    #         [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
    #     )

    #     def step(carry):
    #         x_t, time = carry
    #         suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
    #             observation, x_t, jnp.broadcast_to(time, batch_size)
    #         )
    #         # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
    #         # other
    #         suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
    #         # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
    #         # prefix tokens
    #         prefix_attn_mask = einops.repeat(
    #             prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1]
    #         )
    #         # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
    #         # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
    #         full_attn_mask = jnp.concatenate(
    #             [prefix_attn_mask, suffix_attn_mask], axis=-1
    #         )
    #         assert full_attn_mask.shape == (
    #             batch_size,
    #             suffix_tokens.shape[1],
    #             prefix_tokens.shape[1] + suffix_tokens.shape[1],
    #         )
    #         # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
    #         positions = (
    #             jnp.sum(prefix_mask, axis=-1)[:, None]
    #             + jnp.cumsum(suffix_mask, axis=-1)
    #             - 1
    #         )

    #         (prefix_out, suffix_out), _ = self.PaliGemma.llm(
    #             [None, suffix_tokens],
    #             mask=full_attn_mask,
    #             positions=positions,
    #             kv_cache=kv_cache,
    #         )
    #         assert prefix_out is None
    #         v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

    #         if energy_fn is not None:
    #             # Estimate hat_x1 (predicted clean actions) ~ x_t + |dt| * v_t (affine approx)
    #             dt_abs = jnp.abs(dt)  # since dt negative
    #             hat_x1 = x_t + dt_abs * v_t

    #             # Compute gradient of J w.r.t. hat_x1
    #             grad_J = jax.grad(energy_fn)(hat_x1)  # Shape: [batch, horizon, dim]

    #             # Approximate guidance: g_t = - lambda_t * grad_J (cov approx with scalar schedule)
    #             lambda_t = (guidance_cov_scale**2) * (
    #                 1 - time
    #             )  # Decays as time ->0 (less uncertainty)
    #             g_t = (
    #                 -lambda_t[..., None, None] * grad_J
    #             )  # Broadcast lambda_t if scalar

    #             # Apply scaled guidance
    #             v_t = v_t + guidance_scale * g_t

    #         return x_t + dt * v_t, time + dt

    #     def cond(carry):
    #         x_t, time = carry
    #         # robust to floating-point error
    #         return time >= -dt / 2

    #     x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
    #     return x_0

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
        image_embeddings = self.embed_inputs(observation, train=train, rng=rng)
        _, s, p = image_embeddings.shape
        image_embeddings = jnp.reshape(image_embeddings, (b, t, s, p))

        # Split into history and future segments
        lc_his = image_embeddings
        x_prior = lc_his[:, -1:, :]  # (b, 1, 256, 2048)
        a_next = actions  # (b, horizon, 7)

        # Split RNG to avoid correlation between timestep sampling and noise
        rng_t, rng_n = jax.random.split(rng)
        timestep = (
            jax.random.uniform(rng_t, shape=(x_prior.shape[0],), minval=0.0, maxval=1.0)
            * (1.0 - self._eps)
            + self._eps
        )
        x_noisy = jax.random.normal(rng_n, shape=lc_his.shape)

        # Forward through diffusion transformer
        y_pred, _ = self._diffusion_transformer(x_noisy, lc_his, a_next, timestep)

        past_fused_embedding = self.get_fused_embedding(lc_his, observation)

        future_fused_embedding = self.get_fused_embedding(y_pred, observation)

        return past_fused_embedding, future_fused_embedding

    def compute_regularized_reward(self, state_embedding: jnp.ndarray) -> jnp.ndarray:
        s = state_embedding
        g = self._goal_embedding_path
        b = self._baseline_embedding_path

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

    def reward_model(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
    ):
        # TODO 1: get reward from recent past reward with sliding window
        # TODO 2: store reward history
        # TODO 3: slide window on predicted future embeddings and get future reward
        # TODO 4: normalize reward?
        # TODO 5: determine if reward is increasing or decreasing
        past_embedding, future_embedding = self.predict_future(
            rng, observation, actions, train=False
        )
        past_reward = self.compute_regularized_reward(past_embedding)
        future_reward = self.compute_regularized_reward(future_embedding)
        
        return

        