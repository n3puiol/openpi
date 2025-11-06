import dataclasses

import einops
from pyparsing import Callable
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
    num_layers: int = 6
    eps: float = 1e-5
    image_key: str = "base_0_rgb"

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

    def img_encode(
        self, images: at.Float[at.Array, "*b h w c"]
    ) -> at.Float[at.Array, "*b s emb"]:
        return self.PaliGemma.img(images, train=False)[0]

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
        horizon = t // 2
        # Preprocess observation and encode images
        observation = _model.preprocess_observation(
            rng, observation, train=train, image_keys=list(observation.images.keys())
        )
        image_embeddings = self.img_encode(
            observation.images[self._image_key]
        )  # (b*t, 256, 2048)
        _, s, p = image_embeddings.shape
        image_embeddings = jnp.reshape(image_embeddings, (b, t, s, p))

        # Split into history and future segments
        lc_his = image_embeddings[:, :horizon]  # (b, horizon, 256, 2048)
        x_prior = lc_his[:, -1:, :]  # (b, 1, 256, 2048)
        lc_next = image_embeddings[:, horizon:]  # (b, horizon, 256, 2048)
        a_next = actions[:, horizon:]  # (b, horizon, 7)

        # Build residual target and drift term
        res = jnp.concatenate([x_prior, lc_next], axis=1)
        res = jnp.diff(res, axis=1)
        c_res = -res  # Multiply by drift term to guide diffusion process

        # Split RNG to avoid correlation between timestep sampling and noise
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

        # Losses
        loss = jnp.mean((y_pred - c_res) ** 2)
        aux_loss = jnp.mean((y_pred_tmp - c_res) ** 2)
        return loss + 0.1 * aux_loss

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        energy_fn: (
            Callable[[jnp.ndarray], float] | None
        ) = None,  # NEW: e.g., lambda A: -reward_model(A)
        guidance_scale: float = 1.0,  # NEW: s >1 for stronger guidance
        guidance_cov_scale: float = 5.0,  # NEW: sigma for lambda_t = guidance_cov_scale**2 * (1 - time)
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(
            rng, (batch_size, self.action_horizon, self.action_dim)
        )

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(
                prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1]
            )
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate(
                [prefix_attn_mask, suffix_attn_mask], axis=-1
            )
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = (
                jnp.sum(prefix_mask, axis=-1)[:, None]
                + jnp.cumsum(suffix_mask, axis=-1)
                - 1
            )

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            if energy_fn is not None:
                # Estimate hat_x1 (predicted clean actions) ~ x_t + |dt| * v_t (affine approx)
                dt_abs = jnp.abs(dt)  # since dt negative
                hat_x1 = x_t + dt_abs * v_t

                # Compute gradient of J w.r.t. hat_x1
                grad_J = jax.grad(energy_fn)(hat_x1)  # Shape: [batch, horizon, dim]

                # Approximate guidance: g_t = - lambda_t * grad_J (cov approx with scalar schedule)
                lambda_t = (guidance_cov_scale**2) * (
                    1 - time
                )  # Decays as time ->0 (less uncertainty)
                g_t = (
                    -lambda_t[..., None, None] * grad_J
                )  # Broadcast lambda_t if scalar

                # Apply scaled guidance
                v_t = v_t + guidance_scale * g_t

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    def get_fused_embedding(self, image_token_embeddings, obs, name):
        fused_embeddings = []

        for i in range(image_token_embeddings.shape[1]):
            emb = image_token_embeddings[:, i, :]  # (1, 256, 2048)
            input_mask = []
            ar_mask = []
            token_embeddings = []
            token_embeddings.append(emb)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=emb.shape[1],
                )
            )
            ar_mask.append(0 * input_mask[-1])

            tokenized_inputs_embeddings = self.PaliGemma.llm(
                obs.tokenized_prompt, embed_only=True
            )

            token_embeddings.append(tokenized_inputs_embeddings)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask.append(obs.token_ar_mask)

            input_token_embeddings = jnp.concatenate(token_embeddings, axis=1)
            input_mask = jnp.concatenate(input_mask, axis=1)
            ar_mask = jnp.concatenate(ar_mask, axis=1)

            attn_mask = make_attn_mask(input_mask, ar_mask)

            fused_sequence_embeddings, _, _ = self.PaliGemma.llm(
                embedded_prefix=input_token_embeddings,
                mask=attn_mask,
                return_prelogits=True,
            )
            mask_expanded = jnp.expand_dims(input_mask, axis=-1)
            summed_embeddings = jnp.sum(
                fused_sequence_embeddings * mask_expanded, axis=1
            )
            num_valid_tokens = jnp.sum(input_mask, axis=1, keepdims=True)
            pooled_fused_embedding = summed_embeddings / jnp.maximum(
                num_valid_tokens, 1
            )
            fused_embeddings.append(pooled_fused_embedding)

        return fused_embeddings

    def predict_future(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ):
        b, t, _ = actions.shape
        # Preprocess observation and encode images
        observation = _model.preprocess_observation(
            rng, observation, train=train, image_keys=list(observation.images.keys())
        )
        image_embeddings = self.img_encode(
            observation.images[self._image_key]
        )  # (b*t, 256, 2048)
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

        past_fused_embedding = self.get_fused_embedding(
            lc_his, observation, self._image_key
        )

        future_fused_embedding = self.get_fused_embedding(
            y_pred, observation, self._image_key
        )

        return past_fused_embedding, future_fused_embedding
