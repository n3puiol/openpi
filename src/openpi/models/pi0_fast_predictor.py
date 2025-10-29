import dataclasses
from openpi.models.pi0_fast import Pi0FAST, Pi0FASTConfig
import openpi.shared.nnx_utils as nnx_utils
from openpi.models import model as _model
from openpi.shared import array_typing as at
from typing_extensions import override

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models.dit import DiffusionTransformer


@dataclasses.dataclass(frozen=True)
class Pi0FASTPredictorConfig(Pi0FASTConfig):
    in_channel: int = 2048
    hidden_size: int = 1024
    num_heads: int = 8
    num_layers: int = 6
    eps: float = 1e-5
    image_key: str = "base_0_rgb"
    
    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0_FAST_PREDICTOR

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0FASTPredictor":
        # Construct Rngs with explicit param and dropout keys
        k_params, k_dropout = jax.random.split(rng)  # type: ignore[arg-type]
        rngs = nnx.Rngs(params=k_params, dropout=k_dropout)
        return Pi0FASTPredictor(self, rngs=rngs)
    
    @override
    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        return nnx.All(nnx_utils.PathRegex("PaliGemma.*"))


class Pi0FASTPredictor(Pi0FAST):
    def __init__(self, config: Pi0FASTPredictorConfig, rngs: nnx.Rngs):
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
    
    def img_encode(self, images: at.Float[at.Array, "*b h w c"]) -> at.Float[at.Array, "*b s emb"]:
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
        image_embeddings = self.img_encode(observation.images[self._image_key])  # (b*t, 256, 2048)
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
        y_pred, y_pred_tmp = self._diffusion_transformer(x_noisy, lc_his, a_next, timestep)

        # Losses
        loss = jnp.mean((y_pred - c_res) ** 2)
        aux_loss = jnp.mean((y_pred_tmp - c_res) ** 2)
        return loss + 0.1 * aux_loss
    
    def forward(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah ac"]:
        b, t, _ = actions.shape
        # Preprocess observation and encode images
        observation = _model.preprocess_observation(
            rng, observation, train=train, image_keys=list(observation.images.keys())
        )
        image_embeddings = self.img_encode(observation.images[self._image_key])  # (b*t, 256, 2048)
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
        return y_pred
