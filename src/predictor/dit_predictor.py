import flax.nnx as nnx
import jax.numpy as jnp
import jax

from predictor.diffusion_transformer import DiffusionTransformer
import openpi.policies.policy as _policy
from openpi.models import model as _model
from openpi.shared import array_typing as at


class Predictor(nnx.Module):
    eps: float = 1e-3

    def __init__(
        self,
        policy: _policy.Policy,
        in_channel: int = 2048,
        hidden_size: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        *,
        rngs: nnx.Rngs,
    ):
        self._policy = policy
        self._rngs = rngs

        self._diffusion_transformer = DiffusionTransformer(
            in_channel=in_channel,
            dim=hidden_size,
            num_heads=num_heads,
            n_layers=num_layers,
            rngs=rngs,
        )
    
    def get_siglip_encoding(self, images: _model.HorizonImages) -> at.Float[at.Array, "*b s emb"]:
        images = self.preprocess_images(images)
        print(f"images.shape: {images.shape}")
        image_token_embeddings = self._policy.img_encode(images)
        return image_token_embeddings

    def preprocess_images(self, images: _model.HorizonImages) -> at.Float[at.Array, "*b h w c"]:
        images = jnp.reshape(images, (-1, *images.shape[2:])) # (B*t, H, W, C)
        # Normalize images to [-1, 1]
        images = (images.astype(jnp.float32) / 127.5) - 1.0
        # TODO: add augmentation
        return images

    def add_noise(
        self, x: at.Array, noise: at.Array, timestep: at.Array, c: at.Array
    ) -> at.Array:
        time = timestep.reshape(c.shape[0], *((1,) * (len(c.shape) - 1)))
        x_noisy = x + c * time + time * noise
        return x_noisy

    def compute_loss(
        self,
        images: _model.HorizonImages,
        actions: _model.Actions,
    ):
        # TODO: explore using fast tokenizer for actions
        b, t, h, w, c = images.shape
        horizon = t//2
        
        image_embeddings = self.get_siglip_encoding(images)
        _, s, p = image_embeddings.shape
        print(f"image_embeddings.shape: {image_embeddings.shape}")
        image_embeddings = jnp.reshape(image_embeddings, (b, t, s, p))
        print(f"image_embeddings.shape: {image_embeddings.shape}")

        lc_his = image_embeddings[:, :horizon]  # (b, horizon, 256, 2048)
        print(f"lc_his.shape: {lc_his.shape}")
        x_prior = lc_his[:, -1:, :]  # (b, 1, 256, 2048)
        print(f"x_prior.shape: {x_prior.shape}")
        lc_next = image_embeddings[:, horizon:]  # (b, horizon, 256, 2048)
        print(f"lc_next.shape: {lc_next.shape}")

        res = jnp.concatenate([x_prior, lc_next], axis=1)
        print(f"res.shape: {res.shape}")
        res = jnp.diff(res, axis=1) * 1
        print(f"res.shape: {res.shape}")
        c_res = -1 * res  # Multiply by drift term to guide diffusion process
        print(f"c_res.shape: {c_res.shape}")

        timestep = (
            jax.random.uniform(
                self._rngs(), shape=(c_res.shape[0],), minval=0.0, maxval=1.0
            )
            * (1.0 - self.eps)
            + self.eps
        )
        print(f"timestep.shape: {timestep.shape}")

        noise = jax.random.normal(self._rngs(), shape=c_res.shape)
        print(f"noise.shape: {noise.shape}")

        x_noisy = self.add_noise(
            res,
            noise,
            timestep,
            c_res,
        )
        print(f"x_noisy.shape: {x_noisy.shape}")

        y_pred, y_pred_tmp = self._diffusion_transformer(
            x_noisy, lc_his, actions, timestep
        )
        print(f"y_pred.shape: {y_pred.shape}")
        print(f"y_pred_tmp.shape: {y_pred_tmp.shape}")


if __name__ == "__main__":
    from openpi.policies import policy_config as _policy_config
    from openpi.shared import download
    from openpi.training import config as _config

    cpu_device = jax.devices('cpu')[0]
    print("CPU device:", cpu_device)
    with jax.default_device(cpu_device):
        config = _config.get_config("pi0_fast_libero_predictor")
        checkpoint_dir = download.maybe_download(
            "/scratch/s5649552/.cache/openpi/openpi-assets/checkpoints/pi0_fast_libero_predictor"
        )

        policy = _policy_config.create_trained_policy(config, checkpoint_dir)

        predictor = Predictor(policy=policy, rngs=nnx.Rngs(0, params=0))

        key = jax.random.PRNGKey(0)
        rngs = nnx.Rngs(params=key, dropout=key)
        actions = jax.random.normal(key, (8, 4, 7))
        imgs = jax.random.normal(key, (8, 4, 224, 224, 3))

        predictor.compute_loss(imgs, actions)
