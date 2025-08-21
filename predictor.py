import math
import flax.nnx as nnx
import jax.numpy as jnp
import jax

import openpi.policies.policy as _policy
from openpi.models import model as _model
from openpi.shared import array_typing as at


class Predictor(nnx.Module):
    eps: float = 1e-3

    def __init__(self, policy: _policy.Policy, *, rngs: nnx.Rngs):
        self._policy = policy
        self._rngs = rngs

        self._diffusion_transformer = DiffusionTransformer(rngs=self._rngs)

    def get_siglip_encoding(self, observation: dict) -> tuple:
        token_embeddings, _, _ = self._policy.get_encoding(observation)
        print(f"token_embeddings.shape: {token_embeddings.shape}")
        embedding_img = token_embeddings[:, :256]
        embedding_wrist_img = token_embeddings[:, 256:512]
        embedding_text = token_embeddings[:, 512:]
        return embedding_img, embedding_wrist_img, embedding_text

    def add_noise(
        self, x: at.Array, noise: at.Array, timestep: at.Array, c: at.Array
    ) -> at.Array:
        time = timestep.reshape(c.shape[0], *((1,) * (len(c.shape) - 1)))
        x_noisy = x + c * time + time * noise
        return x_noisy

    def compute_loss(
        self,
        past_observations: dict,
        future_observations: dict,
        actions: _model.Actions,
    ):
        # Get past and future SigLip encodings (img: (B, 256, 2048), wrist: (B, 256, 2048), text: (B, x, 2048))
        past_encoding_img, _, _ = self.get_siglip_encoding(past_observations)
        print(f"past_encoding_img.shape: {past_encoding_img.shape}")
        future_encoding_img, _, _ = self.get_siglip_encoding(future_observations)
        print(f"future_encoding_img.shape: {future_encoding_img.shape}")

        residuals_encoding_img = jnp.concatenate(
            [past_encoding_img, future_encoding_img], axis=1
        )
        print(f"residuals_encoding_img.shape: {residuals_encoding_img.shape}")
        residuals_encoding_img = jnp.diff(residuals_encoding_img, axis=1)
        print(f"residuals_encoding_img.shape: {residuals_encoding_img.shape}")
        c_residuals_encoding_img = (
            -1 * residuals_encoding_img
        )  # Multiply by drift term to guide diffusion process
        print(f"c_residuals_encoding_img.shape: {c_residuals_encoding_img.shape}")

        rng = self._rngs()
        timestep = (
            jax.random.uniform(
                rng, shape=(c_residuals_encoding_img.shape[0],), minval=0.0, maxval=1.0
            )
            * (1.0 - self.eps)
            + self.eps
        )
        print(f"timestep.shape: {timestep.shape}")

        rng = self._rngs()
        noise_encoding_img = jax.random.normal(
            rng, shape=c_residuals_encoding_img.shape
        )
        print(f"noise_encoding_img.shape: {noise_encoding_img.shape}")

        x_noisy = self.add_noise(
            c_residuals_encoding_img,
            noise_encoding_img,
            timestep,
            c_residuals_encoding_img,
        )
        print(f"x_noisy.shape: {x_noisy.shape}")

        x = self._diffusion_transformer(x_noisy, past_encoding_img, actions, timestep)
        print(f"x.shape: {x.shape}")


class DiffusionTransformer(nnx.Module):
    def __init__(
        self,
        in_channel: int = 3,
        hidden_dim: int = 512,
        img_size: int = 256,
        tube_size: int = 8,
        *,
        rngs: nnx.Rngs,
    ):
        self._rngs = rngs
        self._in_channel = in_channel
        self._hidden_dim = hidden_dim
        self._img_size = img_size
        self._tube_size = tube_size

        self.x_embedder = nnx.Linear(
            self._in_channel, self._hidden_dim, rngs=self._rngs
        )
        self.timestep_embedder = TimestepEmbedder(self._hidden_dim, rngs=self._rngs)
        self.action_embedder = ActionEmbedder(self._hidden_dim, rngs=self._rngs)

        self.pos_embed = jnp.zeros(
            (1, self._img_size * self._tube_size, self._hidden_dim)
        )

    def __call__(
        self,
        x_noisy: at.Array,
        past_encodings: at.Array,
        past_actions: _model.Actions,
        timestep: at.Array,
    ):
        b, t, n, c = x_noisy.shape
        x = (
            self.x_embedder(x_noisy).reshape(b, t * n, c) + self.pos_embed
        )  # b, (t*n), c
        print(f"x.shape: {x.shape}")
        timestep_log = jnp.log(timestep + 1e-6)  # Avoid log(0)
        print(f"timestep_log.shape: {timestep_log.shape}")
        t = self.timestep_embedder(timestep_log)  # b, d
        print(f"t.shape: {t.shape}")
        a = self.action_embedder(past_actions)  # b, t, d
        print(f"a.shape: {a.shape}")

        # Combine embeddings
        x = x + t + a
        return x


class ActionEmbedder(nnx.Module):
    """
    Embeds action vectors into a higher-dimensional space using an MLP.

    Attributes:
        linear1 (nnx.Linear): The first linear layer of the MLP.
        linear2 (nnx.Linear): The second linear layer of the MLP.
    """

    def __init__(self, hidden_size: int, input_size: int = 7, *, rngs: nnx.Rngs):
        """
        Initializes the ActionEmbedder module.

        Args:
            hidden_size: The dimensionality of the output embeddings.
            input_size: The dimensionality of the input action vectors.
            rngs: An nnx.Rngs object for initializing the weights of the linear
                  layers.
        """
        super().__init__()
        # The MLP consists of two linear layers with a SiLU activation.
        self.linear1 = nnx.Linear(input_size, hidden_size, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_size, hidden_size, rngs=rngs)

    def __call__(self, action: jnp.ndarray) -> jnp.ndarray:
        """
        Performs the forward pass of the ActionEmbedder.

        Args:
            action: A JAX array of action vectors with shape (B, input_size).

        Returns:
            A JAX array of action embeddings with shape (B, hidden_size).
        """
        x = self.linear1(action)
        x = nnx.silu(x)
        a_emb = self.linear2(x)
        return a_emb


class TimestepEmbedder(nnx.Module):
    """
    Embeds scalar timesteps into vector representations using JAX NNX.

    This module takes a batch of scalar timesteps and projects them into
    a high-dimensional space using sinusoidal embeddings followed by a
    two-layer multilayer perceptron (MLP).

    Attributes:
        linear1 (nnx.Linear): The first linear layer of the MLP.
        linear2 (nnx.Linear): The second linear layer of the MLP.
        frequency_embedding_size (int): The dimensionality of the sinusoidal
                                        timestep embeddings.
    """

    def __init__(
        self, hidden_size: int, frequency_embedding_size: int = 256, *, rngs: nnx.Rngs
    ):
        """
        Initializes the TimestepEmbedder module.

        Args:
            hidden_size: The dimensionality of the output embeddings.
            frequency_embedding_size: The size of the intermediate sinusoidal
                                      embeddings.
            rngs: An nnx.Rngs object for initializing the weights of the linear
                  layers.
        """
        super().__init__()
        # The MLP consists of two linear layers with a SiLU activation in between.
        self.linear1 = nnx.Linear(frequency_embedding_size, hidden_size, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(
        t: jnp.ndarray, dim: int, max_period: int = 10000
    ) -> jnp.ndarray:
        """
        Creates sinusoidal timestep embeddings.

        This is a static method that implements the sinusoidal positional
        embedding formula, which is commonly used in Transformer models.

        Args:
            t: A 1-D JAX array of N indices, one per batch element.
               These may be fractional.
            dim: The dimension of the output embeddings.
            max_period: Controls the minimum frequency of the embeddings.

        Returns:
            An (N, D) JAX array of positional embeddings.
        """
        # The core idea is to use sine and cosine functions of different frequencies.
        # This implementation is based on the one from OpenAI's GLIDE model.
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = jnp.exp(
            -math.log(max_period)
            * jnp.arange(start=0, stop=half, dtype=jnp.float32)
            / half
        )
        # Create the arguments for the sin/cos functions.
        args = t[:, None].astype(jnp.float32) * freqs[None]
        # Concatenate the cosine and sine embeddings.
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        # If the dimension is odd, pad with a zero.
        if dim % 2:
            embedding = jnp.concatenate(
                [embedding, jnp.zeros_like(embedding[:, :1])], axis=-1
            )
        return embedding

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """
        Performs the forward pass of the TimestepEmbedder.

        Args:
            t: A 1-D JAX array of N timesteps.

        Returns:
            An (N, hidden_size) JAX array of timestep embeddings.
        """
        # 1. Create the base sinusoidal frequency embeddings.
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)

        # 2. Pass the frequency embeddings through the MLP.
        x = self.linear1(t_freq)
        x = nnx.silu(x)
        t_emb = self.linear2(x)

        return t_emb


if __name__ == "__main__":
    import numpy as np
    from openpi.policies import libero_policy
    from openpi.policies import policy_config as _policy_config
    from openpi.shared import download
    from openpi.training import config as _config

    config = _config.get_config("pi0_fast_libero")
    checkpoint_dir = download.maybe_download(
        "gs://openpi-assets/checkpoints/pi0_fast_libero"
    )

    policy = _policy_config.create_trained_policy(config, checkpoint_dir)

    predictor = Predictor(policy=policy, rngs=nnx.Rngs(0, params=0))

    past_obs = libero_policy.make_multiframe_libero_example(4)
    future_obs = libero_policy.make_multiframe_libero_example(4)

    actions = np.random.rand(7)

    predictor.compute_loss(past_obs, future_obs, actions)
