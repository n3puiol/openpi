import math
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

    def get_siglip_encoding(self, batch: list[list[dict]]) -> tuple:
        embedding_imgs = []
        embedding_wrist_imgs = []
        embedding_texts = []

        for observations in batch:
            for observation in observations:
                token_embeddings, _, _ = self._policy.get_encoding(observation)
                print(f"token_embeddings.shape: {token_embeddings.shape}")
                embedding_img = token_embeddings[:, :256]
                embedding_wrist_img = token_embeddings[:, 256:512]
                embedding_text = token_embeddings[:, 512:]

                embedding_imgs.append(embedding_img)
                embedding_wrist_imgs.append(embedding_wrist_img)
                embedding_texts.append(embedding_text)

        return (
            jnp.vstack(embedding_imgs),
            jnp.vstack(embedding_wrist_imgs),
            jnp.vstack(embedding_texts),
        )

    def add_noise(
        self, x: at.Array, noise: at.Array, timestep: at.Array, c: at.Array
    ) -> at.Array:
        time = timestep.reshape(c.shape[0], *((1,) * (len(c.shape) - 1)))
        x_noisy = x + c * time + time * noise
        return x_noisy

    def compute_loss(
        self,
        past_observations: list[list[dict]],
        future_observations: list[list[dict]],
        actions: _model.Actions,
    ):
        # explore using fast tokenizer for actions

        b = len(past_observations)
        t = len(past_observations[0])

        # Get past and future SigLip encodings (img: (B*t, 256, 2048), wrist: (B*t, 256, 2048), text: (B*t, x, 2048))
        past_encoding_img, _, _ = self.get_siglip_encoding(past_observations)
        print(f"past_encoding_img.shape: {past_encoding_img.shape}")
        lc_his = np.reshape(past_encoding_img, (b, t, 256, 2048))
        print(f"lc_his.shape: {lc_his.shape}")
        x_prior = lc_his[:, -1:, :, :]  # (B, 1, 256, 2048)
        print(f"x_prior.shape: {x_prior.shape}")
        future_encoding_img, _, _ = self.get_siglip_encoding(future_observations)
        print(f"future_encoding_img.shape: {future_encoding_img.shape}")
        lc_next = np.reshape(future_encoding_img, (b, t, 256, 2048))
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


# class DiffusionTransformer(nnx.Module):
#     def __init__(
#         self,
#         in_channel: int = 2048,
#         hidden_dim: int = 512,
#         img_size: int = 256,
#         tube_size: int = 4,
#         num_layers: int = 12,
#         num_heads: int = 8,
#         mlp_mult: float = 4.0,
#         dropout: float = 0.1,
#         *,
#         rngs: nnx.Rngs,
#     ):
#         self._rngs = rngs
#         self._in_channel = in_channel
#         self._hidden_dim = hidden_dim
#         self._img_size = img_size
#         self._tube_size = tube_size
#         self._num_heads = num_heads
#         self._mlp_mult = mlp_mult

#         self.x_embedder = nnx.Linear(
#             self._in_channel, self._hidden_dim, rngs=self._rngs
#         )
#         self.timestep_embedder = TimestepEmbedder(self._hidden_dim, rngs=self._rngs)
#         self.action_embedder = ActionEmbedder(self._hidden_dim, rngs=self._rngs)

#         self.pos_embed = jnp.zeros(
#             (1, self._img_size * self._tube_size, self._hidden_dim)
#         )

#         self.video_encoder = VideoTransformer(
#             rngs=self._rngs,
#             in_channel=self._in_channel,
#             dim=self._hidden_dim,
#         )

#         self.blocks = [
#             DiTBlock(
#                 self._hidden_dim,
#                 self._num_heads,
#                 self._mlp_mult,
#                 self._hidden_dim,
#                 attn_mode="spatial" if i % 2 == 0 else "temporal",
#                 dropout=0.0,
#                 rngs=self._rngs,
#             )
#             for i in range(num_layers)
#         ]

#     def __call__(
#         self,
#         x_noisy: at.Array,
#         l_his: at.Array,
#         past_actions: _model.Actions,
#         timestep: at.Array,
#     ):
#         b, t, n, c = x_noisy.shape
#         x = self.x_embedder(x_noisy)
#         print(f"x.shape: {x.shape}")
#         x = x.reshape(b, t * n, -1) + self.pos_embed  # b, (t*n), c
#         print(f"x.shape: {x.shape}")

#         timestep_log = jnp.log(timestep + 1e-6)  # Avoid log(0)
#         print(f"timestep_log.shape: {timestep_log.shape}")
#         t_emb = self.timestep_embedder(timestep_log)  # b, d
#         print(f"t_emb.shape: {t_emb.shape}")

#         print(f"past_actions.shape: {past_actions.shape}")
#         a_emb = self.action_embedder(past_actions)  # b, t, d
#         print(f"a_emb.shape: {a_emb.shape}")

#         print(f"l_his.shape: {l_his.shape}")
#         v_emb = self.video_encoder(l_his)
#         print(f"v_emb.shape: {v_emb.shape}")

#         for block in self.blocks:
#             x = block(x, t_emb, a_emb, v_emb, n, T=t)
#         return x


# class ActionEmbedder(nnx.Module):
#     """
#     Embeds action vectors into a higher-dimensional space using an MLP.

#     Attributes:
#         linear1 (nnx.Linear): The first linear layer of the MLP.
#         linear2 (nnx.Linear): The second linear layer of the MLP.
#     """

#     def __init__(self, hidden_size: int, input_size: int = 7, *, rngs: nnx.Rngs):
#         """
#         Initializes the ActionEmbedder module.

#         Args:
#             hidden_size: The dimensionality of the output embeddings.
#             input_size: The dimensionality of the input action vectors.
#             rngs: An nnx.Rngs object for initializing the weights of the linear
#                   layers.
#         """
#         super().__init__()
#         # The MLP consists of two linear layers with a SiLU activation.
#         self.linear1 = nnx.Linear(input_size, hidden_size, rngs=rngs)
#         self.linear2 = nnx.Linear(hidden_size, hidden_size, rngs=rngs)

#     def __call__(self, action: jnp.ndarray) -> jnp.ndarray:
#         """
#         Performs the forward pass of the ActionEmbedder.

#         Args:
#             action: A JAX array of action vectors with shape (B, input_size).

#         Returns:
#             A JAX array of action embeddings with shape (B, hidden_size).
#         """
#         x = self.linear1(action)
#         x = nnx.silu(x)
#         a_emb = self.linear2(x)
#         return a_emb


# class TimestepEmbedder(nnx.Module):
#     """
#     Embeds scalar timesteps into vector representations using JAX NNX.

#     This module takes a batch of scalar timesteps and projects them into
#     a high-dimensional space using sinusoidal embeddings followed by a
#     two-layer multilayer perceptron (MLP).

#     Attributes:
#         linear1 (nnx.Linear): The first linear layer of the MLP.
#         linear2 (nnx.Linear): The second linear layer of the MLP.
#         frequency_embedding_size (int): The dimensionality of the sinusoidal
#                                         timestep embeddings.
#     """

#     def __init__(
#         self, hidden_size: int, frequency_embedding_size: int = 256, *, rngs: nnx.Rngs
#     ):
#         """
#         Initializes the TimestepEmbedder module.

#         Args:
#             hidden_size: The dimensionality of the output embeddings.
#             frequency_embedding_size: The size of the intermediate sinusoidal
#                                       embeddings.
#             rngs: An nnx.Rngs object for initializing the weights of the linear
#                   layers.
#         """
#         super().__init__()
#         # The MLP consists of two linear layers with a SiLU activation in between.
#         self.linear1 = nnx.Linear(frequency_embedding_size, hidden_size, rngs=rngs)
#         self.linear2 = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
#         self.frequency_embedding_size = frequency_embedding_size

#     @staticmethod
#     def timestep_embedding(
#         t: jnp.ndarray, dim: int, max_period: int = 10000
#     ) -> jnp.ndarray:
#         """
#         Creates sinusoidal timestep embeddings.

#         This is a static method that implements the sinusoidal positional
#         embedding formula, which is commonly used in Transformer models.

#         Args:
#             t: A 1-D JAX array of N indices, one per batch element.
#                These may be fractional.
#             dim: The dimension of the output embeddings.
#             max_period: Controls the minimum frequency of the embeddings.

#         Returns:
#             An (N, D) JAX array of positional embeddings.
#         """
#         # The core idea is to use sine and cosine functions of different frequencies.
#         # This implementation is based on the one from OpenAI's GLIDE model.
#         # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
#         half = dim // 2
#         freqs = jnp.exp(
#             -math.log(max_period)
#             * jnp.arange(start=0, stop=half, dtype=jnp.float32)
#             / half
#         )
#         # Create the arguments for the sin/cos functions.
#         args = t[:, None].astype(jnp.float32) * freqs[None]
#         # Concatenate the cosine and sine embeddings.
#         embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
#         # If the dimension is odd, pad with a zero.
#         if dim % 2:
#             embedding = jnp.concatenate(
#                 [embedding, jnp.zeros_like(embedding[:, :1])], axis=-1
#             )
#         return embedding

#     def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
#         """
#         Performs the forward pass of the TimestepEmbedder.

#         Args:
#             t: A 1-D JAX array of N timesteps.

#         Returns:
#             An (N, hidden_size) JAX array of timestep embeddings.
#         """
#         # 1. Create the base sinusoidal frequency embeddings.
#         t_freq = self.timestep_embedding(t, self.frequency_embedding_size)

#         # 2. Pass the frequency embeddings through the MLP.
#         x = self.linear1(t_freq)
#         x = nnx.silu(x)
#         t_emb = self.linear2(x)

#         return t_emb


# class MLP(nnx.Module):
#     def __init__(self, dim: int, hidden_mult: float = 4.0, *, rngs: nnx.Rngs):
#         hidden = int(dim * hidden_mult)
#         self.fc1 = nnx.Linear(dim, hidden, rngs=rngs)
#         self.fc2 = nnx.Linear(hidden, dim, rngs=rngs)

#     def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
#         x = self.fc1(x)
#         x = jax.nn.gelu(x)
#         x = self.fc2(x)
#         return x


# class AdaLNMod(nnx.Module):
#     """Produce per-block modulation vectors (shift/scale/gate) from a cond vector.
#     Returns two triplets for MSA and MLP paths, each of size dim.
#     We use a learnable scaling param initialized to 0 to realize AdaLN-Zero behavior.
#     """

#     def __init__(self, dim: int, hidden: int, *, rngs: nnx.Rngs):
#         self.fc1 = nnx.Linear(dim, hidden, rngs=rngs)
#         self.fc2 = nnx.Linear(hidden, 6 * dim, rngs=rngs)
#         # Start with zero contribution (AdaLN-Zero)
#         self.out_scale = nnx.Param(jnp.array(0.0))

#     def __call__(self, c: jnp.ndarray) -> List[jnp.ndarray]:
#         z = self.fc1(c)
#         z = jax.nn.silu(z)
#         z = self.fc2(z) * self.out_scale
#         # split into 6 parts: (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
#         return jnp.split(z, 6, axis=-1)


# class DiTBlock(nnx.Module):
#     """DiT block that can operate in *spatial* or *temporal* attention mode.

#     Spatial mode: attend within each frame across spatial tokens (H*W) independently for each time step.
#     Temporal mode: attend across the time axis for each spatial location independently.

#     Expect the input token sequence to be arranged as (B, T*H*W, D). For images, T=1.
#     """

#     def __init__(
#         self,
#         dim: int,
#         num_heads: int,
#         mlp_mult: float,
#         cond_hidden: int,
#         dropout: float = 0.0,
#         eps: float = 1e-6,
#         attn_mode: str = "spatial",
#         *,
#         rngs: nnx.Rngs,
#     ):
#         assert attn_mode in (
#             "spatial",
#             "temporal",
#         ), "attn_mode must be 'spatial' or 'temporal'"
#         self.attn_mode = attn_mode

#         self.norm1 = nnx.LayerNorm(dim, epsilon=eps, rngs=rngs)
#         self.attn = Attention(dim, num_heads, dropout=dropout, rngs=rngs)
#         self.cross_attn = CrossAttention(dim, num_heads, dropout=dropout, rngs=rngs)
#         self.proj1_scale = nnx.Param(jnp.array(0.0))  # residual scale (start at 0)

#         self.norm2 = nnx.LayerNorm(dim, epsilon=eps, rngs=rngs)
#         self.mlp = MLP(dim, mlp_mult, rngs=rngs)
#         self.proj2_scale = nnx.Param(jnp.array(0.0))

#         self.mod = AdaLNMod(dim, cond_hidden, rngs=rngs)

#     def _modulate_spatial(
#         self, x_bn: jnp.ndarray, shift: jnp.ndarray, scale: jnp.ndarray
#     ) -> jnp.ndarray:
#         # x_bn: (B*T, N, D); shift/scale: (B*T, D)
#         return (1 + scale)[:, None, :] * x_bn + shift[:, None, :]

#     def _modulate_temporal(
#         self, x_bt: jnp.ndarray, shift: jnp.ndarray, scale: jnp.ndarray
#     ) -> jnp.ndarray:
#         # x_bt: (B*N, T, D); shift/scale: (B*N, T, D)
#         return (1 + scale) * x_bt + shift

#     def __call__(
#         self,
#         x: jnp.ndarray,
#         t_emb: jnp.ndarray,
#         a_emb: jnp.ndarray,
#         v_emb: jnp.ndarray,
#         N: int,
#         mask: Optional[jnp.ndarray] = None,
#         T: int = 1,
#     ) -> jnp.ndarray:
#         """Args:
#         x: (B, T*H*W, D) tokens
#         t: (B, D) timestep embedding
#         a_emb: (B, T, D) action embedding
#         v_emb: (B, D) video embedding
#         N: number of tokens
#         T: number of frames (default 1)
#         """
#         B, L, D = x.shape
#         assert L == T * N, "Sequence length must equal T * N"

#         x = x + self.cross_attn(x, v_emb, mask)

#         if self.attn_mode == "spatial":
#             t_rep = jnp.repeat(t_emb, repeats=T, axis=0)
#             cond_flat = a_emb.reshape(B * T, D)
#             cond_vec = t_rep + cond_flat
#             shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.mod(
#                 cond_vec
#             )
#             x_sp = x.reshape(B, T, N, D).reshape(B * T, N, D)
#             x_sp = self._modulate_spatial(self.norm1(x_sp), shift_msa, scale_msa)
#             attn_out = self.attn(x_sp)
#             x_sp = x_sp + self.proj1_scale * gate_msa[:, None, :] * attn_out
#             x_sp_norm = self.norm2(x_sp)
#             x_sp_mod = self._modulate_spatial(x_sp_norm, shift_mlp, scale_mlp)
#             mlp_out = self.mlp(x_sp_mod)
#             x_sp = x_sp + self.proj2_scale * gate_mlp[:, None, :] * mlp_out
#             x = x_sp.reshape(B, T, N, D).reshape(B, L, D)
#         else:  # temporal
#             # (B*N, T, D) : per-location sequence over time
#             x_tm = x.reshape(B, T, N, D).transpose(0, 2, 1, 3).reshape(B * N, T, D)
#             t_rep = jnp.repeat(t_emb, repeats=N, axis=0)  # (B*N, D)
#             cond_tm = jnp.repeat(a_emb, repeats=N, axis=1)  # (B, T*N, D)
#             cond_tm = cond_tm.reshape(B * N, T, D)
#             cond_vec = t_rep[:, None, :] + cond_tm  # (B*N, T, D)
#             shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.mod(
#                 cond_vec
#             )
#             x_tm = self._modulate_temporal(self.norm1(x_tm), shift_msa, scale_msa)
#             attn_out = self.attn(x_tm)
#             x_tm = x_tm + self.proj1_scale * gate_msa * attn_out
#             x_tm_norm = self.norm2(x_tm)
#             x_tm_mod = self._modulate_temporal(x_tm_norm, shift_mlp, scale_mlp)
#             mlp_out = self.mlp(x_tm_mod)
#             x_tm = x_tm + self.proj2_scale * gate_mlp * mlp_out
#             x = x_tm.reshape(B, N, T, D).transpose(0, 2, 1, 3).reshape(B, L, D)

#         return x


# # -------------------------------
# # Utilities: dropout / drop path
# # -------------------------------


# def _dropout(
#     x: jnp.ndarray, rate: float, rng: Optional[jax.Array], training: bool
# ) -> jnp.ndarray:
#     """Simple per-element dropout."""
#     if not training or rate <= 0.0 or rng is None:
#         return x
#     keep = 1.0 - rate
#     mask = jax.random.bernoulli(rng, p=keep, shape=x.shape)
#     return jnp.where(mask, x / keep, 0.0)


# def _drop_path(
#     x: jnp.ndarray,
#     drop_prob: float,
#     rng: Optional[jax.Array],
#     training: bool,
#     scale_by_keep: bool = True,
# ) -> jnp.ndarray:
#     """Stochastic depth per-sample (broadcast across all non-batch axes)."""
#     if not training or drop_prob <= 0.0 or rng is None:
#         return x
#     keep = 1.0 - drop_prob
#     shape = (x.shape[0],) + (1,) * (x.ndim - 1)
#     mask = jax.random.bernoulli(rng, p=keep, shape=shape).astype(x.dtype)
#     if scale_by_keep and keep > 0.0:
#         mask = mask / keep
#     return x * mask


# # -------------------------------
# # Multi-head Self-Attention
# # -------------------------------


# class Attention(nnx.Module):
#     def __init__(
#         self, dim: int, num_heads: int = 8, dropout: float = 0.0, *, rngs: nnx.Rngs
#     ):
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.att_scale = self.head_dim**-0.5

#         # qkv and output projection
#         self.qkv = nnx.Linear(
#             dim, num_heads * self.head_dim * 3, use_bias=False, rngs=rngs
#         )
#         self.proj = nnx.Linear(num_heads * self.head_dim, dim, rngs=rngs)
#         self.proj_dropout_rate = float(dropout)

#         # last attention weights (for inspection)
#         self.att_weights: Optional[jnp.ndarray] = None

#     def __call__(
#         self,
#         x: jnp.ndarray,
#         mask: Optional[jnp.ndarray] = None,
#         *,
#         rng: Optional[jax.Array] = None,
#         training: bool = True,
#     ) -> jnp.ndarray:
#         """
#         x: (B, N, C)
#         mask:
#           - (B, N)         -> broadcast to (B, 1, 1, N)
#           - (1, N, N)      -> broadcast to (1, 1, N, N)
#           - (B, N, N)      -> broadcast to (B, 1, N, N)
#         """
#         B, N, _ = x.shape

#         # qkv: (B, N, 3 * H * D)
#         qkv = self.qkv(x)
#         qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)  # (B, N, 3, H, D)
#         # (3, B, H, N, D)
#         qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
#         q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, H, N, D)

#         # attention scores: (B, H, N, N)
#         attn = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) * self.att_scale

#         if mask is not None:
#             very_neg = jnp.finfo(attn.dtype).min
#             m = mask.astype(bool)
#             if m.ndim == 2:  # (B, N)
#                 m = m[:, None, None, :]  # (B, 1, 1, N)
#             elif m.ndim == 3 and m.shape[0] == 1:  # (1, N, N)
#                 m = m[None, ...]  # (1, 1, N, N) by broadcasting
#             elif m.ndim == 3:  # (B, N, N)
#                 m = m[:, None, :, :]  # (B, 1, N, N)
#             else:
#                 raise ValueError("mask shape is not correct for attention")
#             attn = jnp.where(m, attn, very_neg)

#         attn = jax.nn.softmax(attn, axis=-1)  # (B, H, N, N)
#         self.att_weights = attn

#         # out: (B, H, N, D) -> (B, N, H*D)
#         out = jnp.matmul(attn, v)
#         out = jnp.transpose(out, (0, 2, 1, 3)).reshape(
#             B, N, self.num_heads * self.head_dim
#         )

#         # output projection + dropout
#         out = self.proj(out)
#         proj_key = None
#         if rng is not None:
#             rng, proj_key = jax.random.split(rng)

#         out = _dropout(out, self.proj_dropout_rate, proj_key, training)
#         return out


# class CrossAttention(nnx.Module):
#     def __init__(
#         self, dim: int, num_heads: int = 8, dropout: float = 0.0, *, rngs: nnx.Rngs
#     ):
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.att_scale = self.head_dim**-0.5

#         # qkv and output projection
#         self.qkv = nnx.Linear(
#             dim, num_heads * self.head_dim * 3, use_bias=False, rngs=rngs
#         )
#         self.proj = nnx.Linear(num_heads * self.head_dim, dim, rngs=rngs)
#         self.proj_dropout_rate = float(dropout)

#     def __call__(
#         self,
#         x: jnp.ndarray,
#         context: jnp.ndarray,
#         mask: Optional[jnp.ndarray] = None,
#         *,
#         rng: Optional[jax.Array] = None,
#         training: bool = True,
#     ) -> jnp.ndarray:
#         """
#         x: (B, N, C)
#         context: (B, M, C)
#         mask:
#           - (B, N)         -> broadcast to (B, 1, 1, N)
#           - (1, N, N)      -> broadcast to (1, 1, N, N)
#           - (B, N, N)      -> broadcast to (B, 1, N, N)
#         """
#         B, N, _ = x.shape
#         _, M, _ = context.shape

#         # qkv: (B, N+M, 3 * H * D)
#         qkv = self.qkv(jnp.concatenate([x, context], axis=1))
#         qkv = qkv.reshape(
#             B, N + M, 3, self.num_heads, self.head_dim
#         )  # (B, N+M, 3, H, D)
#         # (3, B, H, N+M, D)
#         qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
#         q = qkv[0]  # (B, H, N+M, D)
#         k = qkv[1]  # (B, H, M+N ,D)
#         v = qkv[2]  # (B ,H ,N+M ,D)

#         attn = jnp.einsum("bhdn,bhmd->bhmn", q, k) * self.att_scale
#         if mask is not None:
#             attn = jnp.where(mask, attn, jnp.finfo(attn.dtype).min)
#         attn = jax.nn.softmax(attn, axis=-1)

#         out = jnp.einsum("bhmn,bhmd->bhdn", attn, v)
#         out = self.proj(out)
#         proj_key = None
#         if rng is not None:
#             rng, proj_key = jax.random.split(rng)

#         out = _dropout(out, self.proj_dropout_rate, proj_key, training)
#         return out


# # -------------------------------
# # MLP / Feedforward
# # -------------------------------


# class TransformerFeedForwardNN(nnx.Module):
#     def __init__(self, rngs: nnx.Rngs, dim: int, hidden_dim: int, dropout: float = 0.0):
#         self.fc1 = nnx.Linear(dim, hidden_dim, rngs=rngs)
#         self.fc2 = nnx.Linear(hidden_dim, dim, rngs=rngs)
#         self.dropout_rate = float(dropout)

#     def __call__(
#         self, x: jnp.ndarray, *, rng: Optional[jax.Array] = None, training: bool = True
#     ) -> jnp.ndarray:
#         k1, k2 = (None, None)
#         if rng is not None:
#             k1, k2 = jax.random.split(rng)

#         y = self.fc1(x)
#         y = jax.nn.gelu(y)
#         y = _dropout(y, self.dropout_rate, k1, training)
#         y = self.fc2(y)
#         y = _dropout(y, self.dropout_rate, k2, training)
#         return y


# # -------------------------------
# # DropPath module
# # -------------------------------


# class DropPath(nnx.Module):
#     def __init__(self, drop_prob: float = 0.0, *, scale_by_keep: bool = True):
#         self.drop_prob = float(drop_prob)
#         self.scale_by_keep = bool(scale_by_keep)

#     def __call__(
#         self, x: jnp.ndarray, *, rng: Optional[jax.Array] = None, training: bool = True
#     ) -> jnp.ndarray:
#         return _drop_path(x, self.drop_prob, rng, training, self.scale_by_keep)


# # -------------------------------
# # Sinusoidal Position Encoding
# # -------------------------------


# class SinusoidalPositionEncoding(nnx.Module):
#     def __init__(
#         self,
#         input_size: int,
#         inv_freq_factor: float = 10.0,
#         factor_ratio: Optional[float] = None,
#     ):
#         self.input_size = int(input_size)
#         self.inv_freq_factor = float(inv_freq_factor)

#         channels = self.input_size
#         channels = int(jnp.ceil(channels / 2) * 2)
#         self.channels = channels

#         # buffer-like (non-param) values
#         idx = jnp.arange(0, channels, 2, dtype=jnp.float32)
#         self.inv_freq = 1.0 / (self.inv_freq_factor ** (idx / channels))

#         # Optional learnable global scale factor
#         if factor_ratio is None:
#             # non-parameter scalar
#             self.factor = 1.0
#             self._learned_factor = False
#         else:
#             # learnable parameter (shape [1] for broadcasting)
#             self.factor = nnx.Param(jnp.array([factor_ratio], dtype=jnp.float32))
#             self._learned_factor = True

#     def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
#         """
#         x: (B, N, C) or (N, C) — only N (sequence length) is used.
#         returns: (N, channels)
#         """
#         seq_len = x.shape[-2]
#         pos = jnp.arange(seq_len, dtype=self.inv_freq.dtype)  # (N,)
#         sin_inp = jnp.einsum("i,j->ij", pos, self.inv_freq)  # (N, channels//2)
#         emb = jnp.concatenate(
#             [jnp.sin(sin_inp), jnp.cos(sin_inp)], axis=-1
#         )  # (N, channels)

#         factor = (
#             self.factor
#             if isinstance(self.factor, jnp.ndarray)
#             else jnp.array(self.factor, dtype=emb.dtype)
#         )
#         return emb * factor

#     # for symmetry with your original helpers
#     def output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
#         return input_shape

#     def output_size(self, input_size: int) -> int:
#         return input_size


# # -------------------------------
# # One decoder layer
# # -------------------------------


# class DecoderLayer(nnx.Module):
#     def __init__(
#         self,
#         dim: int,
#         num_heads: int,
#         mlp_hidden_size: int,
#         dropout: float,
#         eps: float = 1e-6,
#         *,
#         rngs: nnx.Rngs,
#     ):
#         self.att_norm = nnx.LayerNorm(num_features=dim, epsilon=eps, rngs=rngs)
#         self.att = Attention(
#             dim,
#             num_heads=num_heads,
#             dropout=dropout,
#             rngs=rngs,
#         )
#         self.ff_norm = nnx.LayerNorm(num_features=dim, epsilon=eps, rngs=rngs)
#         self.ff = TransformerFeedForwardNN(rngs, dim, mlp_hidden_size, dropout=dropout)
#         self.drop_path_att = DropPath(dropout)
#         # FFN residual uses the model-level DropPath in your original, but we keep a dedicated one here for clarity
#         self.drop_path_ff = DropPath(dropout)

#     def __call__(
#         self,
#         x: jnp.ndarray,
#         mask: Optional[jnp.ndarray],
#         *,
#         rng: Optional[jax.Array] = None,
#         training: bool = True,
#     ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
#         """
#         Returns (new_x, att_weights)
#         """
#         # Split rng for attention proj-drop, att droppath, ffn dropouts, ffn droppath
#         k_att, k_dp_att, k_ff, k_dp_ff = (None, None, None, None)
#         if rng is not None:
#             k_att, k_dp_att, k_ff, k_dp_ff = jax.random.split(rng, 4)

#         # Self-attention + residual
#         y = self.att_norm(x)
#         y = self.att(y, mask=mask, rng=k_att, training=training)
#         x = x + self.drop_path_att(y, rng=k_dp_att, training=training)
#         att_w = self.att.att_weights

#         # FFN + residual
#         y = self.ff_norm(x)
#         y = self.ff(y, rng=k_ff, training=training)
#         x = x + self.drop_path_ff(y, rng=k_dp_ff, training=training)

#         return x, att_w


# # -------------------------------
# # Transformer Decoder (stack)
# # -------------------------------


# class TransformerDecoder(nnx.Module):
#     def __init__(
#         self,
#         input_size: int,
#         num_layers: int,
#         num_heads: int,
#         mlp_hidden_size: int,
#         dropout: float,
#         *,
#         rngs: nnx.Rngs,
#     ):
#         self.layers: List[DecoderLayer] = []
#         for _ in range(num_layers):
#             self.layers.append(
#                 DecoderLayer(
#                     dim=input_size,
#                     num_heads=num_heads,
#                     mlp_hidden_size=mlp_hidden_size,
#                     dropout=dropout,
#                     rngs=rngs,
#                 )
#             )

#         # cache / helpers
#         self.attention_output = {}  # layer_idx -> att_weights
#         self.seq_len: Optional[int] = None
#         self.num_elements: Optional[int] = None
#         self.mask: Optional[jnp.ndarray] = (
#             None  # (1, N, N) or (1, L*E, L*E) as in your original
#         )

#     def __call__(
#         self,
#         x: jnp.ndarray,
#         mask: Optional[jnp.ndarray] = None,
#         *,
#         rng: Optional[jax.Array] = None,
#         training: bool = True,
#     ) -> jnp.ndarray:
#         """
#         x: (B, N, C)
#         mask: optional bool mask as in Attention.__call__ docstring.
#         """
#         key = rng
#         for layer_idx, layer in enumerate(self.layers):
#             k_layer = None
#             if key is not None:
#                 key, k_layer = jax.random.split(key)
#             x, att_w = layer(
#                 x,
#                 mask=(mask if mask is not None else self.mask),
#                 rng=k_layer,
#                 training=training,
#             )
#             # Store attention weights during eval (to mirror your PyTorch behavior)
#             if not training:
#                 self.attention_output[layer_idx] = att_w
#         return x


# class VideoTransformer(nnx.Module):
#     def __init__(self, rngs: nnx.Rngs, in_channel: int, dim: int, depth: int = 8):
#         super().__init__()
#         self.global_token = nnx.Param(jax.random.normal(rngs(), (1, 1, dim)))
#         self.in_layer = nnx.Linear(in_channel, dim, rngs=rngs)
#         self.transformer_layer = TransformerDecoder(
#             input_size=dim,
#             num_layers=depth,
#             num_heads=8,
#             mlp_hidden_size=dim * 2,
#             dropout=0.1,
#             rngs=rngs,
#         )

#     def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
#         """
#         Forward pass of VideoTransformer.

#         Args:
#             x: Input tensor with shape (B, T, N, C)

#         Returns:
#             Output tensor with shape (B, N, C)
#         """
#         # x: b t n c
#         x = self.in_layer(x)
#         B, T, N, C = x.shape

#         # Rearrange from 'b t n c' to '(b n) t c'
#         x = x.transpose(0, 2, 1, 3).reshape(B * N, T, C)

#         # Add global token: repeat for each sequence and concatenate
#         global_tokens = jnp.tile(self.global_token.value, (x.shape[0], 1, 1))
#         x = jnp.concatenate([global_tokens, x], axis=1)

#         # Pass through transformer and take only the global token output
#         x = self.transformer_layer(x)[:, 0]  # Shape: (B*N, C)

#         # Rearrange back from '(b n) c' to 'b n c'
#         x = x.reshape(B, N, C)

#         return x


if __name__ == "__main__":
    import numpy as np
    from openpi.policies import libero_policy
    from openpi.policies import policy_config as _policy_config
    from openpi.shared import download
    from openpi.training import config as _config
    import openpi.training.data_loader as _data_loader
    import openpi.training.sharding as sharding

    config = _config.get_config("pi0_fast_libero")
    checkpoint_dir = download.maybe_download(
        "gs://openpi-assets/checkpoints/pi0_fast_libero"
    )

    policy = _policy_config.create_trained_policy(config, checkpoint_dir)
    # config = config.unfrozen()
    # config.data.base_config.action_sequence_keys = ('actions', 'image', )
    # config = config.frozen()

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS)
    )
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)

    predictor = Predictor(policy=policy, rngs=nnx.Rngs(0, params=0))

    past_obs = libero_policy.make_multiframe_libero_example(4)
    future_obs = libero_policy.make_multiframe_libero_example(4)

    key = jax.random.PRNGKey(0)
    rngs = nnx.Rngs(params=key, dropout=key)
    actions = jax.random.normal(key, (1, 4, 7))

    predictor.compute_loss([past_obs], [future_obs], actions)
