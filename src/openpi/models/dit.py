from typing import Tuple, Optional

import jax
import jax.numpy as jnp
from flax import nnx


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def sinusoidal_embedding(
    t: jnp.ndarray, dim: int, max_period: float = 10_000.0
) -> jnp.ndarray:
    half = dim // 2
    freqs = jnp.exp(-jnp.log(max_period) * jnp.arange(half, dtype=jnp.float32) / half)
    args = t[:, None].astype(jnp.float32) * freqs[None, :]
    emb = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2 == 1:
        emb = jnp.pad(emb, ((0, 0), (0, 1)))
    return emb


def get_2d_sincos_pos_embed(embed_dim: int, grid: Tuple[int, int]) -> jnp.ndarray:
    t, n = grid

    def _pe(d: int, pos: jnp.ndarray) -> jnp.ndarray:
        assert d % 2 == 0
        omega = jnp.arange(d // 2, dtype=jnp.float32)
        omega = 1.0 / (10_000 ** (omega / (d / 2)))
        out = jnp.einsum("m,d->md", pos.reshape(-1), omega)
        return jnp.concatenate([jnp.sin(out), jnp.cos(out)], axis=-1)

    d_half = embed_dim // 2
    d_other = embed_dim - d_half
    pos_t = jnp.arange(t, dtype=jnp.float32)
    pos_n = jnp.arange(n, dtype=jnp.float32)
    emb_t = _pe(d_half, pos_t)  # [t, d/2]
    emb_n = _pe(d_other, pos_n)  # [n, d/2 or d/2+1]
    emb_t = jnp.repeat(emb_t[:, None, :], n, axis=1)
    emb_n = jnp.repeat(emb_n[None, :, :], t, axis=0)
    emb = jnp.concatenate([emb_t, emb_n], axis=-1)  # [t, n, C]
    return emb.reshape(1, t * n, embed_dim)


def make_block_causal_mask(T: int, N: int) -> jnp.ndarray:
    """
    Creates a mask for Cross-Attention where queries are (T*N) flattened video tokens
    and keys are (N + T) context tokens (N history + T actions).

    History tokens (first N columns) are always visible.
    Action tokens (last T columns) are causally masked based on time block.

    Returns:
        mask: Boolean tensor shape (1, 1, T*N, N+T) ready for broadcasting over Batch and Heads.
              True means allowed, False means masked.
    """
    # 1. Create time indices for the query rows (T*N rows)
    # Each block of N rows corresponds to one time step.
    # e.g., T=2, N=3 -> [0, 0, 0, 1, 1, 1]
    query_time_idx = jnp.repeat(jnp.arange(T), N)

    # 2. Create time indices for the action key columns (last T columns)
    # e.g., T=2 -> [0, 1]
    key_action_time_idx = jnp.arange(T)

    # 3. Create the causal mask for the action section.
    # A query at time t_q can see an action at time t_k if t_q >= t_k.
    # Shape: (T*N, T)
    action_mask = query_time_idx[:, None] >= key_action_time_idx[None, :]

    # 4. Create the history mask (always True).
    # Shape: (T*N, N)
    history_mask = jnp.ones((T * N, N), dtype=jnp.bool_)

    # 5. Concatenate to form the full mask.
    # Shape: (T*N, N + T)
    full_mask = jnp.concatenate([history_mask, action_mask], axis=1)

    # 6. Reshape for broadcasting over Batch and Heads dimensions in attention.
    # Final shape: (1, 1, T*N, N+T)
    return full_mask[None, None, :, :]


def modulate_spatial(
    x: jnp.ndarray, shift: jnp.ndarray, scale: jnp.ndarray
) -> jnp.ndarray:
    """AdaLN: x * (1 + scale) + shift; broadcast over sequence (middle) dimension."""
    # x: (Batch, Seq, Dim), shift/scale: (Batch, Dim) -> (Batch, 1, Dim)
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


def modulate_temporal(
    x: jnp.ndarray, shift: jnp.ndarray, scale: jnp.ndarray
) -> jnp.ndarray:
    """
    AdaLN variant aligned to x's shape (temporal per-token).
    x is (Batch*N, T, C). shift/scale are (Batch*N, C).
    We must broadcast shift/scale over T.
    """
    # FIX: Added [:, None, :] to broadcast (Batch, Dim) -> (Batch, 1, Dim)
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


# -----------------------------------------------------------------------------
# Core Blocks
# -----------------------------------------------------------------------------


class MLP(nnx.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(in_dim, hidden, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, out_dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.fc2(nnx.silu(self.fc1(x)))


# -----------------------------------------------------------------------------
# Embedders
# -----------------------------------------------------------------------------


class TimestepEmbedder(nnx.Module):
    def __init__(self, hidden_size: int, freq_dim: int = 256, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(freq_dim, hidden_size, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.freq_dim = freq_dim

    def __call__(self, t_scalar: jnp.ndarray) -> jnp.ndarray:
        t_emb = sinusoidal_embedding(t_scalar, self.freq_dim)
        return self.fc2(nnx.silu(self.fc1(t_emb)))


# -----------------------------------------------------------------------------
# Simple Video Transformer (temporal encoder with a global token)
# -----------------------------------------------------------------------------


class TransformerBlock(nnx.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ):
        self.norm1 = nnx.LayerNorm(dim, epsilon=1e-6, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=dim,
            qkv_features=dim,
            out_features=dim,
            decode=False,
            rngs=rngs,
        )
        self.norm2 = nnx.LayerNorm(dim, epsilon=1e-6, rngs=rngs)
        self.ffn = MLP(dim, int(dim * mlp_ratio), dim, rngs=rngs)
        self.drop = nnx.Dropout(dropout) if dropout > 0 else None

    def __call__(
        self, x: jnp.ndarray, *, rngs: Optional[nnx.Rngs] = None
    ) -> jnp.ndarray:
        h = self.attn(self.norm1(x))
        if self.drop is not None:
            h = self.drop(h, rngs=rngs)
        x = x + h
        h2 = self.ffn(self.norm2(x))
        if self.drop is not None:
            h2 = self.drop(h2, rngs=rngs)
        return x + h2


class VideoTransformer(nnx.Module):
    def __init__(
        self,
        in_channel: int,
        dim: int,
        depth: int = 8,
        num_heads: int = 8,
        *,
        rngs: nnx.Rngs,
    ):
        self.token = nnx.Param(jnp.zeros((1, 1, dim), dtype=jnp.float32))
        self.inp = nnx.Linear(in_channel, dim, rngs=rngs)
        self.blocks = nnx.Dict(
            {
                f"block_{i}": TransformerBlock(
                    dim, num_heads, mlp_ratio=2.0, dropout=0.0, rngs=rngs
                )
                for i in range(depth)
            }
        )

    def __call__(
        self, x: jnp.ndarray, *, rngs: Optional[nnx.Rngs] = None
    ) -> jnp.ndarray:
        # x: [B, T, N, Cin] -> [B, N, C]
        x = self.inp(x)
        B, T, N, C = x.shape
        x = x.reshape(B * N, T, C)
        cls = jnp.broadcast_to(self.token.value, (B * N, 1, C))
        x = jnp.concatenate([cls, x], axis=1)
        for _, block in self.blocks.items():
            x = block(x, rngs=rngs)
        g = x[:, 0, :]
        return g.reshape(B, N, C)


class CrossAttention(nnx.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        rngs: nnx.Rngs,
        qkv_bias: bool = True,
    ):
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=dim,
            qkv_features=dim,
            out_features=dim,
            decode=False,
            rngs=rngs,
        )

    def __call__(
        self,
        x_q: jnp.ndarray,
        x_kv: jnp.ndarray,
        *,
        mask: Optional[jnp.ndarray] = None,
        rngs: Optional[nnx.Rngs] = None,
    ) -> jnp.ndarray:
        # nnx.MultiHeadAttention expects mask where True means masked (opposite of our convention)
        # Convert our mask (True=allowed) to their format (True=masked)
        attn_mask = None
        if mask is not None:
            attn_mask = ~mask
        return self.attn(x_q, x_kv, mask=attn_mask)


class AdaLNModulator(nnx.Module):
    def __init__(self, dim: int, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(dim, dim, rngs=rngs)
        self.fc2 = nnx.Linear(dim, 6 * dim, rngs=rngs)
        self.fc2.kernel.value = jnp.zeros_like(self.fc2.kernel.value)

    def __call__(self, t: jnp.ndarray):
        # Only takes 't' (timestep), NOT actions
        h = nnx.silu(self.fc1(t))
        h = self.fc2(h)
        return jnp.split(h, 6, axis=-1)


class DiTBlock(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.norm1 = nnx.LayerNorm(hidden_size, epsilon=1e-6, rngs=rngs)
        self.norm3 = nnx.LayerNorm(hidden_size, epsilon=1e-6, rngs=rngs)
        self.self_attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=hidden_size,
            qkv_features=hidden_size,
            out_features=hidden_size,
            decode=False,
            rngs=rngs,
        )
        self.cross = CrossAttention(hidden_size, num_heads, rngs=rngs)
        self.mlp = MLP(hidden_size, int(hidden_size * 4.0), hidden_size, rngs=rngs)
        self.mod = AdaLNModulator(hidden_size, rngs=rngs)

    def __call__(
        self,
        x: jnp.ndarray,
        t_fea: jnp.ndarray,
        context_fea: jnp.ndarray,
        *,
        shape: Tuple[int, int, int, int],
        block_type: str,
        context_mask: Optional[jnp.ndarray] = None,
        rngs: Optional[nnx.Rngs] = None,
    ) -> jnp.ndarray:
        B, T, N, _ = shape

        # 1. Cross-Attention: Attend to History AND Actions
        x_flat = x.reshape(B, T * N, -1)
        x_res = self.cross(x_flat, context_fea, mask=context_mask, rngs=rngs)
        x = x + x_res.reshape(B, T, N, -1)

        # 2. Factorized Self-Attention (Spatial or Temporal)
        if block_type == "spatial":
            # Reshape to (Batch * Time, N_patches, Dim)
            x_bt = x.reshape(B * T, N, -1)

            # Repeat time embedding for every frame
            t_bt = jnp.repeat(t_fea, T, axis=0)  # [B*T, C]

            s_msa, sc_msa, g_msa, s_mlp, sc_mlp, g_mlp = self.mod(t_bt)

            # modulate_spatial handles (B*T, C) -> (B*T, 1, C) broadcasting
            x_tmp = modulate_spatial(self.norm1(x_bt), s_msa, sc_msa)
            y = self.self_attn(x_tmp)

            x_bt = x_bt + (1.0 + g_msa[:, None, :]) * y
            x_bt = x_bt + (1.0 + g_mlp[:, None, :]) * self.mlp(
                modulate_spatial(self.norm3(x_bt), s_mlp, sc_mlp)
            )

            # Reshape back: (B*T, N, C) -> (B, T, N, C)
            x = x_bt.reshape(B, T, N, -1)

        elif block_type == "temporal":
            # Reshape to (Batch * N_patches, Time, Dim)
            x_bn = x.reshape(B, T, N, -1).transpose(0, 2, 1, 3).reshape(B * N, T, -1)

            # Repeat time embedding for every spatial token
            t_bn = jnp.repeat(t_fea, N, axis=0)  # [B*N, C]

            s_msa, sc_msa, g_msa, s_mlp, sc_mlp, g_mlp = self.mod(t_bn)

            # modulate_temporal now handles (B*N, C) -> (B*N, 1, C) broadcasting
            x_tmp = modulate_temporal(self.norm1(x_bn), s_msa, sc_msa)
            y = self.self_attn(x_tmp)

            # Apply gating with explicit broadcasting
            x_bn = x_bn + (1.0 + g_msa[:, None, :]) * y
            x_bn = x_bn + (1.0 + g_mlp[:, None, :]) * self.mlp(
                modulate_temporal(self.norm3(x_bn), s_mlp, sc_mlp)
            )

            # Reshape back: (B*N, T, C) -> (B, N, T, C) -> (B, T, N, C)
            x = x_bn.reshape(B, N, T, -1).transpose(0, 2, 1, 3)

        return x


# -----------------------------------------------------------------------------
# Main Diffusion Transformer
# -----------------------------------------------------------------------------


class DiffusionTransformer(nnx.Module):

    def __init__(
        self,
        in_channel: int,
        hidden_size: int,
        num_heads: int,
        n_layers: int,
        freq_dim: int,
        video_depth: int,
        epsilon: float,
        *,
        rngs: nnx.Rngs,
    ):
        self.in_channel = in_channel
        self.hidden_size = hidden_size

        # Input Embedders
        self.x_embedder = nnx.Linear(in_channel, hidden_size, rngs=rngs)
        self.time_encoder = TimestepEmbedder(hidden_size, freq_dim=freq_dim, rngs=rngs)
        self.action_embedder = nnx.Linear(32, hidden_size, rngs=rngs)

        self.video_encoder = VideoTransformer(
            in_channel=in_channel,
            dim=hidden_size,
            depth=video_depth,
            num_heads=num_heads,
            rngs=rngs,
        )

        # Blocks
        self.n_layers = n_layers
        self.blocks = nnx.Dict(
            {
                f"block_{i}": DiTBlock(hidden_size, num_heads, rngs=rngs)
                for i in range(self.n_layers)
            }
        )

        # Output Head
        self.final_norm = nnx.LayerNorm(hidden_size, epsilon=epsilon, rngs=rngs)
        self.final_linear = nnx.Linear(hidden_size, in_channel, rngs=rngs)

        # Zero-init output for stability
        self.final_linear.kernel.value = jnp.zeros_like(self.final_linear.kernel.value)

    def __call__(
        self,
        x_noisy: jnp.ndarray,
        lc_his: jnp.ndarray,
        action_tokens: jnp.ndarray,
        time: jnp.ndarray,
        *,
        rngs: Optional[nnx.Rngs] = None,
    ) -> jnp.ndarray:
        B, T, N, Cin = x_noisy.shape

        # Embed Noisy Input
        x = self.x_embedder(x_noisy).reshape(B, T * N, self.hidden_size)
        pos = get_2d_sincos_pos_embed(self.hidden_size, (T, N))
        x = x + pos
        x = x.reshape(B, T, N, self.hidden_size)

        # Embed Time (AdaLN Driver)
        t_fea = self.time_encoder(jnp.log(time + 1e-8))  # [B, Hidden_Size]

        # Embed Context (Cross-Attention Targets)
        # History: [B, T_his, N, C] -> VideoTransformer -> [B, N, Hidden_Size]
        v_fea = self.video_encoder(lc_his, rngs=rngs)  # [B, N, Hidden_Size]

        a_fea = self.action_embedder(action_tokens)  # [B, T, Hidden_Size]

        context_fea = jnp.concatenate([v_fea, a_fea], axis=1)

        ctx_mask = make_block_causal_mask(T, N)

        for i in range(self.n_layers):
            mode = "spatial" if i % 2 == 0 else "temporal"
            x = self.blocks[f"block_{i}"](
                x,
                t_fea=t_fea,
                context_fea=context_fea,
                shape=(B, T, N, Cin),
                block_type=mode,
                context_mask=ctx_mask,
                rngs=rngs,
            )

        x = self.final_norm(x)
        y_pred = self.final_linear(x)

        return y_pred
