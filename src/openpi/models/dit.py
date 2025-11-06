from typing import Tuple, Optional

import jax
import jax.numpy as jnp
from flax import nnx


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def modulate_spatial(
    x: jnp.ndarray, shift: jnp.ndarray, scale: jnp.ndarray
) -> jnp.ndarray:
    """AdaLN: x * (1 + scale) + shift; broadcast over tokens."""
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


def modulate_temporal(
    x: jnp.ndarray, shift: jnp.ndarray, scale: jnp.ndarray
) -> jnp.ndarray:
    """AdaLN variant aligned to x's shape (e.g., temporal per-token)."""
    return x * (1.0 + scale) + shift


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


# -----------------------------------------------------------------------------
# Core Blocks
# -----------------------------------------------------------------------------


class MLP(nnx.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(in_dim, hidden, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, out_dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.fc2(nnx.silu(self.fc1(x)))


class MultiHeadAttention(nnx.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        rngs: nnx.Rngs,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.q = nnx.Linear(dim, dim, use_bias=qkv_bias, rngs=rngs)
        self.k = nnx.Linear(dim, dim, use_bias=qkv_bias, rngs=rngs)
        self.v = nnx.Linear(dim, dim, use_bias=qkv_bias, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, rngs=rngs)
        self.attn_drop = nnx.Dropout(attn_drop) if attn_drop > 0 else None
        self.proj_drop = nnx.Dropout(proj_drop) if proj_drop > 0 else None

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        mask: Optional[jnp.ndarray] = None,
        rngs: Optional[nnx.Rngs] = None,
    ) -> jnp.ndarray:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        attn = jnp.einsum("bhnd,bhmd->bhnm", q, k) * self.scale
        if mask is not None:
            attn = jnp.where(mask[:, None, :, :], attn, jnp.full_like(attn, -1e30))
        attn = nnx.softmax(attn, axis=-1)
        if self.attn_drop is not None:
            attn = self.attn_drop(attn, rngs=rngs)
        out = (
            jnp.einsum("bhnm,bhmd->bhnd", attn, v)
            .transpose(0, 2, 1, 3)
            .reshape(B, N, C)
        )
        out = self.proj(out)
        if self.proj_drop is not None:
            out = self.proj_drop(out, rngs=rngs)
        return out


class CrossAttention(nnx.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        rngs: nnx.Rngs,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.q = nnx.Linear(dim, dim, use_bias=qkv_bias, rngs=rngs)
        self.k = nnx.Linear(dim, dim, use_bias=qkv_bias, rngs=rngs)
        self.v = nnx.Linear(dim, dim, use_bias=qkv_bias, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, rngs=rngs)
        self.attn_drop = nnx.Dropout(attn_drop) if attn_drop > 0 else None
        self.proj_drop = nnx.Dropout(proj_drop) if proj_drop > 0 else None

    def __call__(
        self,
        x_q: jnp.ndarray,
        x_kv: jnp.ndarray,
        *,
        mask: Optional[jnp.ndarray] = None,
        rngs: Optional[nnx.Rngs] = None,
    ) -> jnp.ndarray:
        B, Nq, C = x_q.shape
        B2, Nk, C2 = x_kv.shape
        assert B == B2 and C == C2
        q = (
            self.q(x_q)
            .reshape(B, Nq, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k(x_kv)
            .reshape(B, Nk, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v(x_kv)
            .reshape(B, Nk, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        attn = jnp.einsum("bhqd,bhkd->bhqk", q, k) * self.scale
        if mask is not None:
            m = mask[:, None, None, :]
            attn = jnp.where(m, attn, jnp.full_like(attn, -1e30))
        attn = nnx.softmax(attn, axis=-1)
        if self.attn_drop is not None:
            attn = self.attn_drop(attn, rngs=rngs)
        out = (
            jnp.einsum("bhqk,bhkd->bhqd", attn, v)
            .transpose(0, 2, 1, 3)
            .reshape(B, Nq, C)
        )
        out = self.proj(out)
        if self.proj_drop is not None:
            out = self.proj_drop(out, rngs=rngs)
        return out


class AdaLNModulator(nnx.Module):
    def __init__(self, dim: int, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(dim, dim, rngs=rngs)
        self.fc2 = nnx.Linear(dim, 6 * dim, rngs=rngs)
        self.fc2.kernel.value = jnp.zeros_like(self.fc2.kernel.value)
    # Leave bias at its default initialization

    def __call__(self, t: jnp.ndarray):
        h = nnx.silu(self.fc1(t))
        h = self.fc2(h)
        return jnp.split(h, 6, axis=-1)


class DiTBlock(nnx.Module):
    """Single-stream DiT block with spatial & temporal modes."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        rngs: nnx.Rngs,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        self.norm1 = nnx.LayerNorm(hidden_size, epsilon=1e-6, rngs=rngs)
        self.norm3 = nnx.LayerNorm(hidden_size, epsilon=1e-6, rngs=rngs)
        self.self_attn = MultiHeadAttention(
            hidden_size,
            num_heads,
            rngs=rngs,
            qkv_bias=True,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.cross = CrossAttention(
            hidden_size,
            num_heads,
            rngs=rngs,
            qkv_bias=True,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.mlp = MLP(hidden_size, int(hidden_size * 4.0), hidden_size, rngs=rngs)
        self.mod = AdaLNModulator(hidden_size, rngs=rngs)

    def __call__(
        self,
        x: jnp.ndarray,
        t: jnp.ndarray,
        v_fea: jnp.ndarray,
        cond_fea: jnp.ndarray,
        *,
        shape: Tuple[int, int, int, int],
        block_type: str,
        rngs: Optional[nnx.Rngs] = None,
    ) -> jnp.ndarray:
        B, T, N, _ = shape
        # pre cross-attn with video features
        x = x + self.cross(x, v_fea, rngs=rngs)
        if block_type == "spatial":
            # transpose(0,1,2,3) is a no-op; avoid it for clarity
            # x_bta = x.reshape(B, T, N, -1).reshape(B * T, N, -1)
            x_bt = x.reshape(B * T, N, -1)
            # print("is xbt same as xbta?", jnp.all(x_bt == x_bta))
            t_bt = jnp.repeat(t, T, axis=0)  # [B*T, C]
            cond_bt = cond_fea.reshape(B * T, -1)  # [B*T, C]
            t_bt = t_bt + cond_bt
            s_msa, sc_msa, g_msa, s_mlp, sc_mlp, g_mlp = self.mod(t_bt)
            x_tmp = modulate_spatial(self.norm1(x_bt), s_msa, sc_msa)
            y = self.self_attn(x_tmp, rngs=rngs)
            # Use residual scaling 1+g to avoid zeroed gradients at init
            x_bt = x_bt + (1.0 + g_msa[:, None, :]) * y
            x_bt = x_bt + (1.0 + g_mlp[:, None, :]) * self.mlp(
                modulate_spatial(self.norm3(x_bt), s_mlp, sc_mlp)
            )
            x = x_bt.reshape(B, T, N, -1).reshape(B, T * N, -1)
        elif block_type == "temporal":
            x_bn = x.reshape(B, T, N, -1).transpose(0, 2, 1, 3).reshape(B * N, T, -1)
            t_bn = jnp.repeat(t, N, axis=0)  # [B*N, C]
            cond_bn = jnp.repeat(cond_fea, N, axis=0)  # [B*N, T, C]
            t_bn = t_bn[:, None, :] + cond_bn  # [B*N, T, C]
            s_msa, sc_msa, g_msa, s_mlp, sc_mlp, g_mlp = self.mod(t_bn)
            x_tmp = modulate_temporal(self.norm1(x_bn), s_msa, sc_msa)
            y = self.self_attn(x_tmp, rngs=rngs)
            x_bn = x_bn + (1.0 + g_msa) * y
            x_bn = x_bn + (1.0 + g_mlp) * self.mlp(
                modulate_temporal(self.norm3(x_bn), s_mlp, sc_mlp)
            )
            x = x_bn.reshape(B, N, T, -1).transpose(0, 2, 1, 3).reshape(B, T * N, -1)
        else:
            raise ValueError("block_type must be 'spatial' or 'temporal'")
        return x


class CompLayer(nnx.Module):
    def __init__(
        self, hidden_size: int, out_channels: int, num_heads: int, *, rngs: nnx.Rngs
    ):
        self.cross = CrossAttention(hidden_size, num_heads, rngs=rngs, qkv_bias=True)
        self.norm = nnx.LayerNorm(hidden_size, epsilon=1e-6, rngs=rngs)
        self.out = nnx.Linear(hidden_size, out_channels, rngs=rngs)
        self.mod = nnx.Linear(hidden_size, 2 * hidden_size, rngs=rngs)
        self.mod.kernel.value = jnp.zeros_like(self.mod.kernel.value)
    # Leave bias at its default initialization

    def __call__(
        self,
        x: jnp.ndarray,
        v: jnp.ndarray,
        t: jnp.ndarray,
        cond: jnp.ndarray,
        *,
        shape: Tuple[int, int, int, int],
        rngs: Optional[nnx.Rngs] = None,
    ) -> jnp.ndarray:
        B, T, N, _ = shape
        x = x + self.cross(x, v, rngs=rngs)
        x_bt = x.reshape(B, T, N, -1).reshape(B * T, N, -1)
        t_bt = jnp.repeat(t, T, axis=0)
        cond_bt = cond.reshape(B * T, -1)
        t_bt = t_bt + cond_bt
        t_bt = nnx.silu(t_bt)
        shift, scale = jnp.split(self.mod(t_bt), 2, axis=-1)
        x_bt = modulate_spatial(self.norm(x_bt), shift, scale)
        x = x_bt.reshape(B, T, N, -1)
        return self.out(x)


class FinalLayer(nnx.Module):
    def __init__(
        self, hidden_size: int, out_channels: int, num_heads: int, *, rngs: nnx.Rngs
    ):
        self.in_linear = nnx.Linear(out_channels, hidden_size, rngs=rngs)
        self.cross = CrossAttention(hidden_size, num_heads, rngs=rngs, qkv_bias=True)
        self.norm = nnx.LayerNorm(hidden_size, epsilon=1e-6, rngs=rngs)
        self.out = nnx.Linear(hidden_size, out_channels, rngs=rngs)
        self.mod = nnx.Linear(hidden_size, 2 * hidden_size, rngs=rngs)

    def __call__(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        t: jnp.ndarray,
        cond: jnp.ndarray,
        *,
        shape: Tuple[int, int, int, int],
        rngs: Optional[nnx.Rngs] = None,
    ) -> jnp.ndarray:
        B, T, N, _ = shape
        y = self.in_linear(y)
        x = x + self.cross(x, y, rngs=rngs)
        x_bt = x.reshape(B, T, N, -1).reshape(B * T, N, -1)
        t_bt = jnp.repeat(t, T, axis=0)
        cond_bt = cond.reshape(B * T, -1)
        t_bt = t_bt + cond_bt
        t_bt = nnx.silu(t_bt)
        shift, scale = jnp.split(self.mod(t_bt), 2, axis=-1)
        x_bt = modulate_spatial(self.norm(x_bt), shift, scale)
        x = x_bt.reshape(B, T, N, -1)
        return self.out(x)


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


class ActionEmbedder(nnx.Module):
    def __init__(self, hidden_size: int, input_size: int, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(input_size, hidden_size, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_size, hidden_size, rngs=rngs)

    def __call__(self, act: jnp.ndarray) -> jnp.ndarray:
        return self.fc2(nnx.silu(self.fc1(act)))


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
        self.attn = MultiHeadAttention(
            dim,
            num_heads,
            rngs=rngs,
            qkv_bias=True,
            attn_drop=dropout,
            proj_drop=dropout,
        )
        self.norm2 = nnx.LayerNorm(dim, epsilon=1e-6, rngs=rngs)
        self.ffn = MLP(dim, int(dim * mlp_ratio), dim, rngs=rngs)
        self.drop = nnx.Dropout(dropout) if dropout > 0 else None

    def __call__(
        self, x: jnp.ndarray, *, rngs: Optional[nnx.Rngs] = None
    ) -> jnp.ndarray:
        h = self.attn(self.norm1(x), rngs=rngs)
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
            {f"block_{i}": TransformerBlock(dim, num_heads, mlp_ratio=2.0, dropout=0.0, rngs=rngs) for i in range(depth)}
        )
        # self.layers = [
        #     TransformerBlock(dim, num_heads, mlp_ratio=2.0, dropout=0.0, rngs=rngs)
        #     for _ in range(depth)
        # ]

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
        # for layer in self.layers:
        #     x = layer(x, rngs=rngs)
        g = x[:, 0, :]
        return g.reshape(B, N, C)


# -----------------------------------------------------------------------------
# The DiffusionTransformer (NNX, single-stream)
# -----------------------------------------------------------------------------


class DiffusionTransformer(nnx.Module):
    def __init__(
        self,
        in_channel: int = 3,
        dim: int = 768,
        num_heads: int = 8,
        n_layers: int = 12,
        *,
        rngs: nnx.Rngs,
    ):
        self.in_channel = in_channel
        self.dim = dim
        self.num_heads = num_heads
        # Token embedder (single stream)
        self.x_embedder = nnx.Linear(in_channel, dim, rngs=rngs)
        # History encoder
        self.video_encoder = VideoTransformer(
            in_channel=in_channel, dim=dim, depth=8, num_heads=num_heads, rngs=rngs
        )
        # Time + action conditioning
        self.time_encoder = TimestepEmbedder(dim, freq_dim=256, rngs=rngs)
        self.action_encoder = ActionEmbedder(dim, input_size=7, rngs=rngs)
        # Stacked DiT blocks (alternate spatial/temporal)
        self.n_layers = n_layers
        self.blocks = nnx.Dict({f"block_{i}": DiTBlock(dim, num_heads, rngs=rngs) for i in range(self.n_layers)})
        # Output heads
        self.comp = CompLayer(
            dim, out_channels=in_channel, num_heads=num_heads, rngs=rngs
        )
        self.final = FinalLayer(
            dim, out_channels=in_channel, num_heads=num_heads, rngs=rngs
        )

    def __call__(
        self,
        x_noisy: jnp.ndarray,
        lc_his: jnp.ndarray,
        actions: jnp.ndarray,
        time: jnp.ndarray,
        *,
        rngs: Optional[nnx.Rngs] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Args:
          x_noisy: [B, T, N, C_in]
          lc_his: [B, T, N, C_in]
          actions: [B, T, 7]
          time: [B]
        Returns: (y, y_tmp) each [B, T, N, C_in]
        """
        B, T, N, Cin = x_noisy.shape
        shape = (B, T, N, Cin)
        # Tokens + position
        x = self.x_embedder(x_noisy).reshape(B, T * N, self.dim)
        pos = get_2d_sincos_pos_embed(self.dim, (T, N))
        x = x + pos
        # Encode history latents -> [B, N, C]
        v_fea = self.video_encoder(lc_his, rngs=rngs)
        # Conditioning
        t_fea = self.time_encoder(jnp.log(time + 1e-8))  # [B, C]
        act_fea = self.action_encoder(actions)  # [B, T, C]
        # DiT stack
        for i in range(self.n_layers):
            mode = "spatial" if i % 2 == 0 else "temporal"
            x = self.blocks[f"block_{i}"](x, t_fea, v_fea, act_fea, shape=shape, block_type=mode, rngs=rngs)
            # x = block(x, t_fea, v_fea, act_fea, shape=shape, block_type=mode, rngs=rngs)
        # Heads
        y_tmp = self.comp(x, v_fea, t_fea, act_fea, shape=shape, rngs=rngs)
        y = y_tmp + self.final(
            x, y_tmp.reshape(B, T * N, -1), t_fea, act_fea, shape=shape, rngs=rngs
        )
        return y, y_tmp


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    key_params, key_dropout = jax.random.split(key)
    rngs = nnx.Rngs(params=key_params, dropout=key_dropout)
    B, T, N, Cin = 2, 4, 256, 4
    x_noisy = jax.random.normal(key, (B, T, N, Cin))
    lc_his = jax.random.normal(key, (B, T, N, Cin))
    actions = jax.random.normal(key, (B, T, 7))
    t_scalar = jnp.ones((B,), dtype=jnp.float32) * 0.5
    model = DiffusionTransformer(
        in_channel=Cin, dim=256, num_heads=8, n_layers=6, rngs=rngs
    )
    y, ytmp = model(x_noisy, lc_his, actions, t_scalar, rngs=rngs)
    print("y:", y.shape, "tmp:", ytmp.shape)
