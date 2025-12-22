from typing import Tuple, Optional

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


def make_block_causal_mask_with_spatial(
    T: int, N: int, num_history: int
) -> jnp.ndarray:
    """
    Creates a mask for Cross-Attention with explicit spatial dimension handling.

    Args:
        T: Number of future timesteps
        N: Number of spatial tokens per timestep
        num_history: Number of history tokens

    Returns:
        mask: Boolean tensor shape (1, 1, T*N, num_history + T)
    """
    # Query time indices: each spatial token in a time block gets the same time index
    # e.g., T=2, N=3 -> [0, 0, 0, 1, 1, 1]
    query_time_idx = jnp.repeat(jnp.arange(T), N)

    # History tokens are always visible: (T*N, num_history)
    history_mask = jnp.ones((T * N, num_history), dtype=jnp.bool_)

    # Action tokens are causally masked: (T*N, T)
    key_action_time_idx = jnp.arange(T)
    action_mask = query_time_idx[:, None] >= key_action_time_idx[None, :]

    # Concatenate: [history | actions]
    full_mask = jnp.concatenate([history_mask, action_mask], axis=1)

    return full_mask[None, None, :, :]


def modulate(x: jnp.ndarray, shift: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
    """
    AdaLN modulation:  x * (1 + scale) + shift.

    Broadcasts shift/scale over the sequence (middle) dimension.

    Args:
        x: Input tensor of shape (Batch, Seq, Dim)
        shift:  Shift tensor of shape (Batch, Dim)
        scale: Scale tensor of shape (Batch, Dim)

    Returns:
        Modulated tensor of shape (Batch, Seq, Dim)
    """
    # (Batch, Dim) -> (Batch, 1, Dim) for broadcasting over Seq dimension
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


class MultiViewVideoTransformer(nnx.Module):
    """Video Transformer that handles multiple camera views with cross-attention."""

    def __init__(
        self,
        in_channel: int,
        dim: int,
        depth: int = 8,
        num_heads: int = 8,
        num_cross_attn_layers: int = 2,  # Layers for cross-view attention
        *,
        rngs: nnx.Rngs,
    ):
        self.token = nnx.Param(jnp.zeros((1, 1, dim), dtype=jnp.float32))
        self.inp = nnx.Linear(in_channel, dim, rngs=rngs)

        # Learnable view embeddings (will be expanded dynamically)
        self.max_views = 4  # Maximum number of views supported
        self.view_embeddings = nnx.Param(
            jnp.zeros((self.max_views, 1, dim), dtype=jnp.float32)
        )

        # Self-attention blocks (within each view)
        self.blocks = nnx.Dict(
            {
                f"block_{i}": TransformerBlock(
                    dim, num_heads, mlp_ratio=2.0, dropout=0.0, rngs=rngs
                )
                for i in range(depth)
            }
        )

        # Cross-attention blocks (across views)
        self.cross_view_layers = nnx.Dict(
            {
                f"cross_{i}": CrossViewAttentionBlock(dim, num_heads, rngs=rngs)
                for i in range(num_cross_attn_layers)
            }
        )

        self.dim = dim

    def __call__(
        self,
        views: list[jnp.ndarray],  # List of [B, T, N, Cin] tensors, one per view
        *,
        rngs: Optional[nnx.Rngs] = None,
    ) -> jnp.ndarray:
        """
        Process multiple views and return fused history features.

        Args:
            views: List of tensors, each of shape [B, T, N, Cin]
                   where T is history length, N is spatial tokens, Cin is input channels

        Returns:
            Fused features of shape [B, N, C] for the primary view (first in list)
        """
        num_views = len(views)
        if num_views == 0:
            raise ValueError("At least one view must be provided")

        B, T, N, Cin = views[0].shape

        # Process each view through input projection
        view_features = []
        for v_idx, view in enumerate(views):
            x = self.inp(view)  # [B, T, N, C]

            # Add view-specific embedding
            view_emb = self.view_embeddings.value[v_idx]  # [1, C]
            x = x + view_emb[None, None, :, :]  # Broadcast to [B, T, N, C]

            # Reshape for temporal processing:  [B*N, T, C]
            x = x.reshape(B * N, T, -1)

            # Add CLS token
            cls = jnp.broadcast_to(self.token.value, (B * N, 1, self.dim))
            x = jnp.concatenate([cls, x], axis=1)  # [B*N, T+1, C]

            # Self-attention within this view
            for _, block in self.blocks.items():
                x = block(x, rngs=rngs)

            # Extract CLS token as view summary:  [B*N, C] -> [B, N, C]
            view_summary = x[:, 0, :].reshape(B, N, -1)
            view_features.append(view_summary)

        # If only one view, return directly
        if num_views == 1:
            return view_features[0]

        # Stack views for cross-attention:  [B, num_views, N, C]
        stacked = jnp.stack(view_features, axis=1)

        # Apply cross-view attention
        for _, cross_layer in self.cross_view_layers.items():
            stacked = cross_layer(stacked, rngs=rngs)

        # Return primary view features (first view, enriched with cross-view info)
        return stacked[:, 0, :, :]  # [B, N, C]


class CrossViewAttentionBlock(nnx.Module):
    """Cross-attention block that allows views to attend to each other."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 2.0,
        *,
        rngs: nnx.Rngs,
    ):
        self.norm1 = nnx.LayerNorm(dim, epsilon=1e-6, rngs=rngs)
        self.cross_attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=dim,
            qkv_features=dim,
            out_features=dim,
            decode=False,
            rngs=rngs,
        )
        self.norm2 = nnx.LayerNorm(dim, epsilon=1e-6, rngs=rngs)
        self.ffn = MLP(dim, int(dim * mlp_ratio), dim, rngs=rngs)

    def __call__(
        self, x: jnp.ndarray, *, rngs: Optional[nnx.Rngs] = None  # [B, num_views, N, C]
    ) -> jnp.ndarray:
        B, V, N, C = x.shape

        # Reshape to process each spatial position across all views
        # [B, V, N, C] -> [B*N, V, C]
        x_flat = x.transpose(0, 2, 1, 3).reshape(B * N, V, C)

        # Cross-attention: each view attends to all views
        x_norm = self.norm1(x_flat)
        attn_out = self.cross_attn(x_norm)
        x_flat = x_flat + attn_out

        # FFN
        x_flat = x_flat + self.ffn(self.norm2(x_flat))

        # Reshape back: [B*N, V, C] -> [B, V, N, C]
        return x_flat.reshape(B, N, V, C).transpose(0, 2, 1, 3)


class CrossAttention(nnx.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        rngs: nnx.Rngs,
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
            x_tmp = modulate(self.norm1(x_bt), s_msa, sc_msa)
            y = self.self_attn(x_tmp)

            x_bt = x_bt + (1.0 + g_msa[:, None, :]) * y
            x_bt = x_bt + (1.0 + g_mlp[:, None, :]) * self.mlp(
                modulate(self.norm3(x_bt), s_mlp, sc_mlp)
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
            x_tmp = modulate(self.norm1(x_bn), s_msa, sc_msa)
            y = self.self_attn(x_tmp)

            # Apply gating with explicit broadcasting
            x_bn = x_bn + (1.0 + g_msa[:, None, :]) * y
            x_bn = x_bn + (1.0 + g_mlp[:, None, :]) * self.mlp(
                modulate(self.norm3(x_bn), s_mlp, sc_mlp)
            )

            # Reshape back: (B*N, T, C) -> (B, N, T, C) -> (B, T, N, C)
            x = x_bn.reshape(B, N, T, -1).transpose(0, 2, 1, 3)

        return x


# -----------------------------------------------------------------------------
# Main Diffusion Transformer
# -----------------------------------------------------------------------------


class DiffusionTransformer(nnx.Module):
    """DiT that can handle optional action conditioning."""

    def __init__(self, in_channel, hidden_size, num_heads, n_layers, rngs):
        self.hidden_size = hidden_size
        self.x_embedder = nnx.Linear(in_channel, hidden_size, rngs=rngs)
        self.time_encoder = TimestepEmbedder(hidden_size, rngs=rngs)

        self.blocks = nnx.Dict(
            {
                f"block_{i}": DiTBlock(hidden_size, num_heads, rngs=rngs)
                for i in range(n_layers)
            }
        )

        self.final_norm = nnx.LayerNorm(hidden_size, rngs=rngs)
        self.final_linear = nnx.Linear(hidden_size, in_channel, rngs=rngs)
        self.final_linear.kernel.value = jnp.zeros_like(self.final_linear.kernel.value)

    def __call__(
        self,
        x_noisy,
        history_features,
        timestep,
        action_tokens=None,  # Optional [B, T, hidden_size]
    ):
        B, T, N, C = x_noisy.shape

        # Embed noisy input
        x = self.x_embedder(x_noisy)
        x = x.reshape(B, T * N, self.hidden_size)
        pos = get_2d_sincos_pos_embed(self.hidden_size, (T, N))
        x = x + pos
        x = x.reshape(B, T, N, self.hidden_size)

        # Time embedding
        t_fea = self.time_encoder(jnp.log(timestep + 1e-8))

        # Build context: [history | task | actions (optional)]
        context_parts = [history_features]

        num_history = history_features.shape[1]

        if action_tokens is not None:
            context_parts.append(action_tokens)
            # Use mask with spatial dimension handling
            ctx_mask = make_block_causal_mask_with_spatial(T, N, num_history)
        else:
            ctx_mask = None  # Full attention to all context tokens

        context_fea = jnp.concatenate(context_parts, axis=1)

        # Transformer blocks
        for i, (_, block) in enumerate(self.blocks.items()):
            mode = "spatial" if i % 2 == 0 else "temporal"
            x = block(
                x,
                t_fea=t_fea,
                context_fea=context_fea,
                shape=(B, T, N, C),
                block_type=mode,
                context_mask=ctx_mask,
            )

        x = self.final_norm(x)
        return self.final_linear(x)
