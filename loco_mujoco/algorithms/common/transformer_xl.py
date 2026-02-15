import math
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn


def _split_heads(x: jnp.ndarray, n_heads: int) -> jnp.ndarray:
    """Split last dim into (n_heads, head_dim) and transpose to (B, n_heads, T, head_dim)."""
    t, b, d = x.shape
    head_dim = d // n_heads
    x = x.reshape(t, b, n_heads, head_dim)
    return jnp.transpose(x, (1, 2, 0, 3))


def _merge_heads(x: jnp.ndarray) -> jnp.ndarray:
    """Merge (B, n_heads, T, head_dim) -> (T, B, D)."""
    b, h, t, d = x.shape
    x = jnp.transpose(x, (2, 0, 1, 3))
    return x.reshape(t, b, h * d)


def init_mems(n_layers: int, mem_len: int, batch_size: int, model_dim: int) -> List[jnp.ndarray]:
    """Initialize Transformer-XL memories to zeros."""
    if mem_len <= 0:
        return [jnp.zeros((0, batch_size, model_dim)) for _ in range(n_layers)]
    return [jnp.zeros((mem_len, batch_size, model_dim)) for _ in range(n_layers)]


def build_attn_mask(seq_len: int,
                    mem_len: int,
                    episode_ids: Optional[jnp.ndarray] = None,
                    mem_episode_ids: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """
    Build a causal attention mask and optionally mask across episode boundaries.

    Returns:
        mask: bool array with shape (B, seq_len, mem_len + seq_len), True = allowed.
    """
    # causal mask for current segment
    causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
    if mem_len > 0:
        mem_causal = jnp.ones((seq_len, mem_len), dtype=bool)
        causal = jnp.concatenate([mem_causal, causal], axis=1)  # (seq, mem+seq)

    if episode_ids is None:
        # broadcast to batch
        return jnp.broadcast_to(causal[None, :, :], (1, seq_len, mem_len + seq_len))

    # episode boundary mask
    # episode_ids: (seq, B), mem_episode_ids: (mem, B)
    same_curr = episode_ids[:, None, :] == episode_ids[None, :, :]
    same_curr = jnp.transpose(same_curr, (2, 0, 1))  # (B, seq, seq)

    if mem_len > 0 and mem_episode_ids is not None:
        same_mem = episode_ids[:, None, :] == mem_episode_ids[None, :, :]
        same_mem = jnp.transpose(same_mem, (2, 0, 1))  # (B, seq, mem)
        same = jnp.concatenate([same_mem, same_curr], axis=2)
    else:
        same = same_curr

    mask = same & jnp.broadcast_to(causal[None, :, :], same.shape)
    return mask


class MultiHeadSelfAttentionXL(nn.Module):
    """Multi-head self-attention with optional memory (Transformer-XL style)."""

    model_dim: int
    n_heads: int
    dropout: float = 0.0

    def setup(self):
        self.q_proj = nn.Dense(self.model_dim, use_bias=False)
        self.k_proj = nn.Dense(self.model_dim, use_bias=False)
        self.v_proj = nn.Dense(self.model_dim, use_bias=False)
        self.o_proj = nn.Dense(self.model_dim, use_bias=False)
        self.attn_dropout = nn.Dropout(rate=self.dropout)

    def __call__(self,
                 x: jnp.ndarray,
                 mem: Optional[jnp.ndarray],
                 attn_mask: Optional[jnp.ndarray],
                 train: bool) -> jnp.ndarray:
        """
        Args:
            x: (T, B, D)
            mem: (M, B, D) or None
            attn_mask: (B, T, M+T) bool or None
        """
        if mem is not None and mem.shape[0] > 0:
            mem = jax.lax.stop_gradient(mem)
            cat = jnp.concatenate([mem, x], axis=0)
        else:
            cat = x

        q = _split_heads(self.q_proj(x), self.n_heads)
        k = _split_heads(self.k_proj(cat), self.n_heads)
        v = _split_heads(self.v_proj(cat), self.n_heads)

        scale = 1.0 / math.sqrt(q.shape[-1])
        attn_scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale

        if attn_mask is not None:
            # attn_mask: (B, T, K) -> (B, 1, T, K)
            mask = attn_mask[:, None, :, :]
            attn_scores = jnp.where(mask, attn_scores, -1e9)

        attn = jax.nn.softmax(attn_scores, axis=-1)
        attn = self.attn_dropout(attn, deterministic=not train)

        out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
        out = _merge_heads(out)
        return self.o_proj(out)

    def step(self,
             x_t: jnp.ndarray,
             cache: Optional[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]],
             mem_len: Optional[int],
             train: bool) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        """
        Single-step attention with KV cache for fast inference.

        Args:
            x_t: (B, D)
            cache: (k_cache, v_cache, cache_len) with k/v (B, H, L, Hd) and cache_len (B,)
            mem_len: max cache length (overrides module mem_len if provided)
        """
        effective_mem_len = self.mem_len if mem_len is None else mem_len
        if effective_mem_len <= 0:
            q = self.q_proj(x_t)  # (B, D)
            k = self.k_proj(x_t)
            v = self.v_proj(x_t)
            b = q.shape[0]
            head_dim = self.model_dim // self.n_heads
            q = q.reshape(b, self.n_heads, head_dim)
            k = k.reshape(b, self.n_heads, 1, head_dim)
            v = v.reshape(b, self.n_heads, 1, head_dim)

            scale = 1.0 / math.sqrt(head_dim)
            attn_scores = jnp.einsum("bhd,bhkd->bhk", q, k) * scale
            attn = jax.nn.softmax(attn_scores, axis=-1)
            out = jnp.einsum("bhk,bhkd->bhd", attn, v)
            out = out.reshape(b, self.model_dim)
            out = self.o_proj(out)
            return out, None

        q = self.q_proj(x_t)  # (B, D)
        k = self.k_proj(x_t)
        v = self.v_proj(x_t)

        b = q.shape[0]
        head_dim = self.model_dim // self.n_heads
        q = q.reshape(b, self.n_heads, head_dim)
        k = k.reshape(b, self.n_heads, 1, head_dim)
        v = v.reshape(b, self.n_heads, 1, head_dim)

        if cache is None:
            k_cache = jnp.zeros((b, self.n_heads, effective_mem_len, head_dim), dtype=k.dtype)
            v_cache = jnp.zeros((b, self.n_heads, effective_mem_len, head_dim), dtype=v.dtype)
            cache_len = jnp.zeros((b,), dtype=jnp.int32)
        else:
            k_cache, v_cache, cache_len = cache

        def _update_single(kc, vc, kn, vn, cl):
            def _append():
                kc_new = jax.lax.dynamic_update_slice(kc, kn, (0, cl, 0))
                vc_new = jax.lax.dynamic_update_slice(vc, vn, (0, cl, 0))
                return kc_new, vc_new

            def _shift():
                kc_new = jnp.concatenate([kc[:, 1:, :], kn], axis=1)
                vc_new = jnp.concatenate([vc[:, 1:, :], vn], axis=1)
                return kc_new, vc_new

            kc_out, vc_out = jax.lax.cond(cl < effective_mem_len, _append, _shift)
            cl_out = jnp.minimum(cl + 1, effective_mem_len)
            return kc_out, vc_out, cl_out

        k_cache, v_cache, cache_len = jax.vmap(
            _update_single, in_axes=(0, 0, 0, 0, 0)
        )(k_cache, v_cache, k, v, cache_len)

        scale = 1.0 / math.sqrt(head_dim)
        attn_scores = jnp.einsum("bhd,bhkd->bhk", q, k_cache) * scale
        attn = jax.nn.softmax(attn_scores, axis=-1)
        out = jnp.einsum("bhk,bhkd->bhd", attn, v_cache)
        out = out.reshape(b, self.model_dim)
        out = self.o_proj(out)
        return out, (k_cache, v_cache, cache_len)


class TransformerXLLayer(nn.Module):
    model_dim: int
    n_heads: int
    ff_dim: int
    dropout: float = 0.0

    def setup(self):
        self.ln1 = nn.LayerNorm()
        self.attn = MultiHeadSelfAttentionXL(self.model_dim, self.n_heads, self.dropout)
        self.ln2 = nn.LayerNorm()
        self.ffn1 = nn.Dense(self.ff_dim)
        self.ffn2 = nn.Dense(self.model_dim)
        self.dropout_layer = nn.Dropout(rate=self.dropout)

    def __call__(self,
                 x: jnp.ndarray,
                 mem: Optional[jnp.ndarray],
                 attn_mask: Optional[jnp.ndarray],
                 train: bool) -> jnp.ndarray:
        h = self.ln1(x)
        h = self.attn(h, mem, attn_mask, train)
        h = self.dropout_layer(h, deterministic=not train)
        x = x + h

        h = self.ln2(x)
        h = self.ffn1(h)
        h = jax.nn.gelu(h)
        h = self.dropout_layer(h, deterministic=not train)
        h = self.ffn2(h)
        h = self.dropout_layer(h, deterministic=not train)
        return x + h

    def step(self,
             x_t: jnp.ndarray,
             cache: Optional[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]],
             mem_len: Optional[int],
             train: bool) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        h = self.ln1(x_t)
        h, cache = self.attn.step(h, cache, mem_len, train)
        x_t = x_t + h
        h2 = self.ln2(x_t)
        h2 = self.ffn1(h2)
        h2 = jax.nn.gelu(h2)
        h2 = self.ffn2(h2)
        x_t = x_t + h2
        return x_t, cache


class TransformerXL(nn.Module):
    """
    Transformer-XL core with segment-level recurrence.

    The recurrence is implemented by concatenating previous hidden states (memory)
    and stopping gradients through that memory, as described in Transformer-XL.
    """

    model_dim: int
    n_layers: int
    n_heads: int
    ff_dim: int
    dropout: float = 0.0
    mem_len: int = 0

    def setup(self):
        self.layers = [
            TransformerXLLayer(self.model_dim, self.n_heads, self.ff_dim, self.dropout)
            for _ in range(self.n_layers)
        ]

    def __call__(self,
                 x: jnp.ndarray,
                 mems: Optional[List[jnp.ndarray]],
                 attn_mask: Optional[jnp.ndarray],
                 train: bool) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
        new_mems = []
        h = x
        for i, layer in enumerate(self.layers):
            mem = mems[i] if mems is not None else None
            h = layer(h, mem, attn_mask, train)
            if self.mem_len > 0:
                if mem is not None and mem.shape[0] > 0:
                    cat = jnp.concatenate([mem, h], axis=0)
                else:
                    cat = h
                new_mems.append(cat[-self.mem_len:])
            else:
                new_mems.append(jnp.zeros((0, h.shape[1], h.shape[2])))
        return h, new_mems

    def step(self,
             x_t: jnp.ndarray,
             cache: Optional[List[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]],
             train: bool,
             mem_len: Optional[int] = None) -> Tuple[jnp.ndarray, List[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]]:
        """
        Single-step forward with KV-cache for fast inference.

        Args:
            x_t: (B, D)
            cache: list of (k_cache, v_cache, cache_len) per layer
        """
        effective_mem_len = self.mem_len if mem_len is None else mem_len
        new_cache = []
        h = x_t
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            h, layer_cache = layer.step(h, layer_cache, effective_mem_len, train)
            new_cache.append(layer_cache)
        return h, new_cache
