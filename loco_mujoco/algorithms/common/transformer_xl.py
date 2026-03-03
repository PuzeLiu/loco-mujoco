import math
from typing import List, Optional, Any, Tuple

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


def _abs_positional_encoding(length: int, dim: int, dtype: Any, reverse: bool = True) -> jnp.ndarray:
    """Sinusoidal absolute positional encoding."""
    if length <= 0:
        return jnp.zeros((0, dim), dtype=dtype)
    inv_freq = 1.0 / (10000 ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    if reverse:
        pos_seq = jnp.arange(length - 1, -1, -1.0, dtype=jnp.float32)
    else:
        pos_seq = jnp.arange(length, dtype=jnp.float32)
    sinusoid_inp = jnp.einsum("i,j->ij", pos_seq, inv_freq)
    pos_emb = jnp.concatenate([jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)], axis=-1)
    if pos_emb.shape[1] < dim:
        pad = jnp.zeros((length, dim - pos_emb.shape[1]), dtype=pos_emb.dtype)
        pos_emb = jnp.concatenate([pos_emb, pad], axis=-1)


def init_mems(n_layers: int, mem_len: int, batch_size: int, model_dim: int, dtype: Any = jnp.float32) -> List[jnp.ndarray]:
    """Initialize Transformer-XL memories to zeros."""
    if mem_len <= 0:
        return [jnp.zeros((0, batch_size, model_dim), dtype=dtype) for _ in range(n_layers)]
    return [jnp.zeros((mem_len, batch_size, model_dim), dtype=dtype) for _ in range(n_layers)]


class MultiHeadSelfAttentionXL(nn.Module):
    """Multi-head self-attention with optional memory."""

    model_dim: int
    n_heads: int
    dropout: float = 0.0
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    def setup(self):
        self.q_proj = nn.Dense(self.model_dim, use_bias=False, dtype=self.dtype, param_dtype=self.param_dtype)
        self.k_proj = nn.Dense(self.model_dim, use_bias=False, dtype=self.dtype, param_dtype=self.param_dtype)
        self.v_proj = nn.Dense(self.model_dim, use_bias=False, dtype=self.dtype, param_dtype=self.param_dtype)
        self.o_proj = nn.Dense(self.model_dim, use_bias=False, dtype=self.dtype, param_dtype=self.param_dtype)
        self.attn_dropout = nn.Dropout(rate=self.dropout)

    def __call__(self,
                 x: jnp.ndarray,
                 mem: Optional[jnp.ndarray],
                 mem_pos_emb: Optional[jnp.ndarray],
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
            if mem_pos_emb is not None:
                mem = mem + mem_pos_emb[:, None, :]
            cat = jnp.concatenate([mem, x], axis=0)
        else:
            cat = x

        q = _split_heads(self.q_proj(x), self.n_heads)
        k = _split_heads(self.k_proj(cat), self.n_heads)
        v = _split_heads(self.v_proj(cat), self.n_heads)

        acc_dtype = jnp.float32
        q_f = q.astype(acc_dtype)
        k_f = k.astype(acc_dtype)
        v_f = v.astype(acc_dtype)
        scale = jnp.array(1.0 / math.sqrt(q.shape[-1]), dtype=acc_dtype)
        attn_scores = jnp.einsum("bhqd,bhkd->bhqk", q_f, k_f) * scale

        if attn_mask is not None:
            # attn_mask: (B, T, K) -> (B, 1, T, K)
            mask = attn_mask[:, None, :, :]
            attn_scores = jnp.where(mask, attn_scores, -1e9)

        attn = jax.nn.softmax(attn_scores, axis=-1)
        attn = self.attn_dropout(attn, deterministic=not train)

        out = jnp.einsum("bhqk,bhkd->bhqd", attn, v_f)
        out = _merge_heads(out)
        out = out.astype(self.dtype)
        return self.o_proj(out)



class TransformerXLLayer(nn.Module):
    model_dim: int
    n_heads: int
    ff_dim: int
    dropout: float = 0.0
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    def setup(self):
        self.ln1 = nn.LayerNorm(dtype=jnp.float32, param_dtype=self.param_dtype)
        self.attn = MultiHeadSelfAttentionXL(
            self.model_dim,
            self.n_heads,
            self.dropout,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.ln2 = nn.LayerNorm(dtype=jnp.float32, param_dtype=self.param_dtype)
        self.ffn1 = nn.Dense(self.ff_dim, dtype=self.dtype, param_dtype=self.param_dtype)
        self.ffn2 = nn.Dense(self.model_dim, dtype=self.dtype, param_dtype=self.param_dtype)
        self.dropout_layer = nn.Dropout(rate=self.dropout)

    def __call__(self,
                 x: jnp.ndarray,
                 mem: Optional[jnp.ndarray],
                 mem_pos_emb: Optional[jnp.ndarray],
                 attn_mask: Optional[jnp.ndarray],
                 train: bool) -> jnp.ndarray:
        h = self.ln1(x.astype(jnp.float32)).astype(self.dtype)
        h = self.attn(h, mem, mem_pos_emb, attn_mask, train)
        h = self.dropout_layer(h, deterministic=not train)
        x = x + h

        h = self.ln2(x.astype(jnp.float32)).astype(self.dtype)
        h = self.ffn1(h)
        h = jax.nn.gelu(h)
        h = self.dropout_layer(h, deterministic=not train)
        h = self.ffn2(h)
        h = self.dropout_layer(h, deterministic=not train)
        return x + h



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
    positional_encoding: str = "absolute"
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    def setup(self):
        if self.positional_encoding not in ("absolute", "learned", "none", ""):
            raise ValueError(f"Unsupported positional encoding: {self.positional_encoding}")
        if self.positional_encoding == "learned":
            max_len = max(self.mem_len + 1, 1)
            self.pos_embedding = self.param(
                "pos_embedding",
                nn.initializers.normal(stddev=0.02),
                (max_len, self.model_dim),
            )
        self.layers = [
            TransformerXLLayer(
                self.model_dim,
                self.n_heads,
                self.ff_dim,
                self.dropout,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )
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
            mem_pos_emb = None
            if mem is not None and mem.shape[0] > 0 and self.positional_encoding not in ("none", ""):
                mem_len = mem.shape[0]
                if self.positional_encoding == "absolute":
                    mem_pos_emb = _abs_positional_encoding(mem_len, self.model_dim, dtype=self.dtype, reverse=True)
                else:
                    max_len = self.pos_embedding.shape[0]
                    indices = jnp.arange(mem_len)
                    indices = jnp.minimum(indices, max_len - 1)
                    mem_pos_emb = self.pos_embedding[indices]
            h = layer(h, mem, mem_pos_emb, attn_mask, train)
            if self.mem_len > 0:
                if mem is not None and mem.shape[0] > 0:
                    cat = jnp.concatenate([mem, h], axis=0)
                else:
                    cat = h
                new_mems.append(cat[-self.mem_len:])
            else:
                new_mems.append(jnp.zeros((0, h.shape[1], h.shape[2]), dtype=h.dtype))
        return h, new_mems
