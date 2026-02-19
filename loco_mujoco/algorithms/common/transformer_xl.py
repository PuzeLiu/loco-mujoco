import math
from typing import List, Optional, Tuple, Any

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


def _rel_positional_encoding(length: int, dim: int, dtype: Any, reverse: bool = True) -> jnp.ndarray:
    """Sinusoidal relative positional encoding."""
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
    return pos_emb.astype(dtype)


def _rel_shift(x: jnp.ndarray) -> jnp.ndarray:
    """Perform relative shift to align attention scores."""
    b, h, t, k = x.shape
    zero_pad = jnp.zeros((b, h, t, 1), dtype=x.dtype)
    x_padded = jnp.concatenate([zero_pad, x], axis=3)  # (b, h, t, k+1)
    x_padded = x_padded.reshape(b, h, k + 1, t)
    x = x_padded[:, :, 1:, :].reshape(b, h, t, k)
    return x


def init_mems(n_layers: int, mem_len: int, batch_size: int, model_dim: int, dtype: Any = jnp.float32) -> List[jnp.ndarray]:
    """Initialize Transformer-XL memories to zeros."""
    if mem_len <= 0:
        return [jnp.zeros((0, batch_size, model_dim), dtype=dtype) for _ in range(n_layers)]
    return [jnp.zeros((mem_len, batch_size, model_dim), dtype=dtype) for _ in range(n_layers)]


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
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    def setup(self):
        self.q_proj = nn.Dense(self.model_dim, use_bias=False, dtype=self.dtype, param_dtype=self.param_dtype)
        self.k_proj = nn.Dense(self.model_dim, use_bias=False, dtype=self.dtype, param_dtype=self.param_dtype)
        self.r_proj = nn.Dense(self.model_dim, use_bias=False, dtype=self.dtype, param_dtype=self.param_dtype)
        self.v_proj = nn.Dense(self.model_dim, use_bias=False, dtype=self.dtype, param_dtype=self.param_dtype)
        self.o_proj = nn.Dense(self.model_dim, use_bias=False, dtype=self.dtype, param_dtype=self.param_dtype)
        self.attn_dropout = nn.Dropout(rate=self.dropout)
        head_dim = self.model_dim // self.n_heads
        u_init = nn.initializers.normal(stddev=0.02)
        v_init = nn.initializers.normal(stddev=0.02)
        self.u = self.param("u", u_init, (self.n_heads, head_dim))
        self.v = self.param("v", v_init, (self.n_heads, head_dim))

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
        k_len = cat.shape[0]

        q = _split_heads(self.q_proj(x), self.n_heads)
        k = _split_heads(self.k_proj(cat), self.n_heads)
        v = _split_heads(self.v_proj(cat), self.n_heads)

        acc_dtype = jnp.float32
        q_f = q.astype(acc_dtype)
        k_f = k.astype(acc_dtype)
        v_f = v.astype(acc_dtype)
        scale = jnp.array(1.0 / math.sqrt(q.shape[-1]), dtype=acc_dtype)
        r = _rel_positional_encoding(k_len, self.model_dim, dtype=self.dtype, reverse=True)
        r = self.r_proj(r)
        head_dim = self.model_dim // self.n_heads
        r = r.reshape(k_len, self.n_heads, head_dim)
        r = jnp.transpose(r, (1, 0, 2))  # (H, K, Hd)

        u = self.u.astype(acc_dtype)
        v_bias = self.v.astype(acc_dtype)
        ac = jnp.einsum("bhqd,bhkd->bhqk", q_f + u[None, :, None, :], k_f)
        bd = jnp.einsum("bhqd,hkd->bhqk", q_f + v_bias[None, :, None, :], r.astype(acc_dtype))
        bd = _rel_shift(bd)
        attn_scores = (ac + bd) * scale

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
        if mem_len is None:
            effective_mem_len = 0 if cache is None else cache[0].shape[2]
        else:
            effective_mem_len = int(mem_len)
        if effective_mem_len <= 0:
            q = self.q_proj(x_t)  # (B, D)
            k = self.k_proj(x_t)
            v = self.v_proj(x_t)
            b = q.shape[0]
            head_dim = self.model_dim // self.n_heads
            q = q.reshape(b, self.n_heads, head_dim)
            k = k.reshape(b, self.n_heads, 1, head_dim)
            v = v.reshape(b, self.n_heads, 1, head_dim)

            acc_dtype = jnp.float32
            q_f = q.astype(acc_dtype)
            k_f = k.astype(acc_dtype)
            v_f = v.astype(acc_dtype)
            scale = jnp.array(1.0 / math.sqrt(head_dim), dtype=acc_dtype)
            attn_scores = jnp.einsum("bhd,bhkd->bhk", q_f, k_f) * scale
            attn = jax.nn.softmax(attn_scores, axis=-1)
            out = jnp.einsum("bhk,bhkd->bhd", attn, v_f)
            out = out.reshape(b, self.model_dim)
            out = self.o_proj(out.astype(self.dtype))
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

        acc_dtype = jnp.float32
        scale = jnp.array(1.0 / math.sqrt(head_dim), dtype=acc_dtype)

        r = _rel_positional_encoding(effective_mem_len, self.model_dim, dtype=self.dtype, reverse=False)
        r = self.r_proj(r)
        r = r.reshape(effective_mem_len, self.n_heads, head_dim)
        r = jnp.transpose(r, (1, 0, 2))  # (H, L, Hd)
        u = self.u.astype(acc_dtype)
        v_bias = self.v.astype(acc_dtype)

        def _attend_single(q_b, k_b, v_b, cl):
            q_b = q_b.astype(acc_dtype)
            k_b = k_b.astype(acc_dtype)
            v_b = v_b.astype(acc_dtype)

            def _body(i, state):
                max_logit, sum_exp, out = state
                def _include(s):
                    m, s_exp, o = s
                    rel_idx = cl - 1 - i
                    r_i = r[:, rel_idx, :].astype(acc_dtype)
                    logit = (
                        jnp.einsum("hd,hd->h", q_b + u, k_b[:, i, :]).astype(acc_dtype)
                        + jnp.einsum("hd,hd->h", q_b + v_bias, r_i).astype(acc_dtype)
                    ) * scale
                    new_m = jnp.maximum(m, logit)
                    exp_logit = jnp.exp(logit - new_m)
                    s_exp = s_exp * jnp.exp(m - new_m) + exp_logit
                    o = o * jnp.exp(m - new_m)[:, None] + v_b[:, i, :] * exp_logit[:, None]
                    return new_m, s_exp, o

                return jax.lax.cond(i < cl, _include, lambda s: s, state)

            init_max = jnp.full((self.n_heads,), -jnp.inf, dtype=acc_dtype)
            init_sum = jnp.zeros((self.n_heads,), dtype=acc_dtype)
            init_out = jnp.zeros((self.n_heads, head_dim), dtype=acc_dtype)
            max_logit, sum_exp, out = jax.lax.fori_loop(
                0, effective_mem_len, _body, (init_max, init_sum, init_out)
            )
            out = out / (sum_exp[:, None] + 1e-8)
            return out

        out = jax.vmap(_attend_single, in_axes=(0, 0, 0, 0))(q, k_cache, v_cache, cache_len)
        out = out.reshape(b, self.model_dim)
        out = self.o_proj(out.astype(self.dtype))
        return out, (k_cache, v_cache, cache_len)


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
                 attn_mask: Optional[jnp.ndarray],
                 train: bool) -> jnp.ndarray:
        h = self.ln1(x.astype(jnp.float32)).astype(self.dtype)
        h = self.attn(h, mem, attn_mask, train)
        h = self.dropout_layer(h, deterministic=not train)
        x = x + h

        h = self.ln2(x.astype(jnp.float32)).astype(self.dtype)
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
        h = self.ln1(x_t.astype(jnp.float32)).astype(self.dtype)
        h, cache = self.attn.step(h, cache, mem_len, train)
        x_t = x_t + h
        h2 = self.ln2(x_t.astype(jnp.float32)).astype(self.dtype)
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
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    def setup(self):
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
            h = layer(h, mem, attn_mask, train)
            if self.mem_len > 0:
                if mem is not None and mem.shape[0] > 0:
                    cat = jnp.concatenate([mem, h], axis=0)
                else:
                    cat = h
                new_mems.append(cat[-self.mem_len:])
            else:
                new_mems.append(jnp.zeros((0, h.shape[1], h.shape[2]), dtype=h.dtype))
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
