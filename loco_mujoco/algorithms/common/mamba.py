from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn


def mamba_dims(model_dim: int, ff_dim: int, n_heads: int) -> Tuple[int, int, int]:
    expand = 1
    if ff_dim is not None and model_dim > 0:
        expand = max(1, min(2, ff_dim // model_dim))
    d_inner = model_dim * expand
    d_state = max(8, int(n_heads))
    d_conv = 4
    return d_inner, d_state, d_conv


def _init_a_log(key, shape, dtype=jnp.float32):
    d_inner, d_state = shape
    vals = jnp.log(jnp.arange(1, d_state + 1, dtype=dtype))
    return jnp.tile(vals[None, :], (d_inner, 1))


class MambaLayer(nn.Module):
    model_dim: int
    d_inner: int
    d_state: int
    d_conv: int
    dropout: float = 0.0
    dt_rank: int = 1
    dt_min: float = 1e-4
    dt_max: float = 1.0

    def setup(self):
        self.norm = nn.LayerNorm(dtype=jnp.float32)
        self.in_proj = nn.Dense(self.d_inner * 2, use_bias=False, dtype=jnp.float32)
        self.x_proj = nn.Dense(
            self.dt_rank + 2 * self.d_state,
            use_bias=False,
            dtype=jnp.float32,
        )
        self.dt_proj = nn.Dense(self.d_inner, use_bias=True, dtype=jnp.float32)
        self.out_proj = nn.Dense(self.model_dim, dtype=jnp.float32)
        self.dropout_layer = nn.Dropout(rate=self.dropout)

        conv_init = nn.initializers.normal(stddev=0.02)
        self.conv_kernel = self.param("conv_kernel", conv_init, (self.d_conv, self.d_inner))
        self.conv_bias = self.param("conv_bias", nn.initializers.zeros, (self.d_inner,))
        self.A_log = self.param("A_log", _init_a_log, (self.d_inner, self.d_state))
        self.D = self.param("D", nn.initializers.ones, (self.d_inner,))

    def _init_state(self, batch_size: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        conv_len = max(self.d_conv - 1, 0)
        conv_state = jnp.zeros((batch_size, conv_len, self.d_inner), dtype=jnp.float32)
        ssm_state = jnp.zeros((batch_size, self.d_inner, self.d_state), dtype=jnp.float32)
        return conv_state, ssm_state

    def _conv_step(self, x_t: jnp.ndarray, conv_state: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if self.d_conv <= 1:
            conv_out = x_t * self.conv_kernel[0] + self.conv_bias
            return conv_out, conv_state

        x_window = jnp.concatenate([conv_state, x_t[:, None, :]], axis=1)
        conv_out = jnp.sum(x_window * self.conv_kernel[None, :, :], axis=1) + self.conv_bias
        new_conv_state = x_window[:, 1:, :]
        return conv_out, new_conv_state

    def _ssm_step(
        self,
        u_t: jnp.ndarray,
        ssm_state: jnp.ndarray,
        dt: jnp.ndarray,
        b_t: jnp.ndarray,
        c_t: jnp.ndarray,
        a: jnp.ndarray,
        d: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        acc_dtype = jnp.float32
        u_f = u_t.astype(acc_dtype)
        dt_f = dt.astype(acc_dtype)
        b_f = b_t.astype(acc_dtype)
        c_f = c_t.astype(acc_dtype)
        a_f = a.astype(acc_dtype)
        d_f = d.astype(acc_dtype)
        ssm_f = ssm_state.astype(acc_dtype)

        d_a = jnp.exp(dt_f[:, :, None] * a_f[None, :, :])
        d_b = dt_f[:, :, None] * b_f[:, None, :]
        ssm_f = ssm_f * d_a + u_f[:, :, None] * d_b
        y_t = jnp.sum(ssm_f * c_f[:, None, :], axis=-1) + d_f[None, :] * u_f

        ssm_state = ssm_f.astype(ssm_state.dtype)
        return y_t, ssm_state

    def _clamp_dt(self, dt: jnp.ndarray) -> jnp.ndarray:
        return jnp.clip(dt, a_min=self.dt_min, a_max=self.dt_max)

    def __call__(
        self,
        x: jnp.ndarray,
        state: Optional[Tuple[jnp.ndarray, jnp.ndarray]],
        reset_mask: Optional[jnp.ndarray],
        train: bool,
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        t, b = x.shape[0], x.shape[1]
        h = self.norm(x.astype(jnp.float32))
        xz = self.in_proj(h)
        x_in, z = jnp.split(xz, 2, axis=-1)
        # Prime parameter creation outside scan to avoid tracer leaks.
        _ = self.x_proj(jnp.zeros((1, self.d_inner), dtype=jnp.float32))
        _ = self.dt_proj(jnp.zeros((1, self.dt_rank), dtype=jnp.float32))

        if state is None:
            conv_state, ssm_state = self._init_state(b)
        else:
            conv_state, ssm_state = state

        if reset_mask is None:
            reset_mask = jnp.zeros((t, b), dtype=bool)

        a = -jnp.exp(self.A_log)
        d = self.D

        def _step(carry, inputs):
            conv_state, ssm_state = carry
            x_t, z_t, reset_t = inputs

            reset = reset_t[:, None, None]
            conv_state = jnp.where(reset, jnp.zeros_like(conv_state), conv_state)
            ssm_state = jnp.where(reset, jnp.zeros_like(ssm_state), ssm_state)

            conv_out, conv_state = self._conv_step(x_t, conv_state)
            u_t = jax.nn.silu(conv_out)

            x_proj = self.x_proj(u_t)
            dt_part, b_t, c_t = jnp.split(
                x_proj,
                [self.dt_rank, self.dt_rank + self.d_state],
                axis=-1,
            )
            dt = jax.nn.softplus(self.dt_proj(dt_part))
            dt = self._clamp_dt(dt)

            y_t, ssm_state = self._ssm_step(u_t, ssm_state, dt, b_t, c_t, a, d)
            y_t = y_t * jax.nn.silu(z_t)
            return (conv_state, ssm_state), y_t

        (conv_state, ssm_state), y = jax.lax.scan(
            _step,
            (conv_state, ssm_state),
            (x_in, z, reset_mask),
        )

        y = self.out_proj(y)
        y = self.dropout_layer(y, deterministic=not train)
        return x + y, (conv_state, ssm_state)

    def step(
        self,
        x_t: jnp.ndarray,
        state: Optional[Tuple[jnp.ndarray, jnp.ndarray]],
        train: bool,
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        b = x_t.shape[0]
        h = self.norm(x_t.astype(jnp.float32))
        xz = self.in_proj(h)
        x_in, z = jnp.split(xz, 2, axis=-1)

        if state is None:
            conv_state, ssm_state = self._init_state(b)
        else:
            conv_state, ssm_state = state

        conv_out, conv_state = self._conv_step(x_in, conv_state)
        u_t = jax.nn.silu(conv_out)

        x_proj = self.x_proj(u_t)
        dt_part, b_t, c_t = jnp.split(
            x_proj,
            [self.dt_rank, self.dt_rank + self.d_state],
            axis=-1,
        )
        dt = jax.nn.softplus(self.dt_proj(dt_part))
        dt = self._clamp_dt(dt)

        a = -jnp.exp(self.A_log)
        d = self.D
        y_t, ssm_state = self._ssm_step(u_t, ssm_state, dt, b_t, c_t, a, d)
        y_t = y_t * jax.nn.silu(z)

        y_t = self.out_proj(y_t)
        y_t = self.dropout_layer(y_t, deterministic=not train)
        return x_t + y_t, (conv_state, ssm_state)


class Mamba(nn.Module):
    model_dim: int
    n_layers: int
    d_inner: int
    d_state: int
    d_conv: int
    dt_rank: int = 1
    dt_min: float = 1e-4
    dt_max: float = 1.0
    dropout: float = 0.0

    def setup(self):
        self.layers = [
            MambaLayer(
                model_dim=self.model_dim,
                d_inner=self.d_inner,
                d_state=self.d_state,
                d_conv=self.d_conv,
                dt_rank=self.dt_rank,
                dt_min=self.dt_min,
                dt_max=self.dt_max,
                dropout=self.dropout,
            )
            for _ in range(self.n_layers)
        ]

    def __call__(
        self,
        x: jnp.ndarray,
        states: Optional[List[Tuple[jnp.ndarray, jnp.ndarray]]],
        reset_mask: Optional[jnp.ndarray],
        train: bool,
    ) -> Tuple[jnp.ndarray, List[Tuple[jnp.ndarray, jnp.ndarray]]]:
        h = x
        new_states = []
        for i, layer in enumerate(self.layers):
            state = states[i] if states is not None else None
            h, state = layer(h, state, reset_mask, train)
            new_states.append(state)
        return h, new_states

    def step(
        self,
        x_t: jnp.ndarray,
        states: Optional[List[Tuple[jnp.ndarray, jnp.ndarray]]],
        train: bool,
    ) -> Tuple[jnp.ndarray, List[Tuple[jnp.ndarray, jnp.ndarray]]]:
        h = x_t
        new_states = []
        for i, layer in enumerate(self.layers):
            state = states[i] if states is not None else None
            h, state = layer.step(h, state, train)
            new_states.append(state)
        return h, new_states
