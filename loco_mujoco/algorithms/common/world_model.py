from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn

from loco_mujoco.algorithms.common.mamba import Mamba


def _compute_reset_mask(done: jnp.ndarray) -> jnp.ndarray:
    """Reset mask is True at the first step of each episode."""
    b = done.shape[1]
    return jnp.concatenate([jnp.ones((1, b), dtype=bool), done[:-1]], axis=0)


def _state_dtype(dtype: Any) -> Any:
    if dtype in (jnp.float16, jnp.bfloat16):
        return jnp.float32
    return dtype


def _huber_loss(x: jnp.ndarray, delta: float) -> jnp.ndarray:
    delta = jnp.asarray(delta, dtype=x.dtype)
    abs_x = jnp.abs(x)
    quadratic = jnp.minimum(abs_x, delta)
    linear = abs_x - quadratic
    return 0.5 * quadratic ** 2 + delta * linear


class WorldModel(nn.Module):
    """
    Mamba world model that predicts base displacement (and velocity).

    Inputs are sequences of state-action features. Outputs are per-timestep predictions.
    """

    input_dim: int
    model_dim: int
    n_layers: int
    n_heads: int
    ff_dim: int
    dropout: float = 0.0
    mamba_expand: Optional[int] = None
    mamba_d_state: Optional[int] = None
    mamba_d_conv: Optional[int] = None
    mamba_dt_rank: int = 1
    mamba_dt_min: float = 1e-4
    mamba_dt_max: float = 1.0
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    def _mamba_dims(self) -> Tuple[int, int, int]:
        if self.mamba_expand is None:
            expand = 1
            if self.ff_dim is not None and self.model_dim > 0:
                expand = max(1, min(2, self.ff_dim // self.model_dim))
        else:
            expand = max(1, int(self.mamba_expand))
        d_inner = self.model_dim * expand
        if self.mamba_d_state is None:
            d_state = max(8, int(self.n_heads))
        else:
            d_state = max(1, int(self.mamba_d_state))
        if self.mamba_d_conv is None:
            d_conv = 4
        else:
            d_conv = max(1, int(self.mamba_d_conv))
        return d_inner, d_state, d_conv

    def setup(self):
        self.in_proj = nn.Dense(self.model_dim, dtype=self.dtype, param_dtype=self.param_dtype)
        self.in_ln = nn.LayerNorm(dtype=jnp.float32, param_dtype=self.param_dtype)
        d_inner, d_state, d_conv = self._mamba_dims()
        self.mamba = Mamba(
            model_dim=self.model_dim,
            n_layers=self.n_layers,
            d_inner=d_inner,
            d_state=d_state,
            d_conv=d_conv,
            dt_rank=self.mamba_dt_rank,
            dt_min=self.mamba_dt_min,
            dt_max=self.mamba_dt_max,
            dropout=self.dropout,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.out_disp = nn.Dense(3, dtype=self.dtype, param_dtype=self.param_dtype)
        self.out_vel = nn.Dense(3, dtype=self.dtype, param_dtype=self.param_dtype)
        self.dropout_layer = nn.Dropout(rate=self.dropout)

    def __call__(self,
                 x: jnp.ndarray,
                 states: Optional[List[Tuple[jnp.ndarray, jnp.ndarray]]],
                 reset_mask: Optional[jnp.ndarray],
                 train: bool) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], List[Tuple[jnp.ndarray, jnp.ndarray]]]:
        """
        Args:
            x: (T, B, input_dim)
            states: list of (conv_state, ssm_state) per layer
            reset_mask: (T, B) bool mask for episode resets
        """
        x = x.astype(self.dtype)
        h = self.in_proj(x)
        h = self.in_ln(h.astype(jnp.float32)).astype(self.dtype)
        h = self.dropout_layer(h, deterministic=not train)
        h, new_states = self.mamba(h, states, reset_mask, train)
        pred_disp = self.out_disp(h)
        pred_vel = self.out_vel(h)
        return (pred_disp, pred_vel), new_states

    def step(self,
             x_t: jnp.ndarray,
             cache: Optional[List[Tuple[jnp.ndarray, jnp.ndarray]]],
             train: bool = False) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], List[Tuple[jnp.ndarray, jnp.ndarray]]]:
        """
        Single-step prediction with cached Mamba state for fast inference.

        Args:
            x_t: (B, input_dim)
            cache: list of (conv_state, ssm_state) per layer
        """
        x_t = x_t.astype(self.dtype)
        h = self.in_proj(x_t)
        h = self.in_ln(h.astype(jnp.float32)).astype(self.dtype)
        h, new_cache = self.mamba.step(h, cache, train)
        pred_disp = self.out_disp(h)
        pred_vel = self.out_vel(h)
        return (pred_disp, pred_vel), new_cache

    def init_state(self, batch_size: int) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
        d_inner, d_state, d_conv = self._mamba_dims()
        conv_len = max(d_conv - 1, 0)
        conv_state = jnp.zeros((batch_size, conv_len, d_inner), dtype=self.dtype)
        ssm_state = jnp.zeros((batch_size, d_inner, d_state), dtype=_state_dtype(self.dtype))
        return [(conv_state, ssm_state) for _ in range(self.n_layers)]

    def init_cache(self, batch_size: int) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
        return self.init_state(batch_size)


def world_model_loss(model: WorldModel,
                     params: Any,
                     inputs: jnp.ndarray,
                     target_disp: jnp.ndarray,
                     target_vel: jnp.ndarray,
                     target_rot: Optional[jnp.ndarray],
                     done: jnp.ndarray,
                     valid_mask: Optional[jnp.ndarray],
                     aux_vel_weight: float,
                     rng: jax.random.PRNGKey,
                     huber_delta: float = 1.0) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Compute loss for the Mamba world model.

    The input sequence is assumed to be a fixed-length segment.
    rng is used for dropout when train=True.
    valid_mask can be used to ignore padded/invalid timesteps (shape [T, B]).

    If target_rot is provided, target_disp and target_vel are assumed to be in world frame
    and will be converted to the frame of the first element of each episode within the segment.
    """
    t, b = inputs.shape[0], inputs.shape[1]

    reset_mask = _compute_reset_mask(done)
    reset_mask = jnp.asarray(reset_mask)
    if reset_mask.ndim == 1:
        reset_mask = reset_mask[:, None]
    elif reset_mask.ndim > 2:
        reset_mask = jnp.any(reset_mask, axis=tuple(range(2, reset_mask.ndim)))

    if target_rot is not None:
        def _step(carry, inputs):
            ref_disp, ref_rot = carry
            disp_t, rot_t, vel_t, reset_t = inputs

            reset_t = jnp.asarray(reset_t).astype(bool)
            if reset_t.ndim > 1:
                reset_t = jnp.any(reset_t, axis=tuple(range(1, reset_t.ndim)))  # -> (B,)

            # update references on resets
            ref_disp = jnp.where(reset_t[:, None], disp_t, ref_disp)                 # (B,3)
            ref_rot  = jnp.where(reset_t[:, None, None], rot_t, ref_rot)             # (B,3,3)

            rel_world = disp_t - ref_disp

            # transpose last two dims (safe + clearer)
            ref_rot_t = jnp.swapaxes(ref_rot, -1, -2)                                # (B,3,3)

            rel_local = jnp.einsum("bij,bj->bi", ref_rot_t, rel_world)
            vel_local = jnp.einsum("bij,bj->bi", ref_rot_t, vel_t)
            return (ref_disp, ref_rot), (rel_local, vel_local)

        init_disp = target_disp[0]
        init_rot = target_rot[0]
        (_, _), (target_disp, target_vel) = jax.lax.scan(
            _step,
            (init_disp, init_rot),
            (target_disp, target_rot, target_vel, reset_mask),
        )

    if valid_mask is None:
        valid = jnp.ones((t, b), dtype=inputs.dtype)
    else:
        valid = valid_mask.astype(inputs.dtype)
    (pred_disp, pred_vel), _ = model.apply(
        {"params": params},
        inputs,
        states=model.init_state(b),
        reset_mask=reset_mask,
        train=True,
        rngs={"dropout": rng},
    )

    pred_disp_f = pred_disp.astype(jnp.float32)
    pred_vel_f = pred_vel.astype(jnp.float32)
    y_disp_f = target_disp.astype(jnp.float32)
    y_vel_f = target_vel.astype(jnp.float32)
    valid_f = valid.astype(jnp.float32)

    disp_res = pred_disp_f - y_disp_f
    vel_res = pred_vel_f - y_vel_f
    disp_huber_err = jnp.mean(_huber_loss(disp_res, huber_delta), axis=-1)
    vel_huber_err = jnp.mean(_huber_loss(vel_res, huber_delta), axis=-1)
    disp_mse_err = jnp.mean(disp_res ** 2, axis=-1)
    vel_mse_err = jnp.mean(vel_res ** 2, axis=-1)

    disp_loss = jnp.sum(disp_huber_err * valid_f)
    vel_loss = jnp.sum(vel_huber_err * valid_f)
    count = jnp.sum(valid_f)

    disp_loss = disp_loss / (count + 1e-8)
    vel_loss = vel_loss / (count + 1e-8)
    total_loss = disp_loss + aux_vel_weight * vel_loss

    disp_mse = jnp.sum(disp_mse_err * valid_f) / (count + 1e-8)
    vel_mse = jnp.sum(vel_mse_err * valid_f) / (count + 1e-8)

    metrics = {
        "wm_disp_mse": disp_mse,
        "wm_vel_mse": vel_mse,
        "wm_disp_huber": disp_loss,
        "wm_vel_huber": vel_loss,
        "wm_loss": total_loss,
        "wm_disp_rmse": jnp.sqrt(disp_mse + 1e-8),
        "wm_vel_rmse": jnp.sqrt(vel_mse + 1e-8),
    }
    return total_loss, metrics


class WorldModelTXL(nn.Module):
    """
    Mamba world model that predicts base displacement (and velocity).

    Inputs are sequences of state-action features. Outputs are per-timestep predictions.
    """

    input_dim: int
    model_dim: int
    n_layers: int
    n_heads: int
    ff_dim: int
    dropout: float = 0.0
    mamba_expand: Optional[int] = None
    mamba_d_state: Optional[int] = None
    mamba_d_conv: Optional[int] = None
    mamba_dt_rank: int = 1
    mamba_dt_min: float = 1e-4
    mamba_dt_max: float = 1.0
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    def _mamba_dims(self) -> Tuple[int, int, int]:
        if self.mamba_expand is None:
            expand = 1
            if self.ff_dim is not None and self.model_dim > 0:
                expand = max(1, min(2, self.ff_dim // self.model_dim))
        else:
            expand = max(1, int(self.mamba_expand))
        d_inner = self.model_dim * expand
        if self.mamba_d_state is None:
            d_state = max(8, int(self.n_heads))
        else:
            d_state = max(1, int(self.mamba_d_state))
        if self.mamba_d_conv is None:
            d_conv = 4
        else:
            d_conv = max(1, int(self.mamba_d_conv))
        return d_inner, d_state, d_conv

    def setup(self):
        self.in_proj = nn.Dense(self.model_dim, dtype=self.dtype, param_dtype=self.param_dtype)
        self.in_ln = nn.LayerNorm(dtype=jnp.float32, param_dtype=self.param_dtype)
        d_inner, d_state, d_conv = self._mamba_dims()
        self.mamba = Mamba(
            model_dim=self.model_dim,
            n_layers=self.n_layers,
            d_inner=d_inner,
            d_state=d_state,
            d_conv=d_conv,
            dt_rank=self.mamba_dt_rank,
            dt_min=self.mamba_dt_min,
            dt_max=self.mamba_dt_max,
            dropout=self.dropout,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.out_disp = nn.Dense(3, dtype=self.dtype, param_dtype=self.param_dtype)
        self.out_vel = nn.Dense(3, dtype=self.dtype, param_dtype=self.param_dtype)
        self.dropout_layer = nn.Dropout(rate=self.dropout)

    def __call__(self,
                 x: jnp.ndarray,
                 states: Optional[List[Tuple[jnp.ndarray, jnp.ndarray]]],
                 reset_mask: Optional[jnp.ndarray],
                 train: bool) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], List[Tuple[jnp.ndarray, jnp.ndarray]]]:
        """
        Args:
            x: (T, B, input_dim)
            states: list of (conv_state, ssm_state) per layer
            reset_mask: (T, B) bool mask for episode resets
        """
        x = x.astype(self.dtype)
        h = self.in_proj(x)
        h = self.in_ln(h.astype(jnp.float32)).astype(self.dtype)
        h = self.dropout_layer(h, deterministic=not train)
        h, new_states = self.mamba(h, states, reset_mask, train)
        pred_disp = self.out_disp(h)
        pred_vel = self.out_vel(h)
        return (pred_disp, pred_vel), new_states

    def step(self,
             x_t: jnp.ndarray,
             cache: Optional[List[Tuple[jnp.ndarray, jnp.ndarray]]],
             train: bool = False) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], List[Tuple[jnp.ndarray, jnp.ndarray]]]:
        """
        Single-step prediction with cached Mamba state for fast inference.

        Args:
            x_t: (B, input_dim)
            cache: list of (conv_state, ssm_state) per layer
        """
        x_t = x_t.astype(self.dtype)
        h = self.in_proj(x_t)
        h = self.in_ln(h.astype(jnp.float32)).astype(self.dtype)
        h, new_cache = self.mamba.step(h, cache, train)
        pred_disp = self.out_disp(h)
        pred_vel = self.out_vel(h)
        return (pred_disp, pred_vel), new_cache

    def init_state(self, batch_size: int) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
        d_inner, d_state, d_conv = self._mamba_dims()
        conv_len = max(d_conv - 1, 0)
        conv_state = jnp.zeros((batch_size, conv_len, d_inner), dtype=self.dtype)
        ssm_state = jnp.zeros((batch_size, d_inner, d_state), dtype=_state_dtype(self.dtype))
        return [(conv_state, ssm_state) for _ in range(self.n_layers)]

    def init_cache(self, batch_size: int) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
        return self.init_state(batch_size)
