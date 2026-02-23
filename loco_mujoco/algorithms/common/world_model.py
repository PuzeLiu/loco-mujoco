from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn

from loco_mujoco.algorithms.common.mamba import Mamba


def _compute_reset_mask(done: jnp.ndarray) -> jnp.ndarray:
    """Reset mask is True at the first step of each episode."""
    b = done.shape[1]
    return jnp.concatenate([jnp.ones((1, b), dtype=bool), done[:-1]], axis=0)


def init_target_norm_stats(dim: int = 6, dtype=jnp.float32) -> Dict[str, jnp.ndarray]:
    return {
        "mean": jnp.zeros((dim,), dtype=dtype),
        "var": jnp.ones((dim,), dtype=dtype),
        "count": jnp.asarray(1e-4, dtype=dtype),
    }


def target_norm_mean_std(target_norm_stats: Dict[str, jnp.ndarray],
                         eps: float = 1e-6,
                         dtype=jnp.float32) -> Tuple[jnp.ndarray, jnp.ndarray]:
    mean = target_norm_stats["mean"].astype(dtype)
    std = jnp.sqrt(target_norm_stats["var"].astype(dtype) + eps)
    return mean, std


def _update_running_stats_from_moments(mean: jnp.ndarray,
                                       var: jnp.ndarray,
                                       count: jnp.ndarray,
                                       batch_mean: jnp.ndarray,
                                       batch_var: jnp.ndarray,
                                       batch_count: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    delta = batch_mean - mean
    tot_count = count + batch_count
    new_mean = mean + delta * batch_count / (tot_count + 1e-8)
    m_a = var * count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + jnp.square(delta) * count * batch_count / (tot_count + 1e-8)
    new_var = m2 / (tot_count + 1e-8)
    return new_mean, new_var, tot_count


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
        self.in_proj = nn.Dense(self.model_dim, dtype=jnp.float32, param_dtype=jnp.float32)
        self.in_ln = nn.LayerNorm(dtype=jnp.float32, param_dtype=jnp.float32)
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
        )
        self.out_disp = nn.Dense(3, dtype=jnp.float32, param_dtype=jnp.float32)
        self.out_vel = nn.Dense(3, dtype=jnp.float32, param_dtype=jnp.float32)
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
        x = x.astype(jnp.float32)
        h = self.in_proj(x)
        h = self.in_ln(h)
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
        x_t = x_t.astype(jnp.float32)
        h = self.in_proj(x_t)
        h = self.in_ln(h)
        h, new_cache = self.mamba.step(h, cache, train)
        pred_disp = self.out_disp(h)
        pred_vel = self.out_vel(h)
        return (pred_disp, pred_vel), new_cache

    def init_state(self, batch_size: int) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
        d_inner, d_state, d_conv = self._mamba_dims()
        conv_len = max(d_conv - 1, 0)
        conv_state = jnp.zeros((batch_size, conv_len, d_inner), dtype=jnp.float32)
        ssm_state = jnp.zeros((batch_size, d_inner, d_state), dtype=jnp.float32)
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
                     *,
                     normalize_targets: bool = False,
                     target_norm_stats: Optional[Dict[str, jnp.ndarray]] = None,
                     target_norm_eps: float = 1e-6,
                     delta_consistency_weight: float = 0.1,
                     delta_consistency_lags: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),):
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
    y_disp_raw_f = target_disp.astype(jnp.float32)
    y_vel_raw_f = target_vel.astype(jnp.float32)
    valid_f = valid.astype(jnp.float32)

    if normalize_targets:
        if target_norm_stats is None:
            target_norm_stats = init_target_norm_stats(dim=6, dtype=jnp.float32)
        mean, std = target_norm_mean_std(target_norm_stats, eps=target_norm_eps, dtype=jnp.float32)
        var = target_norm_stats["var"].astype(jnp.float32)
        count_stats = target_norm_stats["count"].astype(jnp.float32)

        disp_mean = mean[:3].reshape((1, 1, 3))
        vel_mean = mean[3:].reshape((1, 1, 3))
        disp_std = std[:3].reshape((1, 1, 3))
        vel_std = std[3:].reshape((1, 1, 3))

        y_disp_f = (y_disp_raw_f - disp_mean) / disp_std
        y_vel_f = (y_vel_raw_f - vel_mean) / vel_std

        target_all = jnp.concatenate([y_disp_raw_f, y_vel_raw_f], axis=-1)  # (T,B,6)
        valid_exp = valid_f[..., None]
        batch_count = jnp.sum(valid_f)
        batch_mean = jnp.sum(target_all * valid_exp, axis=(0, 1)) / (batch_count + 1e-8)
        centered = target_all - batch_mean.reshape((1, 1, -1))
        batch_var = jnp.sum((centered ** 2) * valid_exp, axis=(0, 1)) / (batch_count + 1e-8)
        new_mean, new_var, new_count = _update_running_stats_from_moments(
            mean, var, count_stats, batch_mean, batch_var, batch_count
        )
        out_target_norm_stats = {
            "mean": new_mean,
            "var": new_var,
            "count": new_count,
        }
    else:
        y_disp_f = y_disp_raw_f
        y_vel_f = y_vel_raw_f
        new_mean, new_var = None, None
        out_target_norm_stats = target_norm_stats

    disp_err = jnp.mean((pred_disp_f - y_disp_f) ** 2, axis=-1)
    vel_err = jnp.mean((pred_vel_f - y_vel_f) ** 2, axis=-1)

    count = jnp.sum(valid_f)
    disp_loss = jnp.sum(disp_err * valid_f) / (count + 1e-8)
    vel_loss = jnp.sum(vel_err * valid_f) / (count + 1e-8)

    if normalize_targets:
        pred_disp_raw = pred_disp_f * disp_std + disp_mean
        pred_vel_raw = pred_vel_f * vel_std + vel_mean
    else:
        pred_disp_raw = pred_disp_f
        pred_vel_raw = pred_vel_f

    disp_err_raw = jnp.mean((pred_disp_raw - y_disp_raw_f) ** 2, axis=-1)
    vel_err_raw = jnp.mean((pred_vel_raw - y_vel_raw_f) ** 2, axis=-1)
    disp_loss_raw = jnp.sum(disp_err_raw * valid_f) / (count + 1e-8)
    vel_loss_raw = jnp.sum(vel_err_raw * valid_f) / (count + 1e-8)

    # =========================
    # NEW: delta-sum consistency loss (multi-lag)
    # Enforce (pred_disp[t] - pred_disp[t-k]) ~ (target_disp[t] - target_disp[t-k])
    # while ignoring pairs that cross resets and padded timesteps.
    # =========================
    def _no_reset_cross_mask(reset_mask_bool: jnp.ndarray, k: int) -> jnp.ndarray:
        """
        reset_mask_bool: (T,B) boolean; True indicates reset at that timestep.
        Returns mask (T,B) where True means the interval (t-k+1..t) has NO resets.
        For t < k, mask is False (invalid).
        """
        r = reset_mask_bool.astype(jnp.int32)  # (T,B)
        c = jnp.cumsum(r, axis=0)              # (T,B)
        # sum of resets in (t-k+1..t) = c[t] - c[t-k]
        c_shift = jnp.concatenate([jnp.zeros((k, b), dtype=c.dtype), c[:-k]], axis=0)
        window_sum = c - c_shift               # (T,B)
        ok = window_sum == 0
        # first k steps don't have a full window
        ok = ok.at[:k].set(False)
        return ok

    reset_bool = reset_mask.astype(bool)  # (T,B)

    delta_consistency_loss = jnp.array(0.0, dtype=jnp.float32)
    delta_terms = 0.0

    for k in delta_consistency_lags:
        # skip lags longer than segment
        if k <= 0 or k >= t:
            continue

        # (T,B,3) deltas, but only valid for t>=k
        pred_delta = pred_disp_f - jnp.concatenate([pred_disp_f[:k], pred_disp_f[:-k]], axis=0)
        targ_delta = y_disp_f    - jnp.concatenate([y_disp_f[:k],    y_disp_f[:-k]],    axis=0)

        # masks: valid at both endpoints + no reset crossing inside window
        ok_no_reset = _no_reset_cross_mask(reset_bool, k).astype(jnp.float32)  # (T,B)
        ok_valid = (valid_f *
                    jnp.concatenate([jnp.zeros((k, b), dtype=valid_f.dtype), valid_f[:-k]], axis=0))
        w = ok_valid * ok_no_reset  # (T,B)

        # per-step mse on delta (T,B)
        delta_err = jnp.mean((pred_delta - targ_delta) ** 2, axis=-1)

        denom = jnp.sum(w) + 1e-8
        delta_k_loss = jnp.sum(delta_err * w) / denom

        delta_consistency_loss = delta_consistency_loss + delta_k_loss
        delta_terms = delta_terms + 1.0

    # average across lags actually used
    delta_consistency_loss = delta_consistency_loss / (delta_terms + 1e-8)

    # =========================
    # Total loss
    # =========================
    total_loss = disp_loss + aux_vel_weight * vel_loss + delta_consistency_weight * delta_consistency_loss
    # total_loss = disp_loss + aux_vel_weight * vel_loss

    metrics = {
        "wm_disp_mse": disp_loss_raw,
        "wm_vel_mse": vel_loss_raw,
        "wm_loss": total_loss,
        "wm_disp_rmse": jnp.sqrt(disp_loss_raw + 1e-8),
        "wm_vel_rmse": jnp.sqrt(vel_loss_raw + 1e-8),
        "wm_delta_consistency_loss": delta_consistency_loss,
    }
    if normalize_targets:
        metrics["wm_disp_mse_norm"] = disp_loss
        metrics["wm_vel_mse_norm"] = vel_loss
        metrics["wm_disp_rmse_norm"] = jnp.sqrt(disp_loss + 1e-8)
        metrics["wm_vel_rmse_norm"] = jnp.sqrt(vel_loss + 1e-8)
        metrics["wm_target_mean"] = new_mean
        metrics["wm_target_std"] = jnp.sqrt(new_var + target_norm_eps)

    return total_loss, metrics, out_target_norm_stats
