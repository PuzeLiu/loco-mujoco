from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn

from loco_mujoco.algorithms.common.transformer_xl import TransformerXL, init_mems, build_attn_mask


def _pad_to_multiple(x: jnp.ndarray, multiple: int, pad_value: float = 0.0) -> Tuple[jnp.ndarray, int]:
    """Pad the time dimension to a multiple of `multiple`."""
    t = x.shape[0]
    pad_len = (multiple - (t % multiple)) % multiple
    if pad_len == 0:
        return x, 0
    pad_shape = (pad_len,) + x.shape[1:]
    pad = jnp.full(pad_shape, pad_value, dtype=x.dtype)
    return jnp.concatenate([x, pad], axis=0), pad_len


def _segment(x: jnp.ndarray, seg_len: int) -> jnp.ndarray:
    """Reshape [T, ...] -> [N, seg_len, ...], assuming T is multiple of seg_len."""
    t = x.shape[0]
    n_seg = t // seg_len
    return x.reshape((n_seg, seg_len) + x.shape[1:])


def _compute_episode_ids(done: jnp.ndarray) -> jnp.ndarray:
    """
    Compute per-timestep episode ids from done flags.

    done[t] is True if the transition at time t ended an episode.
    We treat time t+1 as a new episode.
    """
    b = done.shape[1]
    start = jnp.concatenate([jnp.ones((1, b), dtype=bool), done[:-1]], axis=0)
    return jnp.cumsum(start, axis=0).astype(jnp.int32)


def _update_mem_episode_ids(mem_episode_ids: jnp.ndarray,
                            episode_ids_seg: jnp.ndarray,
                            mem_len: int) -> jnp.ndarray:
    if mem_len <= 0:
        return mem_episode_ids
    cat = jnp.concatenate([mem_episode_ids, episode_ids_seg], axis=0)
    return cat[-mem_len:]


class WorldModelTXL(nn.Module):
    """
    Transformer-XL world model that predicts base displacement (and velocity).

    Inputs are sequences of state-action features. Outputs are per-timestep predictions.
    """

    input_dim: int
    model_dim: int
    n_layers: int
    n_heads: int
    ff_dim: int
    dropout: float = 0.0
    mem_len: int = 0
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    def setup(self):
        self.in_proj = nn.Dense(self.model_dim, dtype=self.dtype, param_dtype=self.param_dtype)
        self.in_ln = nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype)
        self.txl = TransformerXL(
            model_dim=self.model_dim,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            ff_dim=self.ff_dim,
            dropout=self.dropout,
            mem_len=self.mem_len,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.out_disp = nn.Dense(3, dtype=self.dtype, param_dtype=self.param_dtype)
        self.out_vel = nn.Dense(3, dtype=self.dtype, param_dtype=self.param_dtype)
        self.dropout_layer = nn.Dropout(rate=self.dropout)

    def __call__(self,
                 x: jnp.ndarray,
                 mems: Optional[List[jnp.ndarray]],
                 attn_mask: Optional[jnp.ndarray],
                 train: bool) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], List[jnp.ndarray]]:
        """
        Args:
            x: (T, B, input_dim)
            mems: list of (mem_len, B, model_dim) per layer
            attn_mask: (B, T, mem_len+T) bool mask
        """
        x = x.astype(self.dtype)
        h = self.in_proj(x)
        h = self.in_ln(h)
        h = self.dropout_layer(h, deterministic=not train)
        h, new_mems = self.txl(h, mems, attn_mask, train)
        pred_disp = self.out_disp(h)
        pred_vel = self.out_vel(h)
        return (pred_disp, pred_vel), new_mems

    def step(self,
             x_t: jnp.ndarray,
             cache: Optional[List[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]],
             train: bool = False,
             mem_len: Optional[int] = None) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], List[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]]:
        """
        Single-step prediction with KV-cache for fast inference.

        Args:
            x_t: (B, input_dim)
            cache: list of (k_cache, v_cache, cache_len) per layer
            mem_len: optional override for cache length
        """
        x_t = x_t.astype(self.dtype)
        h = self.in_proj(x_t)
        h = self.in_ln(h)
        h, new_cache = self.txl.step(h, cache, train, mem_len=mem_len)
        pred_disp = self.out_disp(h)
        pred_vel = self.out_vel(h)
        return (pred_disp, pred_vel), new_cache


def world_model_loss(model: WorldModelTXL,
                     params: Any,
                     inputs: jnp.ndarray,
                     target_disp: jnp.ndarray,
                     target_vel: jnp.ndarray,
                     done: jnp.ndarray,
                     valid_mask: Optional[jnp.ndarray],
                     seg_len: int,
                     mem_len: int,
                     disp_window: int,
                     aux_vel_weight: float,
                     rng: jax.random.PRNGKey) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Compute loss for the Transformer-XL world model.

    The sequence is optionally segmented with Transformer-XL recurrence.
    rng is used for dropout when train=True.
    valid_mask can be used to ignore padded/invalid timesteps (shape [T, B]).
    disp_window computes per-episode trailing displacement targets inside the loss.
    """
    t, b = inputs.shape[0], inputs.shape[1]

    # episode ids for masking attention across episode boundaries
    episode_ids = _compute_episode_ids(done)

    inputs, pad_len = _pad_to_multiple(inputs, seg_len, pad_value=0.0)
    target_disp, _ = _pad_to_multiple(target_disp, seg_len, pad_value=0.0)
    target_vel, _ = _pad_to_multiple(target_vel, seg_len, pad_value=0.0)
    episode_ids, _ = _pad_to_multiple(episode_ids, seg_len, pad_value=episode_ids[-1, 0])

    if disp_window > 0:
        t_pad = target_disp.shape[0]
        if disp_window < t_pad:
            pad_disp = jnp.zeros((disp_window,) + target_disp.shape[1:], dtype=target_disp.dtype)
            pad_ep = jnp.full((disp_window,) + episode_ids.shape[1:], -1, dtype=episode_ids.dtype)
            prev_disp = jnp.concatenate([pad_disp, target_disp[:-disp_window]], axis=0)[-t_pad:]
            prev_ep = jnp.concatenate([pad_ep, episode_ids[:-disp_window]], axis=0)[-t_pad:]
            same_ep = episode_ids == prev_ep
            target_disp = jnp.where(same_ep[..., None], target_disp - prev_disp, target_disp)

    if valid_mask is None:
        valid = jnp.ones((t, b), dtype=inputs.dtype)
    else:
        valid = valid_mask.astype(inputs.dtype)
    if pad_len > 0:
        valid = jnp.concatenate([valid, jnp.zeros((pad_len, b), dtype=inputs.dtype)], axis=0)
    seg_inputs = _segment(inputs, seg_len)
    seg_disp = _segment(target_disp, seg_len)
    seg_vel = _segment(target_vel, seg_len)
    seg_ep = _segment(episode_ids, seg_len)
    seg_valid = _segment(valid, seg_len)

    model_dtype = getattr(model, "dtype", jnp.float32)
    mems = init_mems(model.n_layers, mem_len, b, model.model_dim, dtype=model_dtype)
    mem_episode_ids = jnp.zeros((mem_len, b), dtype=jnp.int32) if mem_len > 0 else jnp.zeros((0, b), dtype=jnp.int32)

    def _segment_step(carry, seg):
        mems, mem_ep, rng = carry
        rng, subkey = jax.random.split(rng)
        x_seg, y_disp_seg, y_vel_seg, ep_seg, valid_seg = seg
        x_seg = x_seg.astype(model_dtype)

        attn_mask = build_attn_mask(seg_len, mem_len, ep_seg, mem_ep)
        (pred_disp, pred_vel), new_mems = model.apply(
            {"params": params},
            x_seg,
            mems=mems,
            attn_mask=attn_mask,
            train=True,
            rngs={"dropout": subkey},
        )

        pred_disp_f = pred_disp.astype(jnp.float32)
        pred_vel_f = pred_vel.astype(jnp.float32)
        y_disp_f = y_disp_seg.astype(jnp.float32)
        y_vel_f = y_vel_seg.astype(jnp.float32)
        valid_f = valid_seg.astype(jnp.float32)

        disp_err = jnp.mean((pred_disp_f - y_disp_f) ** 2, axis=-1)
        vel_err = jnp.mean((pred_vel_f - y_vel_f) ** 2, axis=-1)

        disp_loss = jnp.sum(disp_err * valid_f)
        vel_loss = jnp.sum(vel_err * valid_f)
        count = jnp.sum(valid_f)

        new_mem_ep = _update_mem_episode_ids(mem_ep, ep_seg, mem_len)
        return (new_mems, new_mem_ep, rng), (disp_loss, vel_loss, count)

    (_, _, _), (disp_loss, vel_loss, count) = jax.lax.scan(
        _segment_step,
        (mems, mem_episode_ids, rng),
        (seg_inputs, seg_disp, seg_vel, seg_ep, seg_valid),
    )

    disp_loss = jnp.sum(disp_loss) / (jnp.sum(count) + 1e-8)
    vel_loss = jnp.sum(vel_loss) / (jnp.sum(count) + 1e-8)
    total_loss = disp_loss + aux_vel_weight * vel_loss

    metrics = {
        "wm_disp_mse": disp_loss,
        "wm_vel_mse": vel_loss,
        "wm_loss": total_loss,
        "wm_disp_rmse": jnp.sqrt(disp_loss + 1e-8),
        "wm_vel_rmse": jnp.sqrt(vel_loss + 1e-8),
    }
    return total_loss, metrics
