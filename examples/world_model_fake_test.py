#!/usr/bin/env python3
import argparse
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from loco_mujoco.algorithms.common.world_model import WorldModel, world_model_loss


def _make_done_signal(t: int, b: int, min_len: int, max_len: int, key: jax.random.PRNGKey) -> jnp.ndarray:
    rng = jax.random.split(key, b)
    done = np.zeros((t, b), dtype=bool)
    for i in range(b):
        cur = 0
        while cur < t:
            ep_len = int(jax.random.randint(rng[i], (), min_len, max_len + 1))
            end = min(cur + ep_len - 1, t - 1)
            done[end, i] = True
            cur = end + 1
    return jnp.array(done)


def _make_fake_data(
    t: int,
    b: int,
    obs_dim: int,
    action_dim: int,
    key: jax.random.PRNGKey,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    key, k_phase, k_freq, k_noise, k_action, k_done = jax.random.split(key, 6)
    time = jnp.linspace(0.0, 4.0 * jnp.pi, t)
    phase = jax.random.uniform(k_phase, (b, 3), minval=0.0, maxval=2.0 * jnp.pi)
    freq = jax.random.uniform(k_freq, (b, 3), minval=0.5, maxval=1.5)
    base_vel = jnp.sin(time[:, None, None] * freq[None, :, :] + phase[None, :, :])
    base_vel = base_vel + 0.05 * jax.random.normal(k_noise, base_vel.shape)
    base_disp = jnp.cumsum(base_vel, axis=0) * 0.05
    yaw = 0.2 * jnp.sin(time)[:, None] + 0.1 * jax.random.normal(k_noise, (t, b))
    cy = jnp.cos(yaw)
    sy = jnp.sin(yaw)
    zeros = jnp.zeros_like(cy)
    ones = jnp.ones_like(cy)
    base_rot = jnp.stack(
        [
            jnp.stack([cy, -sy, zeros], axis=-1),
            jnp.stack([sy, cy, zeros], axis=-1),
            jnp.stack([zeros, zeros, ones], axis=-1),
        ],
        axis=-2,
    )

    obs = jnp.zeros((t, b, obs_dim), dtype=jnp.float32)
    if obs_dim >= 3:
        obs = obs.at[:, :, :3].set(base_disp)
    if obs_dim >= 6:
        obs = obs.at[:, :, 3:6].set(base_vel)
    if obs_dim > 6:
        extra = jax.random.normal(k_noise, (t, b, obs_dim - 6)) * 0.1
        obs = obs.at[:, :, 6:].set(extra)

    action = jax.random.normal(k_action, (t, b, action_dim)) * 0.2
    inputs = jnp.concatenate([obs, action], axis=-1)

    done = _make_done_signal(t, b, min_len=max(8, t // 8), max_len=max(12, t // 3), key=k_done)
    return inputs, base_disp, base_vel, base_rot, done


def _reset_mask(done: jnp.ndarray) -> jnp.ndarray:
    b = done.shape[1]
    return jnp.concatenate([jnp.ones((1, b), dtype=bool), done[:-1]], axis=0)


def _to_first_frame(
    target_disp: jnp.ndarray,
    target_rot: jnp.ndarray,
    target_vel: jnp.ndarray,
    done: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    reset_mask = _reset_mask(done)

    def _step(carry, inputs):
        ref_disp, ref_rot = carry
        disp_t, rot_t, vel_t, reset_t = inputs
        reset = reset_t[:, None]
        ref_disp = jnp.where(reset, disp_t, ref_disp)
        ref_rot = jnp.where(reset[:, None, None], rot_t, ref_rot)
        rel_world = disp_t - ref_disp
        ref_rot_t = jnp.transpose(ref_rot, (0, 2, 1))
        rel_local = jnp.einsum("bij,bj->bi", ref_rot_t, rel_world)
        vel_local = jnp.einsum("bij,bj->bi", ref_rot_t, vel_t)
        return (ref_disp, ref_rot), (rel_local, vel_local)

    init_disp = target_disp[0]
    init_rot = target_rot[0]
    (_, _), (disp_local, vel_local) = jax.lax.scan(
        _step,
        (init_disp, init_rot),
        (target_disp, target_rot, target_vel, reset_mask),
    )
    return disp_local, vel_local

def main() -> None:
    parser = argparse.ArgumentParser(description="World model fake-data test.")
    parser.add_argument("--seg-len", type=int, default=64)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--obs-dim", type=int, default=12)
    parser.add_argument("--action-dim", type=int, default=4)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--aux-vel-weight", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="wm_fake_test.png")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-jit", action="store_true")
    args = parser.parse_args()

    key = jax.random.PRNGKey(args.seed)
    inputs, target_disp, target_vel, target_rot, done = _make_fake_data(
        args.seg_len,
        args.batch,
        args.obs_dim,
        args.action_dim,
        key,
    )

    model = WorldModel(
        input_dim=args.obs_dim + args.action_dim,
        model_dim=args.model_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
    )

    key, init_key = jax.random.split(key)
    params = model.init(
        init_key,
        inputs,
        states=model.init_state(args.batch),
        reset_mask=_reset_mask(done),
        train=True,
    )["params"]

    tx = optax.adam(args.lr)
    opt_state = tx.init(params)

    def loss_fn(params, rng, inputs, target_disp, target_vel, target_rot, done, aux_vel_weight):
        loss, metrics, _ = world_model_loss(
            model,
            params,
            inputs,
            target_disp,
            target_vel,
            target_rot,
            done,
            None,
            aux_vel_weight,
            rng,
        )
        return loss, metrics

    def train_step(params, opt_state, rng, inputs, target_disp, target_vel, target_rot, done, aux_vel_weight):
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params,
            rng,
            inputs,
            target_disp,
            target_vel,
            target_rot,
            done,
            aux_vel_weight,
        )
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, metrics

    if not args.no_jit:
        train_step = jax.jit(train_step, static_argnames=("aux_vel_weight",))

    losses = []
    metrics = None
    for _ in range(args.steps):
        key, step_key = jax.random.split(key)
        params, opt_state, loss, metrics = train_step(
            params,
            opt_state,
            step_key,
            inputs,
            target_disp,
            target_vel,
            target_rot,
            done,
            args.aux_vel_weight,
        )
        losses.append(float(loss))

    key, eval_key = jax.random.split(key)
    (pred_disp, pred_vel), _ = model.apply(
        {"params": params},
        inputs,
        states=model.init_state(args.batch),
        reset_mask=_reset_mask(done),
        train=False,
        rngs={"dropout": eval_key},
    )

    target_disp_local, target_vel_local = _to_first_frame(
        target_disp,
        target_rot,
        target_vel,
        done,
    )
    disp_mse = jnp.mean((pred_disp - target_disp_local) ** 2)
    vel_mse = jnp.mean((pred_vel - target_vel_local) ** 2)
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Disp MSE: {float(disp_mse):.6f} | Vel MSE: {float(vel_mse):.6f}")
    if metrics is not None:
        print({k: float(v) for k, v in metrics.items()})

    import matplotlib.pyplot as plt

    t = np.arange(args.seg_len)
    b0 = 0
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    for i in range(3):
        axes[i, 0].plot(t, np.array(target_disp_local[:, b0, i]), label="target")
        axes[i, 0].plot(t, np.array(pred_disp[:, b0, i]), label="pred", alpha=0.8)
        axes[i, 0].set_ylabel(f"disp[{i}]")
        axes[i, 1].plot(t, np.array(target_vel_local[:, b0, i]), label="target")
        axes[i, 1].plot(t, np.array(pred_vel[:, b0, i]), label="pred", alpha=0.8)
        axes[i, 1].set_ylabel(f"vel[{i}]")
    axes[0, 0].legend()
    axes[0, 1].legend()
    axes[-1, 0].set_xlabel("t")
    axes[-1, 1].set_xlabel("t")
    fig.suptitle("World Model Fake Data: Target vs Pred")

    fig2, ax2 = plt.subplots(figsize=(8, 3))
    ax2.plot(losses)
    ax2.set_title("Training loss")
    ax2.set_xlabel("step")
    ax2.set_ylabel("loss")

    fig.tight_layout()
    fig2.tight_layout()

    fig.savefig(args.out, dpi=150)
    fig2.savefig(args.out.replace(".png", "_loss.png"), dpi=150)
    print(f"Saved plots to {args.out} and {args.out.replace('.png', '_loss.png')}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
