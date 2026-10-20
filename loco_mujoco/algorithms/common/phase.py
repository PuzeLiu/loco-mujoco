import math

import jax
import jax.numpy as jnp
from flax.core import FrozenDict

from loco_mujoco.algorithms.common.networks import PHASE_HEAD_MODULE_NAME


PHASE_VISIBILITY_HISTORY_MODES = ("latest", "any_history", "always")


def normalize_visibility_history_mode(mode: str) -> str:
    mode = str(mode)
    if mode not in PHASE_VISIBILITY_HISTORY_MODES:
        raise ValueError(
            "phase_prediction.visibility_history_mode must be one of "
            f"{PHASE_VISIBILITY_HISTORY_MODES}, got {mode!r}"
        )
    return mode


def ball_position_visible(
    obs_frames,
    position_indices,
    visibility_history_mode: str,
    epsilon: float = 1.0e-8,
):
    """Return whether phase prediction is valid in the chosen visibility mode."""
    mode = normalize_visibility_history_mode(visibility_history_mode)
    if mode == "always":
        return jnp.ones(obs_frames.shape[:-2], dtype=bool)
    if mode == "latest":
        ball_position = obs_frames[..., -1, position_indices]
        return jnp.any(jnp.abs(ball_position) > epsilon, axis=-1)

    ball_position_history = obs_frames[..., position_indices]
    return jnp.any(jnp.abs(ball_position_history) > epsilon, axis=(-2, -1))


def advance_phase_history(
    phase_history,
    phase_history_valid,
    phase_prediction,
    ball_visible,
):
    """Advance predicted phase state, aging invalid predictions out one frame at a time."""
    phase_history = jnp.roll(phase_history, shift=-1, axis=-1)
    phase_history = phase_history.at[..., -1].set(
        jnp.where(ball_visible, phase_prediction, 0.0)
    )
    phase_history_valid = jnp.roll(phase_history_valid, shift=-1, axis=-1)
    phase_history_valid = phase_history_valid.at[..., -1].set(ball_visible)
    return phase_history, phase_history_valid


def encode_supercycle_phase(phase, num_hands: int, num_balls: int):
    """Encode a normalized full-pattern phase into the legacy 8D time features."""
    super_beats = math.lcm(int(num_hands), int(num_balls))
    pattern_angle = 2.0 * jnp.pi * phase * (super_beats / float(num_balls))
    hand_angle = 2.0 * jnp.pi * phase * (super_beats / float(num_hands))
    return jnp.stack(
        [
            jnp.sin(pattern_angle),
            jnp.cos(pattern_angle),
            jnp.sin(hand_angle),
            jnp.cos(hand_angle),
            jnp.sin(2.0 * hand_angle),
            jnp.cos(2.0 * hand_angle),
            jnp.sin(pattern_angle),
            jnp.cos(pattern_angle),
        ],
        axis=-1,
    )


def phase_training_enabled(curriculum_stage, train_from_curriculum_stage):
    """Select rollout samples whose predicting head is allowed to learn."""
    return jnp.asarray(curriculum_stage) >= train_from_curriculum_stage


def phase_training_start_stage(phase_cfg):
    """Delay predictor training only in teacher-forcing mode."""
    if not phase_cfg.get('teacher_forcing', False):
        return 0
    stage = int(phase_cfg.get('train_from_curriculum_stage', 0))
    if stage < 0:
        raise ValueError('phase_prediction.train_from_curriculum_stage must be non-negative')
    if stage > 0 and phase_cfg.get('joint_training', False):
        raise ValueError('Phase stage gating requires phase_prediction.joint_training=false')
    return stage


def apply_gradients_with_phase_gate(train_state, grads, training_enabled):
    """Update PPO while freezing the isolated phase params and optimizer state.

    Zero phase gradients alone still advance Adam counters and can move weights
    through resumed momentum or AdamW decay. Restore the entire phase partition
    when the minibatch contains no eligible phase samples.
    """
    updated = train_state.apply_gradients(grads=grads)

    def freeze_phase(state):
        phase_params = train_state.params[PHASE_HEAD_MODULE_NAME]
        if isinstance(state.params, FrozenDict):
            params = state.params.copy({PHASE_HEAD_MODULE_NAME: phase_params})
        else:
            params = {**state.params, PHASE_HEAD_MODULE_NAME: phase_params}
        inner_states = {
            **state.opt_state.inner_states,
            'phase': train_state.opt_state.inner_states['phase'],
        }
        return state.replace(params=params,
                             opt_state=state.opt_state._replace(inner_states=inner_states))

    return jax.lax.cond(training_enabled, lambda state: state, freeze_phase, updated)
