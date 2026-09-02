import math

import jax.numpy as jnp


PHASE_VISIBILITY_HISTORY_MODES = ("latest", "any_history")


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
    """Return whether the selected ball position is visible in the chosen history scope."""
    mode = normalize_visibility_history_mode(visibility_history_mode)
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
