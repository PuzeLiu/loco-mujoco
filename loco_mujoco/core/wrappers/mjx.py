from copy import deepcopy
from functools import partial
import dataclasses
from typing import Optional, Tuple, Union, Any, Dict, TYPE_CHECKING

import numpy as np
import jax
import jax.numpy as jnp
from flax import struct

from loco_mujoco.core.mujoco_mjx import Mjx, MjxState
from loco_mujoco.core.utils.env import Box

if TYPE_CHECKING:
    from loco_mujoco.algorithms.common.world_model import WorldModelTXL


class LocoMjxWrapper:

    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        """
        Allow proxy access to regular attributes of Mjx env.
        """
        return getattr(self.env, name)

    def reset(self, rng_key, env_id=None):
        state = self.env.mjx_reset(rng_key, env_id=env_id)
        return state.observation, state

    def step(self, state, action):
        next_state = self.env.mjx_step(state, action)
        next_obs = jnp.where(next_state.done, next_state.additional_carry.final_observation, next_state.observation)
        return (next_obs, next_state.reward, next_state.absorbing, next_state.done,
                next_state.info, next_state)


@struct.dataclass
class BaseWrapperState:

    def __getattr__(self, name):
        """
        Allow proxy access to all attributes of all States.
        """
        try:
            if name in self.__dict__.keys():
                return self.__dict__[name]
            else:
                return getattr(self.env_state, name)
        except AttributeError as e:
            raise AttributeError(f"Attribute '{name}' not found in any env state nor the MjxState.") from e

    def find(self, cls):

        if isinstance(self, cls):
            return self
        elif isinstance(self.env_state, MjxState) and cls != MjxState:
            raise AttributeError(f"Class '{cls}' not found")
        else:
            return self.env_state.find(cls)


class BaseWrapper:

    def __init__(self, env):
        # if it's the bare Mjx class, wrap it in the LocoMjxWrapper first
        if issubclass(env.__class__, Mjx):
            self.env = LocoMjxWrapper(env)
        else:
            self.env = env

    def reset(self, rng_key):
        return self.env.reset(rng_key)

    def step(self, state, action):
        return self.env.step(state, action)

    def __getattr__(self, name):

        return getattr(self.env, name)

    def find_attr(self, state, attr_name):
        # Recursively search for the attribute
        if hasattr(state, attr_name):
            return getattr(state, attr_name)

        # If the attribute is not found, check env_state recursively
        if hasattr(state, 'env_state') and state.env_state is not None:
            return self.find_attr(state.env_state, attr_name)

        # If the attribute or env_state isn't found
        raise AttributeError(f"Attribute '{attr_name}' not found")

    def unwrapped(self):
        # find first env which is not a subclass of BaseWrapper
        if isinstance(self.env, BaseWrapper):
            return self.env.unwrapped()
        else:
            return self.env.env

@struct.dataclass
class SummaryMetrics:
    mean_episode_return: float = 0.0
    mean_episode_length: float = 0.0
    max_timestep: int = 0.0

@struct.dataclass
class SummaryRichMetrics:
    mean_episode_return: float = 0.0
    mean_episode_length: float = 0.0
    max_timestep: int = 0.0
    frac_absorbed: float = 0.0
    juggle_absorbed: float = 0.0
    loco_absorbed: float = 0.0
    mean_episode_return_components: dict = dataclasses.field(default_factory=lambda: {})
    curriculum_step: int = 0

@struct.dataclass
class Metrics:
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int
    done: bool

@struct.dataclass
class RichMetrics:
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int
    done: bool
    absorbed: bool
    juggle_absorbed: bool
    loco_absorbed: bool
    episode_return_components: dict[str, float]
    returned_episode_return_components: dict[str, float]
    curriculum_step: int


@struct.dataclass
class LogEnvState(BaseWrapperState):
    env_state: MjxState
    metrics: Metrics

@struct.dataclass
class RichLogEnvState(BaseWrapperState):
    env_state: MjxState
    metrics: RichMetrics


class LogWrapper(BaseWrapper):
    """Log the episode returns and lengths."""

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng_key, env_id=None):
        obs, env_state = self.env.reset(rng_key, env_id=env_id)
        state = LogEnvState(env_state, metrics=Metrics(0, 0, 0, 0, 0, False))
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: LogEnvState, action: Union[int, float]):

        # make a step
        next_observation, reward, absorbing, done, info, env_state = self.env.step(state.env_state, action)

        new_episode_return = state.metrics.episode_returns + reward
        new_episode_length = state.metrics.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            metrics=Metrics(
                episode_returns=new_episode_return * (1 - done),
                episode_lengths=new_episode_length * (1 - done),
                returned_episode_returns=state.metrics.returned_episode_returns * (1 - done)
                                         + new_episode_return * done,
                returned_episode_lengths=state.metrics.returned_episode_lengths * (1 - done)
                                         + new_episode_length * done,
                timestep=state.metrics.timestep + 1,
                done=done,),
        )
        return next_observation, reward, absorbing, done, info, state
    

class RichLogWrapper(BaseWrapper):
    """Log the episode returns and lengths."""

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng_key, env_id=None):
        obs, env_state = self.env.reset(rng_key, env_id=env_id)
        initial_reward_components = env_state.additional_carry.reward_state.reward_components
        zero_components = {key: 0.0 for key in initial_reward_components.keys()}

        state = RichLogEnvState(env_state, 
                            metrics=RichMetrics(0, 0, 0, 0, 0, False, False, 
                                                False, False, zero_components.copy(), zero_components.copy(), 0))
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: RichLogEnvState, action: Union[int, float]):

        # make a step
        next_observation, reward, absorbing, done, info, env_state = self.env.step(state.env_state, action)

        new_episode_return = state.metrics.episode_returns + reward
        new_episode_length = state.metrics.episode_lengths + 1

        reward_components = state.metrics.episode_return_components
        new_episode_return_components = dict()
        episode_return_components = dict()
        returned_episode_return_components = dict()
        current_keys = set(env_state.additional_carry.reward_state.reward_components.keys())

        if len(reward_components) == 0:
            for key in current_keys:
                new_episode_return_components[key] = env_state.additional_carry.reward_state.reward_components[key]
                episode_return_components[key] = new_episode_return_components[key] * (1 - done)
                returned_episode_return_components[key] = new_episode_return_components[key] * done
        else:
            all_keys = current_keys.union(set(state.metrics.episode_return_components.keys()))
            
            for key in all_keys:
                old_episode_value = state.metrics.episode_return_components.get(key, 0.0)
                old_returned_value = state.metrics.returned_episode_return_components.get(key, 0.0)

                current_reward = env_state.additional_carry.reward_state.reward_components.get(key, 0.0)
                
                new_episode_return_components[key] = old_episode_value + current_reward
                episode_return_components[key] = new_episode_return_components[key] * (1 - done)
                returned_episode_return_components[key] = old_returned_value * (1 - done) + \
                    new_episode_return_components[key] * done

        
        state = RichLogEnvState(
            env_state=env_state,
            metrics=RichMetrics(
                episode_returns=new_episode_return * (1 - done),
                episode_lengths=new_episode_length * (1 - done),
                returned_episode_returns=state.metrics.returned_episode_returns * (1 - done)
                                         + new_episode_return * done,
                returned_episode_lengths=state.metrics.returned_episode_lengths * (1 - done)
                                         + new_episode_length * done,
                timestep=state.metrics.timestep + 1,
                done=done,
                absorbed=absorbing,
                juggle_absorbed=env_state.additional_carry.terminal_state_handler_state.is_absorbing_dict["juggle"],
                loco_absorbed=env_state.additional_carry.terminal_state_handler_state.is_absorbing_dict["loco"],
                episode_return_components=episode_return_components,
                returned_episode_return_components=returned_episode_return_components,
                curriculum_step=env_state.additional_carry.curriculum.step,
                ),
        )
        return next_observation, reward, absorbing, done, info, state


@struct.dataclass
class NStepWrapperState(BaseWrapperState):
    env_state: MjxState
    observation_buffer: jnp.ndarray


class NStepWrapper(BaseWrapper):

    def __init__(self, env, n_steps):
        super().__init__(env)
        self.n_steps = n_steps
        self.info = self.update_info(env.info)

    def update_info(self, info):
        new_info = deepcopy(info)
        high = np.tile(info.observation_space.high, self.n_steps)
        low = np.tile(info.observation_space.low, self.n_steps)
        observation_space = Box(low, high)
        new_info.observation_space = observation_space
        return new_info

    def reset(self, rng_key, env_id=None):
        obs, env_state = self.env.reset(rng_key, env_id=env_id)
        observation_buffer = jnp.tile(jnp.zeros_like(obs), (self.n_steps, 1))
        observation_buffer = observation_buffer.at[-1].set(obs)
        state = NStepWrapperState(env_state, observation_buffer)
        obs = jnp.reshape(observation_buffer, (-1,))
        return obs, state

    def step(self, state: NStepWrapperState, action: Union[int, float]):

        # make a step
        next_observation, reward, absorbing, done, info, env_state = self.env.step(state.env_state, action)

        # add observation to the buffer
        observation_buffer = state.observation_buffer
        observation_buffer = jnp.roll(observation_buffer, shift=-1, axis=0)
        observation_buffer = observation_buffer.at[-1].set(next_observation)
        state = NStepWrapperState(env_state, observation_buffer)
        next_observation = jnp.reshape(observation_buffer, (-1,))

        return next_observation, reward, absorbing, done, info, state


@struct.dataclass
class WorldModelWrapperState(BaseWrapperState):
    env_state: MjxState
    wm_kv_cache: Any
    wm_params: Any
    wm_buffer_inputs: jnp.ndarray
    wm_buffer_disp: jnp.ndarray
    wm_buffer_vel: jnp.ndarray
    wm_buffer_done: jnp.ndarray
    wm_buffer_valid: jnp.ndarray
    wm_pred_disp_history: jnp.ndarray
    wm_disp_history: jnp.ndarray
    wm_disp_hist_idx: jnp.ndarray
    wm_disp_steps_since_done: jnp.ndarray


class WorldModelWrapper(BaseWrapper):

    def __init__(self,
                 env,
                 model: "WorldModelTXL",
                 wm_obs_ind: jnp.ndarray,
                 eval_mem_len: int,
                 buffer_length: int = 0,
                 buffer_dtype: Any = jnp.float32,
                 disp_window: int = 0):
        super().__init__(env)
        self.model = model
        self.wm_obs_ind = wm_obs_ind
        self.eval_mem_len = int(eval_mem_len)
        self.buffer_length = int(buffer_length)
        self.buffer_dtype = self._resolve_buffer_dtype(buffer_dtype)
        self.disp_window = int(disp_window or 0)

    def _resolve_buffer_dtype(self, buffer_dtype):
        if isinstance(buffer_dtype, str):
            name = buffer_dtype.lower()
            if name in ("float16", "fp16"):
                return jnp.float16
            if name in ("bfloat16", "bf16"):
                return jnp.bfloat16
            if name in ("float32", "fp32"):
                return jnp.float32
        return buffer_dtype

    def _build_inputs(self, obs, action):
        wm_obs = obs[..., self.wm_obs_ind]
        wm_inputs = jnp.concatenate([wm_obs, action], axis=-1)
        return wm_inputs

    def _reset_cache(self, cache, done):
        if cache is None:
            return None
        done_mask = done[:, None, None, None]
        new_cache = []
        for k_cache, v_cache, cache_len in cache:
            k_cache = jnp.where(done_mask, jnp.zeros_like(k_cache), k_cache)
            v_cache = jnp.where(done_mask, jnp.zeros_like(v_cache), v_cache)
            cache_len = jnp.where(done, jnp.zeros_like(cache_len), cache_len)
            new_cache.append((k_cache, v_cache, cache_len))
        return new_cache

    def reset(self, rng_key, env_id=None):
        obs, env_state = self.env.reset(rng_key, env_id=env_id)
        batch_size = obs.shape[0]
        model_dtype = getattr(self.model, "dtype", jnp.float32)
        if self.eval_mem_len > 0:
            head_dim = self.model.model_dim // self.model.n_heads
            wm_kv_cache = [
                (
                    jnp.zeros((batch_size, self.model.n_heads, self.eval_mem_len, head_dim), dtype=model_dtype),
                    jnp.zeros((batch_size, self.model.n_heads, self.eval_mem_len, head_dim), dtype=model_dtype),
                    jnp.zeros((batch_size,), dtype=jnp.int32),
                )
                for _ in range(self.model.n_layers)
            ]
        else:
            wm_kv_cache = None
        if self.buffer_length > 0:
            wm_buffer_inputs = jnp.zeros(
                (self.buffer_length, batch_size, self.model.input_dim),
                dtype=self.buffer_dtype,
            )
            wm_buffer_disp = jnp.zeros((self.buffer_length, batch_size, 3), dtype=self.buffer_dtype)
            wm_buffer_vel = jnp.zeros((self.buffer_length, batch_size, 3), dtype=self.buffer_dtype)
            wm_buffer_done = jnp.zeros((self.buffer_length, batch_size), dtype=bool)
            wm_buffer_valid = jnp.zeros((self.buffer_length, batch_size), dtype=self.buffer_dtype)
        else:
            wm_buffer_inputs = jnp.zeros((0, batch_size, self.model.input_dim), dtype=obs.dtype)
            wm_buffer_disp = jnp.zeros((0, batch_size, 3), dtype=obs.dtype)
            wm_buffer_vel = jnp.zeros((0, batch_size, 3), dtype=obs.dtype)
            wm_buffer_done = jnp.zeros((0, batch_size), dtype=bool)
            wm_buffer_valid = jnp.zeros((0, batch_size), dtype=obs.dtype)
        if self.disp_window > 0:
            wm_pred_disp_history = jnp.zeros((self.disp_window, batch_size, 3), dtype=obs.dtype)
            wm_disp_history = jnp.zeros((self.disp_window, batch_size, 3), dtype=obs.dtype)
            wm_disp_hist_idx = jnp.array(0, dtype=jnp.int32)
            wm_disp_steps_since_done = jnp.zeros((batch_size,), dtype=jnp.int32)
        else:
            wm_pred_disp_history = jnp.zeros((0, batch_size, 3), dtype=obs.dtype)
            wm_disp_history = jnp.zeros((0, batch_size, 3), dtype=obs.dtype)
            wm_disp_hist_idx = jnp.array(0, dtype=jnp.int32)
            wm_disp_steps_since_done = jnp.zeros((batch_size,), dtype=jnp.int32)
        state = WorldModelWrapperState(
            env_state=env_state,
            wm_kv_cache=wm_kv_cache,
            wm_params=None,
            wm_buffer_inputs=wm_buffer_inputs,
            wm_buffer_disp=wm_buffer_disp,
            wm_buffer_vel=wm_buffer_vel,
            wm_buffer_done=wm_buffer_done,
            wm_buffer_valid=wm_buffer_valid,
            wm_pred_disp_history=wm_pred_disp_history,
            wm_disp_history=wm_disp_history,
            wm_disp_hist_idx=wm_disp_hist_idx,
            wm_disp_steps_since_done=wm_disp_steps_since_done,
        )
        return obs, state

    def step(self, state: WorldModelWrapperState, action: Union[int, float]):
        obs = state.observation
        wm_kv_cache = state.wm_kv_cache
        wm_buffer_inputs = state.wm_buffer_inputs
        wm_buffer_disp = state.wm_buffer_disp
        wm_buffer_vel = state.wm_buffer_vel
        wm_buffer_done = state.wm_buffer_done
        wm_buffer_valid = state.wm_buffer_valid
        wm_pred_disp_history = state.wm_pred_disp_history
        wm_disp_history = state.wm_disp_history
        wm_disp_hist_idx = state.wm_disp_hist_idx
        wm_disp_steps_since_done = state.wm_disp_steps_since_done

        batch_size = obs.shape[0]
        pred_disp = jnp.zeros((batch_size, 3), dtype=obs.dtype)
        pred_vel = jnp.zeros((batch_size, 3), dtype=obs.dtype)

        if state.wm_params is not None:
            wm_inputs = self._build_inputs(obs, action)
            model_dtype = getattr(self.model, "dtype", wm_inputs.dtype)
            wm_inputs = wm_inputs.astype(model_dtype)
            if self.eval_mem_len > 0:
                (pred_disp, pred_vel), wm_kv_cache = self.model.apply(
                    {"params": state.wm_params},
                    wm_inputs,
                    cache=wm_kv_cache,
                    train=False,
                    mem_len=self.eval_mem_len,
                    method=self.model.__class__.step,
                )
            else:
                (pred_disp, pred_vel), _ = self.model.apply(
                    {"params": state.wm_params},
                    wm_inputs,
                    cache=None,
                    train=False,
                    mem_len=0,
                    method=self.model.__class__.step,
                )
                wm_kv_cache = None
        pred_disp_abs = pred_disp
        if self.disp_window > 0:
            prev_pred_disp = wm_pred_disp_history[wm_disp_hist_idx]
            use_diff = wm_disp_steps_since_done >= self.disp_window
            pred_disp_abs = jnp.where(use_diff[:, None], prev_pred_disp + pred_disp, pred_disp)
            wm_pred_disp_history = wm_pred_disp_history.at[wm_disp_hist_idx].set(pred_disp_abs)

        def _set_pred_disp(env_state):
            fields = getattr(env_state, "__dataclass_fields__", {})
            if "additional_carry" in fields:
                prev_disp = env_state.additional_carry.pred_disp
                new_pred_disp = pred_disp_abs.astype(prev_disp.dtype)
                return env_state.replace(
                    additional_carry=env_state.additional_carry.replace(pred_disp=new_pred_disp)
                )
            if "env_state" in fields:
                return env_state.replace(env_state=_set_pred_disp(env_state.env_state))
            return env_state

        env_state = _set_pred_disp(state.env_state)
        # step env
        next_obs, reward, absorbing, done, info, env_state = self.env.step(env_state, action)

        base_disp = info["base_disp"]
        if self.disp_window > 0:
            prev_disp = wm_disp_history[wm_disp_hist_idx]
            use_diff = wm_disp_steps_since_done >= self.disp_window
            target_disp = jnp.where(use_diff[:, None], base_disp - prev_disp, base_disp)
            wm_disp_history = wm_disp_history.at[wm_disp_hist_idx].set(base_disp)
            wm_disp_hist_idx = (wm_disp_hist_idx + 1) % self.disp_window
            wm_disp_steps_since_done = jnp.where(done, 0, wm_disp_steps_since_done + 1)
        else:
            target_disp = base_disp
            pred_disp_abs = pred_disp
        wm_pred_disp_mse = jnp.mean((pred_disp - target_disp) ** 2, axis=-1)
        wm_pred_disp_abs_mse = jnp.mean((pred_disp_abs - base_disp) ** 2, axis=-1)
        wm_pred_vel_mse = jnp.mean((pred_vel - info["base_linvel"]) ** 2, axis=-1)

        wm_kv_cache = self._reset_cache(wm_kv_cache, done)

        info = dict(info)
        info["wm_pred_disp_mse"] = wm_pred_disp_mse
        info["wm_pred_disp_abs_mse"] = wm_pred_disp_abs_mse
        info["wm_pred_vel_mse"] = wm_pred_vel_mse

        if self.buffer_length > 0:
            wm_inputs = self._build_inputs(obs, action).astype(self.buffer_dtype)
            wm_buffer_inputs = jnp.roll(wm_buffer_inputs, shift=-1, axis=0)
            wm_buffer_disp = jnp.roll(wm_buffer_disp, shift=-1, axis=0)
            wm_buffer_vel = jnp.roll(wm_buffer_vel, shift=-1, axis=0)
            wm_buffer_done = jnp.roll(wm_buffer_done, shift=-1, axis=0)
            wm_buffer_valid = jnp.roll(wm_buffer_valid, shift=-1, axis=0)

            wm_buffer_inputs = wm_buffer_inputs.at[-1].set(wm_inputs)
            wm_buffer_disp = wm_buffer_disp.at[-1].set(info["base_disp"].astype(self.buffer_dtype))
            wm_buffer_vel = wm_buffer_vel.at[-1].set(info["base_linvel"].astype(self.buffer_dtype))
            wm_buffer_done = wm_buffer_done.at[-1].set(done)
            wm_buffer_valid = wm_buffer_valid.at[-1].set(jnp.ones_like(done, dtype=wm_buffer_valid.dtype))

        state = WorldModelWrapperState(
            env_state=env_state,
            wm_kv_cache=wm_kv_cache,
            wm_params=state.wm_params,
            wm_buffer_inputs=wm_buffer_inputs,
            wm_buffer_disp=wm_buffer_disp,
            wm_buffer_vel=wm_buffer_vel,
            wm_buffer_done=wm_buffer_done,
            wm_buffer_valid=wm_buffer_valid,
            wm_pred_disp_history=wm_pred_disp_history,
            wm_disp_history=wm_disp_history,
            wm_disp_hist_idx=wm_disp_hist_idx,
            wm_disp_steps_since_done=wm_disp_steps_since_done,
        )
        return next_obs, reward, absorbing, done, info, state


class VecEnv(BaseWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.reset = jax.vmap(self.env.reset, in_axes=(0, 0))
        self.step = jax.vmap(self.env.step, in_axes=(0, 0))


@struct.dataclass
class NormalizeVecRewEnvState(BaseWrapperState):
    env_state: MjxState
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    return_val: float


class NormalizeVecReward(BaseWrapper):

    def __init__(self, env, gamma):
        super().__init__(env)
        self.gamma = gamma

    def reset(self, key, env_id=None):
        obs, state = self.env.reset(key, env_id)
        batch_count = obs.shape[0]
        state = NormalizeVecRewEnvState(
            mean=0.0,
            var=1.0,
            count=1e-4,
            return_val=jnp.zeros((batch_count,)),
            env_state=state,
        )
        return obs, state

    def step(self, state, action):
        next_observation, reward, absorbing, done, info, env_state = self.env.step(state.env_state, action)

        return_val = (state.return_val * self.gamma * (1 - done) + reward)

        batch_mean = jnp.mean(return_val, axis=0)
        batch_var = jnp.var(return_val, axis=0)
        batch_count = next_observation.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecRewEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            return_val=return_val,
            env_state=env_state,
        )

        return next_observation, reward / jnp.sqrt(state.var + 1e-8), absorbing, done, info, state


def _parse_agent_reward_by_terms(rew_dict: Dict[str, jnp.ndarray],
                                 reward_terms: list) -> jnp.ndarray:
    """
    Sum reward components whose key contains any of the terms in reward_terms.
    Example:
      term "ball_pos" matches key "ball_pos_reward".

    Returns:
      (num_envs,) reward.
    """
    tot = None
    for k, v in rew_dict.items():
        # substring match to be robust to suffixes like "_reward" / "_cost"
        if any(term in k for term in reward_terms):
            tot = v if tot is None else (tot + v)

    if tot is None:
        # JIT-safe fallback
        example = next(iter(rew_dict.values()))
        tot = jnp.zeros_like(example)
    return tot


@struct.dataclass
class NormalizeVecRewEnvDictState(BaseWrapperState):
    env_state: MjxState
    mean: dict[str, jnp.ndarray]
    var: dict[str, jnp.ndarray]
    count: dict[str, float]
    return_val: dict[str, float]


class NormalizeVecRewardDict(BaseWrapper):

    def __init__(self, env, gamma):
        super().__init__(env)
        self.gamma = gamma
        self.env_cfg = env.env_cfg
        self.agent_names = self.env_cfg.agent.keys()
        stand_cfg = getattr(self.env_cfg, "stand_phase", None)
        self.stand_phase_enabled = False
        self.stand_phase_active_agents = set()
        if stand_cfg is not None and stand_cfg.get("enabled", False):
            self.stand_phase_enabled = True
            self.stand_phase_active_agents = set(stand_cfg.get("active_agents", []))

    def reset(self, key, env_id=None):
        obs, state = self.env.reset(key, env_id)
        batch_count = obs.shape[0]
        state = NormalizeVecRewEnvDictState(
            mean={name: 0.0 for name in self.agent_names},
            var={name: 1.0 for name in self.agent_names},
            count={name: 1e-4 for name in self.agent_names},
            return_val={name: jnp.zeros((batch_count,)) for name in self.agent_names},
            env_state=state,
        )
        return obs, state

    def step(self, state, action):
        next_observation, reward, absorbing, done, info, env_state = self.env.step(state.env_state, action)
        reward_components = env_state.additional_carry.reward_state.reward_components
        absorbing_dict = env_state.additional_carry.terminal_state_handler_state.is_absorbing_dict
        no_ball = info.get("no_ball", None)
        if no_ball is None:
            no_ball = jnp.zeros_like(done, dtype=bool)
        agent_rewards = {}
        new_mean_dict = {}
        new_var_dict = {}
        new_count_dict = {}
        new_return_val_dict = {}
        for name in self.agent_names:
            absorb = absorbing_dict[self.env_cfg.agent[name].absorbing_key]
            terms = list(self.env_cfg.agent[name].reward_terms)
            cur_reward = _parse_agent_reward_by_terms(reward_components, terms)

            def _unmasked_stats(_):
                return_val = (state.return_val[name] * self.gamma * (1 - done) + cur_reward)
                batch_mean = jnp.mean(return_val, axis=0)
                batch_var = jnp.var(return_val, axis=0)
                # only count non-absorbing steps
                batch_count = jnp.sum(~absorb)

                delta = batch_mean - state.mean[name]
                tot_count = state.count[name] + batch_count

                new_mean = state.mean[name] + delta * batch_count / tot_count
                m_a = state.var[name] * state.count[name]
                m_b = batch_var * batch_count
                M2 = m_a + m_b + jnp.square(delta) * state.count[name] * batch_count / tot_count
                new_var = M2 / tot_count
                new_count = tot_count
                if self.env_cfg.agent[name].normalize_reward:
                    agent_reward = cur_reward / jnp.sqrt(new_var + 1e-8)
                else:
                    agent_reward = cur_reward
                return new_mean, new_var, new_count, return_val, agent_reward

            def _masked_stats(_):
                mask = jnp.logical_not(no_ball)
                return_val = (state.return_val[name] * self.gamma * (1 - done) + cur_reward)
                return_val = jnp.where(mask, return_val, state.return_val[name])
                valid = jnp.logical_and(mask, ~absorb)
                batch_count = jnp.sum(valid)

                def _update(_):
                    batch_mean = jnp.sum(jnp.where(valid, return_val, 0.0)) / batch_count
                    batch_var = jnp.sum(jnp.where(valid, (return_val - batch_mean) ** 2, 0.0)) / batch_count

                    delta = batch_mean - state.mean[name]
                    tot_count = state.count[name] + batch_count

                    new_mean = state.mean[name] + delta * batch_count / tot_count
                    m_a = state.var[name] * state.count[name]
                    m_b = batch_var * batch_count
                    M2 = m_a + m_b + jnp.square(delta) * state.count[name] * batch_count / tot_count
                    new_var = M2 / tot_count
                    new_count = tot_count
                    if self.env_cfg.agent[name].normalize_reward:
                        agent_reward = cur_reward / jnp.sqrt(new_var + 1e-8)
                    else:
                        agent_reward = cur_reward
                    agent_reward = jnp.where(mask, agent_reward, jnp.zeros_like(agent_reward))
                    return new_mean, new_var, new_count, return_val, agent_reward

                def _keep(_):
                    return state.mean[name], state.var[name], state.count[name], state.return_val[name], jnp.zeros_like(cur_reward)

                return jax.lax.cond(batch_count > 0, _update, _keep, operand=None)

            if self.stand_phase_enabled and name not in self.stand_phase_active_agents:
                masking_active = jnp.any(no_ball)
                new_mean, new_var, new_count, return_val, agent_reward = jax.lax.cond(
                    masking_active,
                    _masked_stats,
                    _unmasked_stats,
                    operand=None,
                )
            else:
                new_mean, new_var, new_count, return_val, agent_reward = _unmasked_stats(None)

            agent_rewards[name] = agent_reward
            new_mean_dict[name] = new_mean
            new_var_dict[name] = new_var
            new_count_dict[name] = new_count
            new_return_val_dict[name] = return_val

        state = NormalizeVecRewEnvDictState(
            mean=new_mean_dict,
            var=new_var_dict,
            count=new_count_dict,
            return_val=new_return_val_dict,
            env_state=env_state,
        )

        info['agent_rewards'] = agent_rewards

        return next_observation, reward, absorbing, done, info, state
