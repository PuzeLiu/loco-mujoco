import ast
import pickle
from pathlib import Path
from omegaconf import open_dict
import warnings
from dataclasses import dataclass
from typing import Any, Dict
from omegaconf import DictConfig, OmegaConf, ListConfig

import numpy as np
import jax
import jax.numpy as jnp
from flax import struct
import flax
import optax

from loco_mujoco.algorithms import (JaxRLAlgorithmBase, AgentConfBase, AgentStateBase, ActorCritic,
                                    Transition, IPPOTransition, TrainState, TrainStateBuffer, MetricHandlerTransition, AdaptiveLRState)
# from loco_mujoco.core.wrappers import LogWrapper, NStepWrapper, LogEnvState, VecEnv, NormalizeVecReward, SummaryMetrics
from loco_mujoco.core.wrappers import (RichLogWrapper, NStepWrapper, RichLogEnvState, VecEnv,
                                       NormalizeVecRewardDict, SummaryRichMetrics, WorldModelWrapper)
from loco_mujoco.core.wrappers.mjx import BaseWrapper
from loco_mujoco.utils import MetricsHandler, ValidationSummary
from loco_mujoco.algorithms.common.world_model import WorldModel, world_model_loss, init_target_norm_stats
from loco_mujoco.algorithms.world_model_agent import IPPOWorldConf, IPPOWorldState


def _scatter_actions(num_envs: int,
                     full_action_dim: int,
                     agent_actions: Dict[str, jnp.ndarray],
                     agent_action_idx: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """
    Build full env action vector from per-agent actions using scatter assignment.

    agent_actions[name]: (num_envs, agent_dim)
    agent_action_idx[name]: (agent_dim,) int indices into full action dim

    Returns:
      action: (num_envs, full_action_dim)
    """
    full = jnp.zeros((num_envs, full_action_dim), dtype=jnp.float32)
    for name, a in agent_actions.items():
        idx = agent_action_idx[name]
        full = full.at[:, idx].set(a)
    return full


@dataclass(frozen=True)
class IPPOAgentConf(AgentConfBase):
    config: DictConfig
    networks: Dict[str, Any]  # Dict[str, ActorCritic]
    txs: Dict[str, Any]       # Dict[str, optax tx]

    def serialize(self):
        """
        Serialize the agent configuration and network configuration.

        Returns:
            Serialized agent configuration as a dictionary.

        """
        conf_dict = OmegaConf.to_container(self.config, resolve=True, throw_on_missing=True)
        serialized_networks = {k: flax.serialization.to_state_dict(v) for k, v in self.networks.items()}
        return {"config": conf_dict, "networks": serialized_networks}

    @classmethod
    def from_dict(cls, d):
        config = OmegaConf.create(d["config"])
        # Rebuild networks from state dict; requires ActorCritic class + same ctor signature.
        # In practice you may already have a model registry; adjust accordingly.
        # Here we assume ActorCritic is a flax Module and can be restored with from_state_dict.
        networks = {k: flax.serialization.from_state_dict(ActorCritic, sd)
                    for k, sd in d["networks"].items()}
        txs = {k: IPPOJax._get_optimizer(config) for k in networks.keys()}
        return cls(config=config, networks=networks, txs=txs)


@struct.dataclass
class IPPOAgentState(AgentStateBase):
    train_states: Dict[str, Any]  # Dict[str, TrainState]

    def serialize(self):
        serialized = {k: flax.serialization.to_state_dict(v) for k, v in self.train_states.items()}
        return {"train_states": serialized}

    @classmethod
    def from_dict(cls, d, agent_conf: IPPOAgentConf):
        # Rebuild TrainState per agent.
        # NOTE: This assumes your TrainState can be built from serialized dict exactly like before.
        # If your TrainState.create is required, adapt accordingly.
        train_states = {}
        for agent_name, ts_dict in d["train_states"].items():
            train_states[agent_name] = TrainState(
                apply_fn=agent_conf.networks[agent_name].apply,
                tx=agent_conf.txs[agent_name],
                **ts_dict
            )
        return cls(train_states=train_states)


class IPPOJax(JaxRLAlgorithmBase):

    _agent_conf = IPPOAgentConf
    _agent_state = IPPOAgentState

    @classmethod
    def serialize(cls,
                  agent_conf: IPPOAgentConf,
                  agent_state: IPPOAgentState,
                  world_conf: IPPOWorldConf | None = None,
                  world_state: IPPOWorldState | None = None):
        serialized = super().serialize(agent_conf, agent_state)
        if world_conf is not None:
            serialized["world_model_conf"] = world_conf.serialize()
        if world_state is not None:
            serialized["world_model_state"] = world_state.serialize()
        return serialized

    @classmethod
    def save_agent(cls,
                   path,
                   agent_conf: IPPOAgentConf,
                   agent_state: IPPOAgentState,
                   world_conf: IPPOWorldConf | None = None,
                   world_state: IPPOWorldState | None = None):
        """Save agent + optional world model to a single file."""
        path = Path(path)
        path = path / (cls.__name__ + "_saved")
        path = path.with_suffix(cls._saved_agent_suffix)
        serialized_state = cls.serialize(agent_conf, agent_state, world_conf, world_state)
        with open(path, 'wb') as file:
            pickle.dump(serialized_state, file)
        print(f"\nSaved agent to: {path}\n")
        return path

    @classmethod
    def from_dict(cls, d):
        """ Load conf and state of an agent from a dictionary. """
        agent_conf = cls._agent_conf.from_dict(d["agent_conf"])
        agent_state = cls._agent_state.from_dict(d["agent_state"], agent_conf)
        world_conf = None
        world_state = None
        if "world_model_conf" in d:
            world_conf = IPPOWorldConf.from_dict(d["world_model_conf"])
        if "world_model_state" in d:
            world_state = IPPOWorldState.from_dict(d["world_model_state"], world_conf)
        return agent_conf, agent_state, world_conf, world_state

    @classmethod
    def init_agent_conf(cls, env, config):

        with (open_dict(config.experiment)):
            config.experiment.num_updates = (
                    config.experiment.total_timesteps // config.experiment.num_steps // config.experiment.num_envs)
            config.experiment.minibatch_size = (
                    config.experiment.num_envs * config.experiment.num_steps // config.experiment.num_minibatches)
            config.experiment.validation_interval = config.experiment.num_updates // config.experiment.validation.num
            config.experiment.validation.num = int(
                config.experiment.num_updates // config.experiment.validation_interval)

        # INIT NETWORK
        hidden_layers = config.experiment.hidden_layers \
            if isinstance(config.experiment.hidden_layers, (list, ListConfig)) \
            else ast.literal_eval(config.experiment.hidden_layers)

        networks = {}
        txs = {}

        # config.agent is a DictConfig: {agent_name: {...}}
        for agent_name, agent_cfg in config.env.agent.items():
            if agent_cfg.get("obs_actor_group", None) is not None:
                actor_obs_ind = env.obs_container.get_obs_ind_by_group(agent_cfg.obs_actor_group)
            else:
                actor_obs_ind = jnp.arange(env.mdp_info.observation_space.shape[0])

            if agent_cfg.get("obs_critic_group", None) is not None:
                critic_obs_ind = env.obs_container.get_obs_ind_by_group(agent_cfg.obs_critic_group)
            else:
                critic_obs_ind = jnp.arange(env.mdp_info.observation_space.shape[0])

            if hasattr(config.experiment, "len_obs_history") and config.experiment.len_obs_history > 1:
                obs_len = env.info.observation_space.shape[0]
                actor_obs_ind = jnp.concatenate([actor_obs_ind + i * obs_len
                                                 for i in range(config.experiment.len_obs_history)])
                critic_obs_ind = jnp.concatenate([critic_obs_ind + i * obs_len
                                                  for i in range(config.experiment.len_obs_history)])

            action_dim = len(agent_cfg.action_idx)

            networks[agent_name] = ActorCritic(
                action_dim,
                activation=config.experiment.activation,
                init_std=config.experiment.init_std,
                learnable_std=config.experiment.learnable_std,
                hidden_layer_dims=hidden_layers,
                actor_obs_ind=actor_obs_ind,
                critic_obs_ind=critic_obs_ind,
                # random=agent_cfg.get("random_action", False)
            )

            txs[agent_name] = cls._get_optimizer(config)
        return cls._agent_conf(config=config, networks=networks, txs=txs)

    @classmethod
    def init_world_conf(cls, env, config) -> IPPOWorldConf | None:
        wm_cfg = config.experiment.get("world_model", None)
        if wm_cfg is None or not wm_cfg.get("enabled", False):
            return None

        assert wm_cfg.get("obs_group", None) is not None, "World model obs_group or obs_ind must be configured."
        obs_group = wm_cfg.get("obs_group", None)
        obs_ind = env.obs_container.get_obs_ind_by_group(obs_group)
        if obs_ind.size == 0:
            raise ValueError(f"World model obs_group '{obs_group}' produced empty indices.")
        obs_ind = jnp.array(obs_ind, dtype=jnp.int32)
        input_dim = int(obs_ind.shape[0])
        action_dim = int(env.info.action_space.shape[0])
        input_dim += action_dim
        with open_dict(config.experiment.world_model):
            config.experiment.world_model.obs_ind = obs_ind.tolist()
            config.experiment.world_model.input_dim = input_dim
            config.experiment.world_model.action_dim = action_dim

        model = WorldModel(
            input_dim=input_dim,
            model_dim=wm_cfg.model_dim,
            n_layers=wm_cfg.n_layers,
            n_heads=wm_cfg.n_heads,
            ff_dim=wm_cfg.ff_dim,
            dropout=wm_cfg.dropout,
            mamba_expand=wm_cfg.get("mamba_expand", None),
            mamba_d_state=wm_cfg.get("mamba_d_state", None),
            mamba_d_conv=wm_cfg.get("mamba_d_conv", None),
            mamba_dt_rank=int(wm_cfg.mamba_dt_rank),
            mamba_dt_min=float(wm_cfg.mamba_dt_min),
            mamba_dt_max=float(wm_cfg.mamba_dt_max),
        )
        tx = cls._get_world_model_optimizer(config)
        return IPPOWorldConf(
            config=config,
            model=model,
            tx=tx,
            obs_ind=obs_ind,
            obs_group=obs_group,
        )
    
    @classmethod
    def _get_optimizer(cls, config):
        if config.experiment.get("adaptive_lr", False) or config.experiment.get("anneal_lr", False):
            tx = optax.chain(
                optax.clip_by_global_norm(config.experiment.max_grad_norm),
                optax.inject_hyperparams(optax.adamw)(
                    learning_rate=config.experiment.lr, # Initial LR
                    weight_decay=config.experiment.weight_decay,
                    eps=1e-5
                ),
            )
        else:
             tx = optax.chain(
                optax.clip_by_global_norm(config.experiment.max_grad_norm),
                optax.adamw(config.experiment.lr, weight_decay=config.experiment.weight_decay, eps=1e-5),
            )

        return tx

    @classmethod
    def _get_world_model_optimizer(cls, config):
        wm_cfg = config.experiment.world_model
        return optax.chain(
            optax.clip_by_global_norm(config.experiment.max_grad_norm),
            optax.adamw(wm_cfg.lr, weight_decay=wm_cfg.weight_decay, eps=1e-5),
        )

    @classmethod
    def build_train_fn(cls, env, agent_conf: IPPOAgentConf, world_conf: IPPOWorldConf | None = None, mh: MetricsHandler = None, wandb_run=None):
        """Returns the main train function of IPPO with optional timestep offset."""
        config = agent_conf.config.experiment
        wrapped_env = env if isinstance(env, BaseWrapper) else cls._wrap_env(env, config, world_conf=world_conf)
        return (
            lambda rng_key, agent_state=None, world_state=None, env_state=None, timesteps=0: cls._train_fn(
                rng_key,
                wrapped_env,
                agent_conf,
                agent_state,
                world_conf=world_conf,
                world_state=world_state,
                mh=mh,
                wandb_run=wandb_run,
                env_state=env_state,
                timesteps=timesteps,
            )
        )
    
    @classmethod
    def _train_fn(cls, rng, env,
                  agent_conf: IPPOAgentConf,
                  agent_state: IPPOAgentState = None,
                  world_conf: IPPOWorldConf | None = None,
                  world_state: IPPOWorldState | None = None,
                  mh: MetricsHandler = None,
                  wandb_run=None,
                  env_state=None,
                  timesteps: int = 0,
                  ):
        # extract static agent info
        config = agent_conf.config.experiment

        if not isinstance(env, BaseWrapper):
            env = cls._wrap_env(env, config, world_conf=world_conf)

        agent_names = tuple(agent_conf.networks.keys())
        world_model_enabled = world_conf is not None
        wm_obs_ind = world_conf.obs_ind if world_model_enabled else None
        wm_cfg = config.world_model if world_model_enabled else None
        target_norm_enabled = bool(wm_cfg.get("target_norm", True)) if world_model_enabled else False

        stand_cfg = agent_conf.config.env.get("stand_phase", None)
        stand_phase_enabled = False
        stand_phase_active_agents = set()
        if stand_cfg is not None and stand_cfg.get("enabled", False):
            stand_phase_enabled = True
            stand_phase_active_agents = set(stand_cfg.get("active_agents", []))

        # === CHANGE: action indices per agent for scattering ===
        agent_action_idx = {
            name: jnp.array(agent_conf.config.env.agent[name].action_idx, dtype=jnp.int32)
            for name in agent_names
        }

        full_action_dim = env.info.action_space.shape[0]
        num_envs = config.num_envs

        global_timesteps = timesteps if env_state is None else jnp.zeros_like(timesteps)

        # extract current agent state
        world_model_state = world_state
        if agent_state is not None:
            train_states = agent_state.train_states
        else:
            train_states = None

        if train_states is None:
            train_states = {}
            for name in agent_names:
                rng, subkey = jax.random.split(rng)
                init_x = jnp.zeros(env.info.observation_space.shape)
                params = agent_conf.networks[name].init(subkey, init_x)

                adaptive_lr_state = None
                if config.get("adaptive_lr", False):
                    adaptive_lr_state = AdaptiveLRState(learning_rate=jnp.array(config.lr))

                train_states[name] = TrainState.create(
                    apply_fn=agent_conf.networks[name].apply,
                    params=params["params"],
                    run_stats=params["run_stats"],
                    adaptive_lr_state=adaptive_lr_state,
                    tx=agent_conf.txs[name],
                )

        if world_model_enabled and world_model_state is None:
            rng, subkey = jax.random.split(rng)
            dummy_x = jnp.zeros((1, 1, wm_cfg.input_dim), dtype=jnp.float32)
            params = world_conf.model.init(subkey, dummy_x, mems=None, attn_mask=None, train=False)
            wm_ts = TrainState.create(
                apply_fn=world_conf.model.apply,
                params=params["params"],
                run_stats={},
                adaptive_lr_state=None,
                tx=world_conf.tx,
            )
            world_model_state = IPPOWorldState(
                train_state=wm_ts,
                target_norm_stats=init_target_norm_stats(dim=6, dtype=jnp.float32),
            )

        # INIT ENV (only if no previous env_state is provided)
        if env_state is None:
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, config.num_envs)
            obsv, env_state = env.reset(reset_rng, env_id=jnp.arange(config.num_envs))
        else:
            obsv = env_state.observation

        train_state_buffer = TrainStateBuffer.create(next(iter(train_states.values())), config.validation.num)
        if world_model_enabled:
            wm_target_mean = world_model_state.target_norm_stats["mean"]
            wm_target_std = jnp.sqrt(world_model_state.target_norm_stats["var"] + 1e-6)
            env_state = env_state.replace(
                wm_params=world_model_state.train_state.params,
                wm_target_norm_mean=wm_target_mean,
                wm_target_norm_std=wm_target_std,
            )

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_states, world_model_state, env_state, last_obs, train_state_buffer, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                agent_actions = {}
                agent_values = {}
                agent_log_probs = {}

                # update run_stats per agent during action selection
                new_train_states = dict(train_states)

                # split rng for agents deterministically
                keys = jax.random.split(_rng, len(agent_names) + 1)
                _rng_next = keys[0]
                agent_keys = keys[1:]

                for name, k in zip(agent_names, agent_keys):
                    ts = new_train_states[name]
                    net = agent_conf.networks[name]

                    y, updates = net.apply(
                        {"params": ts.params, "run_stats": ts.run_stats},
                        last_obs,
                        mutable=["run_stats"],
                    )
                    pi, value = y

                    ts = ts.replace(run_stats=updates["run_stats"])
                    a = pi.sample(seed=k)
                    lp = pi.log_prob(a)

                    agent_actions[name] = a
                    agent_values[name] = value
                    agent_log_probs[name] = lp
                    new_train_states[name] = ts

                action = _scatter_actions(num_envs, full_action_dim, agent_actions, agent_action_idx)

                # STEP ENV
                obsv, reward, absorbing, done, info, env_state = env.step(env_state, action)
                agent_rewards = info['agent_rewards']
                # GET METRICS
                log_env_state = env_state.find(RichLogEnvState)
                logged_metrics = log_env_state.metrics
                transition = IPPOTransition(
                    done=done,
                    absorbing=absorbing,
                    absorbing_dict=env_state.additional_carry.terminal_state_handler_state.is_absorbing_dict,
                    action=action,
                    action_dict=agent_actions,
                    value=agent_values,
                    reward=agent_rewards,
                    log_prob=agent_log_probs,
                    obs=last_obs,
                    info=info,
                    traj_state=env_state.additional_carry.traj_state,
                    metrics=logged_metrics,
                )
                runner_state = (new_train_states, world_model_state, env_state, obsv, train_state_buffer, _rng_next)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config.num_steps
            )

            train_states, world_model_state, env_state, last_obs, train_state_buffer, rng = runner_state
            traj_metrics = traj_batch.metrics
            absorb_ratio = jnp.sum(
                jnp.where(traj_metrics.done, traj_metrics.absorbed, 0.0)
            ) / (jnp.sum(traj_metrics.done) + 1e-6)
            absorb_ratio = absorb_ratio * jnp.ones_like(env_state.additional_carry.curriculum.absorb_ratio)
            curriculum = env_state.additional_carry.curriculum
            curriculum = curriculum.replace(absorb_ratio=absorb_ratio)
            def _set_curriculum(state):
                fields = getattr(state, "__dataclass_fields__", {})
                if "additional_carry" in fields:
                    return state.replace(
                        additional_carry=state.additional_carry.replace(curriculum=curriculum)
                    )
                if "env_state" in fields:
                    return state.replace(env_state=_set_curriculum(state.env_state))
                return state

            env_state = _set_curriculum(env_state)
            runner_state = (train_states, world_model_state, env_state, last_obs, train_state_buffer, rng)

            world_model_metrics = None
            if world_model_enabled:
                # Train world model on pre-step observations + action to match info targets.
                rng, wm_rng = jax.random.split(rng)
                wm_inputs = env_state.wm_buffer_inputs
                wm_disp = env_state.wm_buffer_disp
                wm_linvel = env_state.wm_buffer_vel
                wm_rot = env_state.wm_buffer_rot
                wm_done = env_state.wm_buffer_done
                wm_valid = env_state.wm_buffer_valid
                wm_done = jnp.where(
                    wm_valid > 0,
                    wm_done,
                    jnp.ones_like(wm_done, dtype=bool),
                )

                wm_ts = world_model_state.train_state
                wm_target_norm_stats = world_model_state.target_norm_stats
                wm_epochs = int(wm_cfg.get("epochs", 1))
                wm_minibatch_size = int(wm_cfg.get("minibatch_size", wm_inputs.shape[1]))
                wm_batch_size = wm_inputs.shape[1] if wm_minibatch_size >= wm_inputs.shape[1] else wm_minibatch_size
                wm_num_minibatches = 1 if wm_batch_size == wm_inputs.shape[1] else wm_inputs.shape[1] // wm_batch_size
                seg_len = int(wm_cfg.seg_len)
                assert wm_inputs.shape[0] >= seg_len, "World model input sequence length must be greater than or equal to segment length."

                def _sample_segment(inputs, disp, vel, rot, done, valid, rng):
                    t = inputs.shape[0]
                    b = inputs.shape[1]
                    if valid is None:
                        valid = jnp.ones((t, b), dtype=inputs.dtype)
                    max_start = t - seg_len + 1
                    start = jax.random.randint(rng, (), 0, max_start)

                    def _slice(x):
                        start_idx = (start, 0) + (0,) * (x.ndim - 2)
                        return jax.lax.dynamic_slice(x, start_idx, (seg_len,) + x.shape[1:])

                    rot_slice = None if rot is None else _slice(rot)
                    return _slice(inputs), _slice(disp), _slice(vel), rot_slice, _slice(done), _slice(valid)

                def _wm_minibatch_update(carry, batch):
                    wm_ts, wm_target_norm_stats = carry
                    mb_idx, mb_rng = batch
                    batch_inputs = jnp.take(wm_inputs, mb_idx, axis=1)
                    batch_disp = jnp.take(wm_disp, mb_idx, axis=1)
                    batch_linvel = jnp.take(wm_linvel, mb_idx, axis=1)
                    batch_rot = None if wm_rot is None else jnp.take(wm_rot, mb_idx, axis=1)
                    batch_done = jnp.take(wm_done, mb_idx, axis=1)
                    batch_valid = None if wm_valid is None else jnp.take(wm_valid, mb_idx, axis=1)
                    sample_rng, loss_rng = jax.random.split(mb_rng)
                    batch_inputs, batch_disp, batch_linvel, batch_rot, batch_done, batch_valid = _sample_segment(
                        batch_inputs,
                        batch_disp,
                        batch_linvel,
                        batch_rot,
                        batch_done,
                        batch_valid,
                        sample_rng,
                    )

                    def _wm_loss_fn(params):
                        loss, wm_metrics, new_target_norm_stats = world_model_loss(
                            world_conf.model,
                            params,
                            batch_inputs,
                            batch_disp,
                            batch_linvel,
                            batch_rot,
                            batch_done,
                            batch_valid,
                            wm_cfg.aux_vel_weight,
                            loss_rng,
                            normalize_targets=target_norm_enabled,
                            target_norm_stats=wm_target_norm_stats,
                        )
                        return loss, (wm_metrics, new_target_norm_stats)

                    (_, (wm_metrics, new_target_norm_stats)), wm_grads = jax.value_and_grad(
                        _wm_loss_fn, has_aux=True
                    )(wm_ts.params)
                    wm_target_norm_stats = new_target_norm_stats
                    wm_ts = wm_ts.apply_gradients(grads=wm_grads)
                    return (wm_ts, wm_target_norm_stats), wm_metrics

                def _wm_epoch_update(carry, _):
                    wm_ts, wm_target_norm_stats, rng = carry
                    rng, perm_key, mb_key = jax.random.split(rng, 3)
                    perm = jax.random.permutation(perm_key, wm_inputs.shape[1])
                    perm = perm[: wm_num_minibatches * wm_batch_size]
                    perm = perm.reshape((wm_num_minibatches, wm_batch_size))
                    mb_keys = jax.random.split(mb_key, wm_num_minibatches)
                    (wm_ts, wm_target_norm_stats), mb_metrics = jax.lax.scan(
                        _wm_minibatch_update,
                        (wm_ts, wm_target_norm_stats),
                        (perm, mb_keys),
                    )
                    mb_metrics = jax.tree.map(lambda x: jnp.mean(x, axis=0), mb_metrics)
                    return (wm_ts, wm_target_norm_stats, rng), mb_metrics

                (wm_ts, wm_target_norm_stats, wm_rng), epoch_metrics = jax.lax.scan(
                    _wm_epoch_update,
                    (wm_ts, wm_target_norm_stats, wm_rng),
                    None,
                    wm_epochs,
                )
                world_model_metrics = jax.tree.map(lambda x: jnp.mean(x, axis=0), epoch_metrics)
                world_model_state = world_model_state.replace(
                    train_state=wm_ts,
                    target_norm_stats=wm_target_norm_stats,
                )
                env_state = env_state.replace(
                    wm_params=wm_ts.params,
                    wm_target_norm_mean=wm_target_norm_stats["mean"],
                    wm_target_norm_std=jnp.sqrt(wm_target_norm_stats["var"] + 1e-6),
                )

            # CALCULATE ADVANTAGE
            # bootstrap values per agent
            last_vals = {}
            for name in agent_names:
                ts = train_states[name]
                net = agent_conf.networks[name]
                y, _ = net.apply({"params": ts.params, "run_stats": ts.run_stats}, last_obs, mutable=["run_stats"])
                _, last_val = y
                last_vals[name] = last_val

            def _calculate_gae(traj_batch: IPPOTransition, last_vals: Dict[str, jnp.ndarray]):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done = transition.done

                    new_gae = {}
                    adv_out = {}

                    for name in agent_names:
                        value = transition.value[name]
                        reward = transition.reward[name]
                        nv = next_value[name]
                        g = gae[name]
                        absorbing_key = agent_conf.config.env.agent[name].absorbing_key
                        absorbing = transition.absorbing_dict[absorbing_key]

                        delta = reward + config.gamma * nv * (1.0 - absorbing) - value
                        g = delta + config.gamma * config.gae_lambda * (1.0 - done) * g

                        new_gae[name] = g
                        adv_out[name] = g

                    return (new_gae, transition.value), adv_out

                init_gae = {name: jnp.zeros_like(last_vals[name]) for name in agent_names}
                init_next = last_vals

                (_, _), advantages = jax.lax.scan(
                    _get_advantages,
                    (init_gae, init_next),
                    traj_batch,
                    reverse=True,
                    unroll=16
                )

                # targets = advantages + values (per-agent)
                targets = {name: advantages[name] + traj_batch.value[name] for name in agent_names}
                return advantages, targets

            advantages, targets = _calculate_gae(traj_batch, last_vals)

            # UPDATE ACTOR & CRITIC NETWORK
            def _update_epoch(update_state, unused):
                train_states, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)

                batch_size = config.minibatch_size * config.num_minibatches
                assert batch_size == config.num_steps * config.num_envs

                permutation = jax.random.permutation(_rng, batch_size)

                # Build pytree batch
                batch = (traj_batch, advantages, targets)
                batch = jax.tree.map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
                shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=0), batch)
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(x, [config.num_minibatches, -1] + list(x.shape[1:])),
                    shuffled_batch
                )
                def _update_minibatch(train_states, batch_info):
                    traj_mb, adv_mb, tgt_mb = batch_info
                    new_train_states = dict(train_states)

                    # aggregate logs (optional)
                    loss_logs = {}

                    for name in agent_names:
                        ts = new_train_states[name]
                        net = agent_conf.networks[name]

                        mask = jnp.ones(traj_mb.done.shape, dtype=jnp.float32)
                        if stand_phase_enabled and name not in stand_phase_active_agents:
                            mask = 1.0 - traj_mb.info["no_ball"]
                            mask = mask.astype(jnp.float32)
                        mask_denom = jnp.sum(mask) + 1e-8

                        def _loss_fn(params):
                            # RERUN NETWORK
                            y, _ = net.apply({"params": params, "run_stats": ts.run_stats},
                                                traj_mb.obs, mutable=["run_stats"])
                            pi, value = y
                            # recompute logprob on stored per-agent actions
                            a = traj_mb.action_dict[name]
                            log_prob = pi.log_prob(a)

                            # value loss (clipped)
                            old_v = traj_mb.value[name]
                            v_clipped = old_v + (value - old_v).clip(-config.clip_eps, config.clip_eps)
                            v_loss1 = jnp.square(value - tgt_mb[name])
                            v_loss2 = jnp.square(v_clipped - tgt_mb[name])
                            value_loss = 0.5 * jnp.maximum(v_loss1, v_loss2)
                            value_loss = (value_loss * mask).sum() / mask_denom

                            # actor loss (PPO clip)
                            ratio = jnp.exp(log_prob - traj_mb.log_prob[name])
                            gae = adv_mb[name]
                            gae_mean = (gae * mask).sum() / mask_denom
                            gae_var = ((gae - gae_mean) ** 2 * mask).sum() / mask_denom
                            gae = (gae - gae_mean) / (jnp.sqrt(gae_var) + 1e-8)

                            loss1 = ratio * gae
                            loss2 = jnp.clip(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps) * gae
                            loss_actor = -(jnp.minimum(loss1, loss2) * mask).sum() / mask_denom

                            entropy = (pi.entropy() * mask).sum() / mask_denom

                            total_loss = loss_actor + config.vf_coef * value_loss - config.ent_coef * entropy
                            return total_loss, (value_loss, loss_actor, entropy, ratio)

                        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                        (total_loss, (value_loss, loss_actor, entropy, ratio)), grads = grad_fn(ts.params)\
                        
                        # apply grads
                        ts = ts.apply_gradients(grads=grads)

                    # adaptive lr (per-agent, same logic)
                        if config.get("adaptive_lr", False):
                            current_lr = ts.adaptive_lr_state.learning_rate
                            eps = 1e-7
                            approx_kl = ((ratio - 1.0 + eps) - jnp.log(ratio + eps))
                            approx_kl = (approx_kl * mask).sum() / mask_denom

                            next_lr = jax.lax.cond(
                                approx_kl > config.kl_target * config.kl_margin,
                                lambda lr: lr / config.kl_lr_scale,
                                lambda lr: lr,
                                current_lr,
                            )
                            next_lr = jax.lax.cond(
                                approx_kl < config.kl_target / config.kl_margin,
                                lambda lr: lr * config.kl_lr_scale,
                                lambda lr: lr,
                                next_lr,
                            )
                            next_lr = jnp.clip(next_lr, config.lr_min, config.lr_max)

                            # update injected hyperparams (same hack, per-agent)
                            old_h = ts.opt_state[1].hyperparams
                            new_h = {**old_h, "learning_rate": next_lr}
                            new_inject = ts.opt_state[1]._replace(hyperparams=new_h)
                            new_opt_state = tuple(ts.opt_state[i] if i != 1 else new_inject
                                                  for i in range(len(ts.opt_state)))
                            ts = ts.replace(opt_state=new_opt_state,
                                            adaptive_lr_state=AdaptiveLRState(learning_rate=next_lr))

                        new_train_states[name] = ts
                        loss_logs[name] = (total_loss, value_loss, loss_actor, entropy)

                    return new_train_states, loss_logs

                train_states, loss_logs = jax.lax.scan(_update_minibatch, train_states, minibatches)
                update_state = (train_states, traj_batch, advantages, targets, rng)
                return update_state, loss_logs

            update_state = (train_states, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config.update_epochs)

            train_states = update_state[0]
            rng = update_state[-1]

            ref_agent = agent_names[0]

            logged_metrics = traj_batch.metrics

            mean_episode_return_components = dict()
            for key in logged_metrics.returned_episode_return_components.keys():
                mean_episode_return_components[key] = jnp.sum(jnp.where(logged_metrics.done, logged_metrics.returned_episode_return_components[key], 0.0)) / jnp.sum(logged_metrics.done)

            metric = SummaryRichMetrics(
                mean_episode_return=jnp.sum(jnp.where(logged_metrics.done, logged_metrics.returned_episode_returns, 0.0)) / jnp.sum(logged_metrics.done),
                mean_episode_length=jnp.sum(jnp.where(logged_metrics.done, logged_metrics.returned_episode_lengths, 0.0)) / jnp.sum(logged_metrics.done),
                max_timestep=global_timesteps + jnp.max(logged_metrics.timestep * config.num_envs),
                frac_absorbed=jnp.sum(jnp.where(logged_metrics.done, logged_metrics.absorbed, 0.0)) / (jnp.sum(logged_metrics.done) + 1e-6),
                juggle_absorbed=jnp.sum(jnp.where(logged_metrics.done, logged_metrics.juggle_absorbed, 0.0)) / (jnp.sum(logged_metrics.done) + 1e-6),
                loco_absorbed=jnp.sum(jnp.where(logged_metrics.done, logged_metrics.loco_absorbed, 0.0)) / (jnp.sum(logged_metrics.done) + 1e-6),
                mean_episode_return_components=mean_episode_return_components,
                curriculum_step=jnp.mean(logged_metrics.curriculum_step),
            )

            def callback(metric, live_info):
                mean_ep_return = metric.mean_episode_return
                mean_ep_length = metric.mean_episode_length
                timestep = metric.max_timestep
                frac_absorbed = metric.frac_absorbed
                juggle_absorbed = metric.juggle_absorbed
                loco_absorbed = metric.loco_absorbed
                mean_ep_return_components = metric.mean_episode_return_components                
                curriculum_step = metric.curriculum_step

                if config.debug:
                    print(f"timestep={timestep}, episodic return={mean_ep_return}, episodic length={mean_ep_length}, absorbed={frac_absorbed}")
                else:
                    wandb_log_dict = dict()
                    wandb_log_dict["Live Info/Mean Episode Return"] = mean_ep_return
                    wandb_log_dict["Live Info/Mean Episode Length"] = mean_ep_length
                    wandb_log_dict["Live Info/Absorbed Envs"] = frac_absorbed
                    wandb_log_dict["Live Info/Curriculum Step"] = curriculum_step
                    wandb_log_dict["Live Info/Juggle Absorbed"] = juggle_absorbed
                    wandb_log_dict["Live Info/Loco Absorbed"] = loco_absorbed
                    # also log other live info
                    for key, value in live_info.items():
                        if "/" in key:
                            wandb_log_dict[key] = value
                        else:
                            wandb_log_dict["Live Info/" + key] = value

                    for key in mean_ep_return_components.keys():
                        group = "Live Return Components"
                        if mean_ep_return_components[key] != 0.0:
                            wandb_log_dict[group + '/' + key] = mean_ep_return_components[key]
                        else:
                            continue
                    wandb_run.log(wandb_log_dict, step=timestep)                

            ts = train_states[ref_agent]
            live_info = {
                "Learning Rate": ts.adaptive_lr_state.learning_rate
                if config.get("adaptive_lr", False) else config.lr,
            }
            if world_model_metrics is not None:
                wm_abs_count = jnp.sum(
                    traj_batch.info.get(
                        "wm_pred_disp_abs_count",
                        jnp.zeros_like(traj_batch.info["wm_pred_disp_abs_dist"]),
                    )
                )
                wm_abs_dist = jnp.sum(traj_batch.info["wm_pred_disp_abs_dist"]) / (wm_abs_count + 1e-8)
                live_info.update({
                    "World Model/train_loss": world_model_metrics["wm_loss"],
                    "World Model/train_delta_consistency_loss": world_model_metrics["wm_delta_consistency_loss"],
                    "World Model/train_disp_rmse": world_model_metrics["wm_disp_rmse"],
                    "World Model/train_vel_rmse": world_model_metrics["wm_vel_rmse"],
                    "World Model/rollout_disp_relative_distance": jnp.mean(traj_batch.info["wm_pred_disp_dist"]),
                    "World Model/rollout_disp_distance": wm_abs_dist,
                    "World Model/rollout_vel_rmse": jnp.sqrt(jnp.mean(traj_batch.info["wm_pred_vel_mse"]) + 1e-8),
                })
            jax.debug.callback(callback, metric, live_info=live_info)

            # no train state buffering during training (validation only at end)

            runner_state = (train_states, world_model_state, env_state, last_obs, train_state_buffer, rng)
            return runner_state, (metric, metric.max_timestep)

        rng, _rng = jax.random.split(rng)
        runner_state = (train_states, world_model_state, env_state, obsv, train_state_buffer, _rng)
        runner_state, (metric, global_timesteps) = jax.lax.scan(
            _update_step, runner_state, None, config.num_updates
        )

        agent_state = cls._agent_state(train_states=runner_state[0])
        world_state = runner_state[1]
        env_state_out = runner_state[2]

        return {"agent_state": agent_state,
                "world_state": world_state,
                "env_state": env_state_out,
                "training_metrics": metric,
                "global_timesteps": global_timesteps}

    @classmethod
    def play_policy(cls, env,
                    agent_conf: IPPOAgentConf,
                    agent_state: IPPOAgentState,
                    n_envs: int, n_steps=None, render=True,
                    record=False, rng=None, deterministic=False,
                    use_mujoco=False, wrap_env=True,
                    train_state_seed=None):

        if use_mujoco and wrap_env:
            if hasattr(agent_conf.experiment, "len_obs_history"):
                assert agent_conf.experiment.len_obs_history == 1, "len_obs_history must be 1 for mujoco envs."
        if use_mujoco:
            assert n_envs == 1, "Only one mujoco env can be run at a time."

        config = agent_conf.config.experiment
        train_states = agent_state.train_states
        agent_names = tuple(train_states.keys())

        if config.get("n_seeds", 1) > 1:
            assert train_state_seed is not None, (
                "Loaded train state has multiple seeds. "
                "Please specify train_state_seed for replay."
            )
            train_states = jax.tree.map(
                lambda x: x[train_state_seed],
                train_states
            )

        # ----------------------------
        # Deterministic mode
        # ----------------------------
        if deterministic:
            for name in agent_names:
                ts = train_states[name]
                ts = ts.replace(
                    params={
                        **ts.params,
                        "log_std": jnp.ones_like(ts.params["log_std"]) * -jnp.inf
                    }
                )
                train_states[name] = ts

        # ----------------------------
        # Action indices per agent
        # ----------------------------
        agent_action_idx = {
            name: jnp.array(agent_conf.config.env.agent[name].action_idx, dtype=jnp.int32)
            for name in agent_names
        }

        full_action_dim = env.info.action_space.shape[0]

        if not render and n_steps is None and not record:
            warnings.warn("No rendering, no record, no n_steps specified. This will run forever with no effect.")

        # create env
        # if wrap_env and not use_mujoco:
        #     env = cls._wrap_env(env, config)

        if rng is None:
            rng = jax.random.key(0)

        keys = jax.random.split(rng, n_envs + 1)
        rng, env_keys = keys[0], keys[1:]

        if use_mujoco:
            obs = env.reset()
            env_state = None
        else:
            keys = jax.random.split(rng, n_envs + 1)
            rng, env_keys = keys[0], keys[1:]
            from flax import nnx
            jv_reset = nnx.jit(nnx.vmap(env.mjx_reset, in_axes=(0, 0)))
            jv_step = nnx.jit(nnx.vmap(env.mjx_step, in_axes=(0, 0)))
            state = jv_reset(env_keys, jnp.arange(n_envs))
            obs = state.observation

        if n_steps is None:
            n_steps = jnp.iinfo(jnp.int32).max

        for i in range(n_steps):

            # SAMPLE ACTION
            rng, step_rng = jax.random.split(rng)
            agent_actions = {}
            new_train_states = dict(train_states)

            keys = jax.random.split(step_rng, len(agent_names) + 1)
            agent_keys = keys[1:]

            for name, k in zip(agent_names, agent_keys):
                ts = new_train_states[name]
                net = agent_conf.networks[name]

                y, updates = net.apply(
                    {"params": ts.params, "run_stats": ts.run_stats},
                    obs,
                    mutable=["run_stats"]
                )
                pi, _ = y

                ts = ts.replace(run_stats=updates["run_stats"])
                a = pi.sample(seed=k)

                agent_actions[name] = a
                new_train_states[name] = ts

            train_states = new_train_states

            # === scatter into full env action ===
            action = jnp.zeros((n_envs, full_action_dim), dtype=jnp.float32)
            for name, a in agent_actions.items():
                idx = agent_action_idx[name]
                action = action.at[:, idx].set(a)

            # STEP ENV
            if use_mujoco:
                obs, reward, absorbing, done, info = env.step(action)
            else:
                state = jv_step(state, action) 
                obs = state.observation

            # RENDER
            if use_mujoco:
                env.render(record=True)
            else:
                env.mjx_render(state)

            # RESET MUJOCO ENV (MJX resets by itself)
            if use_mujoco:
                if done:
                    obs = env.reset()

        env.stop()

    @classmethod
    def play_policy_mujoco(cls, env,
                           agent_conf: IPPOAgentConf,
                           agent_state: IPPOAgentState,
                           n_steps=None, render=True,
                           record=False, rng=None, deterministic=False,
                           train_state_seed=None):

        cls.play_policy(env, agent_conf, agent_state, 1, n_steps, render, record, rng, deterministic,
                        True, False, train_state_seed)

    @staticmethod
    def _wrap_env(env, config, world_conf: IPPOWorldConf | None = None):

        if "len_obs_history" in config and config.len_obs_history > 1:
            env = NStepWrapper(env, config.len_obs_history)
        env = RichLogWrapper(env)
        env = VecEnv(env)
        if config.normalize_env:
            env = NormalizeVecRewardDict(env, config.gamma)
        if world_conf is not None:
            wm_cfg = config.world_model
            env = WorldModelWrapper(
                env,
                world_conf.model,
                world_conf.obs_ind,
                buffer_length=wm_cfg.get("buffer_length", 0),
                buffer_dtype=wm_cfg.get("buffer_dtype", "bfloat16"),
                seg_len=wm_cfg.get("seg_len", 1),
            )
        return env
