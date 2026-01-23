import ast
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
from loco_mujoco.core.wrappers import RichLogWrapper, NStepWrapper, RichLogEnvState, VecEnv, NormalizeVecRewardDict, SummaryRichMetrics
from loco_mujoco.utils import MetricsHandler, ValidationSummary


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

def step_to_bin(step, B, max_episode_len):
    step = jnp.clip(step.astype(jnp.int32), 0, max_episode_len - 1)
    return (step * B) // max_episode_len  # [N] -> bins in [0,B-1]

def buffer_update(buf, value, obs, carry):
    """
    buf.qpos:  [B, nq]
    buf.qvel:  [B, nv]
    buf.value: [B]
    value: [N]
    step: [N]
    last_qpos: [N, nq]
    last_qvel: [N, nv]
    """
    B = buf.value.shape[1]
    N = value.shape[0]
    step = carry.time_step_in_episode
    env_id = carry.env_id
    dt = 0.02
    max_pattern_len = jnp.round(carry.pattern_state.pattern_cycle_time / dt).astype(jnp.int32)
    step_pattern_len = carry.time_step_in_episode % max_pattern_len
    bins = step_to_bin(step_pattern_len, B, max_pattern_len)               # [N]
    qpos = buf.last_qpos
    qvel = buf.last_qvel
    delta_action = carry.delta_action
    # 1) best candidate value per bin from the batch
    neg_inf = jnp.array(-jnp.inf, dtype=value.dtype)
    best_val_per_bin = jnp.full((B,), neg_inf, dtype=value.dtype)
    best_val_per_bin = best_val_per_bin.at[bins].max(value)   # scatter max

    # 2) pick one candidate index per bin achieving best_val_per_bin
    # mask candidates that are NOT best in their bin
    is_best = value == best_val_per_bin[bins]                 # [N] bool

    # tie-break: choose smallest index among best candidates in each bin
    idx = jnp.arange(N, dtype=jnp.int32)
    big = jnp.iinfo(jnp.int32).max
    idx_masked = jnp.where(is_best, idx, big)                 # [N]
    best_idx_per_bin = jnp.full((B,), big, dtype=jnp.int32)
    best_idx_per_bin = best_idx_per_bin.at[bins].min(idx_masked)  # scatter min

    # bins that received no candidate in this batch
    has_candidate = best_idx_per_bin != big                   # [B] bool

    # gather the chosen candidates (use safe index 0 where empty)
    safe_idx = jnp.where(has_candidate, best_idx_per_bin, 0)  # [B]
    qpos_new = qpos[safe_idx]                                 # [B, nq]
    qvel_new = qvel[safe_idx]                                 # [B, nv]
    val_new  = value[safe_idx]                                # [B]
    step_new = step[safe_idx].astype(jnp.int32)               # [B]
    jax.debug.print("step_new: {}", step_new)
    env_id_new = env_id[safe_idx].astype(jnp.int32)           # [B]
    obs_new = obs[safe_idx]                                   # [B, obs_dim]
    delta_action_new = delta_action[safe_idx]                 # [B, action_dim]
    # Replace rule vs existing buffer:
    # - if slot empty -> fill if has_candidate
    # - else replace only if val_new > buf.value
    buf_value = buf.value[0]
    buf_step = buf.step[0]
    buf_qpos = buf.qpos[0]
    buf_qvel = buf.qvel[0]
    buf_env_id = buf.env_id[0]
    buf_obs = buf.obs[0]
    buf_delta_action = buf.delta_action[0]
    do_update = has_candidate & (val_new > buf_value)

    qpos_out = jnp.where(do_update[:, None], qpos_new, buf_qpos)[None, :, :]
    qvel_out = jnp.where(do_update[:, None], qvel_new, buf_qvel)[None, :, :]
    val_out  = jnp.where(do_update, val_new, buf_value)[None, :]
    step_out = jnp.where(do_update, step_new, buf_step)[None, :]
    jax.debug.print("do_update: {}", do_update)
    jax.debug.print("step_out: {}", step_out)
    env_id_out = jnp.where(do_update, env_id_new, buf_env_id)[None, :]
    obs_out = jnp.where(do_update[:, None], obs_new, buf_obs)[None, :, :]
    delta_action_out = jnp.where(do_update[:, None], delta_action_new, buf_delta_action)[None, :, :]
    # repeat buf.value with val_out
    buf_qpos_out = jnp.repeat(qpos_out, N, axis=0)
    buf_qvel_out = jnp.repeat(qvel_out, N, axis=0)
    buf_value_out = jnp.repeat(val_out, N, axis=0)
    buf_step_out = jnp.repeat(step_out, N, axis=0)
    buf_env_id_out = jnp.repeat(env_id_out, N, axis=0)
    buf_obs_out = jnp.repeat(obs_out, N, axis=0)
    buf_delta_action_out = jnp.repeat(delta_action_out, N, axis=0)
    return buf.replace(qpos=buf_qpos_out, qvel=buf_qvel_out, value=buf_value_out, step=buf_step_out, env_id=buf_env_id_out, obs=buf_obs_out, delta_action=buf_delta_action_out)

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
    def _train_fn(cls, rng, env,
                  agent_conf: IPPOAgentConf,
                  agent_state: IPPOAgentState = None,
                  mh: MetricsHandler = None,
                  wandb_run=None,
                  ):

        # extract static agent info
        config = agent_conf.config.experiment

        env = cls._wrap_env(env, config)

        agent_names = tuple(agent_conf.networks.keys())

        # === CHANGE: action indices per agent for scattering ===
        agent_action_idx = {
            name: jnp.array(agent_conf.config.env.agent[name].action_idx, dtype=jnp.int32)
            for name in agent_names
        }

        full_action_dim = env.info.action_space.shape[0]
        num_envs = config.num_envs

        # extract current agent state
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

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config.num_envs)
        obsv, env_state = env.reset(reset_rng, env_id=jnp.arange(config.num_envs))

        train_state_buffer = TrainStateBuffer.create(next(iter(train_states.values())), config.validation.num)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_states, env_state, last_obs, train_state_buffer, rng = runner_state

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
                # update absorb ratio
                absorb_ratio = jnp.sum(jnp.where(logged_metrics.done, logged_metrics.absorbed, 0.0)) / (jnp.sum(logged_metrics.done) + 1e-6)
                absorb_ratio = absorb_ratio * jnp.ones_like(env_state.additional_carry.curriculum.absorb_ratio)
                curriculum = env_state.additional_carry.curriculum
                curriculum = curriculum.replace(absorb_ratio=absorb_ratio)
                env_state = env_state.replace(
                    env_state=env_state.env_state.replace(
                        env_state=env_state.env_state.env_state.replace(
                            additional_carry=env_state.env_state.env_state.additional_carry.replace(curriculum=curriculum)
                        )
                    )
                )

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
                runner_state = (new_train_states, env_state, obsv, train_state_buffer, _rng_next)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config.num_steps
            )

            # CALCULATE ADVANTAGE
            train_states, env_state, last_obs, train_state_buffer, rng = runner_state
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
                            value_loss = 0.5 * jnp.maximum(v_loss1, v_loss2).mean()

                            # actor loss (PPO clip)
                            ratio = jnp.exp(log_prob - traj_mb.log_prob[name])
                            gae = adv_mb[name]
                            gae = (gae - gae.mean()) / (gae.std() + 1e-8)

                            loss1 = ratio * gae
                            loss2 = jnp.clip(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps) * gae
                            loss_actor = -jnp.minimum(loss1, loss2).mean()

                            entropy = pi.entropy().mean()

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
                            approx_kl = jnp.mean((ratio - 1.0 + eps) - jnp.log(ratio + eps))

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
            ref_train_state = train_states[ref_agent]

            counter = ((ref_train_state.step + 1)
                    // config.num_minibatches
                    // config.update_epochs)

            logged_metrics = traj_batch.metrics

            mean_episode_return_components = dict()
            for key in logged_metrics.returned_episode_return_components.keys():
                mean_episode_return_components[key] = jnp.sum(jnp.where(logged_metrics.done, logged_metrics.returned_episode_return_components[key], 0.0)) / jnp.sum(logged_metrics.done)

            metric = SummaryRichMetrics(
                mean_episode_return=jnp.sum(jnp.where(logged_metrics.done, logged_metrics.returned_episode_returns, 0.0)) / jnp.sum(logged_metrics.done),
                mean_episode_length=jnp.sum(jnp.where(logged_metrics.done, logged_metrics.returned_episode_lengths, 0.0)) / jnp.sum(logged_metrics.done),
                max_timestep=jnp.max(logged_metrics.timestep * config.num_envs),
                frac_absorbed=jnp.sum(jnp.where(logged_metrics.done, logged_metrics.absorbed, 0.0)) / (jnp.sum(logged_metrics.done) + 1e-6),
                mean_episode_return_components=mean_episode_return_components,
                curriculum_step=jnp.mean(logged_metrics.curriculum_step),
            )

            def _evaluation_step():

                def _eval_env(runner_state, unused):
                    train_states, env_state, last_obs, train_state_buffer, rng = runner_state

                    # SELECT ACTION
                    rng, _rng = jax.random.split(rng)
                    agent_actions = {}
                    new_train_states = dict(train_states)

                    keys = jax.random.split(_rng, len(agent_names) + 1)
                    rng_next = keys[0]
                    agent_keys = keys[1:]

                    for name, k in zip(agent_names, agent_keys):
                        ts = new_train_states[name]
                        net = agent_conf.networks[name]

                        y, updates = net.apply(
                            {"params": ts.params, "run_stats": ts.run_stats},
                            last_obs,
                            mutable=["run_stats"],
                        )
                        pi, _ = y
                        ts = ts.replace(run_stats=updates["run_stats"])
                        a = pi.sample(seed=k)

                        agent_actions[name] = a
                        new_train_states[name] = ts

                    action = _scatter_actions(
                        config.validation.num_envs,
                        env.info.action_space.shape[0],
                        agent_actions,
                        agent_action_idx
                    )
                    # STEP ENV
                    obsv, reward, absorbing, done, info, env_state = env.step(env_state, action)

                    # GET METRICS
                    log_env_state = env_state.find(RichLogEnvState)
                    logged_metrics = log_env_state.metrics

                    transition = MetricHandlerTransition(env_state, logged_metrics)

                    runner_state = (new_train_states, env_state, obsv, train_state_buffer, rng_next)
                    return runner_state, transition
                

                rng = runner_state[-1]
                reset_rng = jax.random.split(rng, config.validation.num_envs)
                obsv, env_state = env.reset(reset_rng, env_id=jnp.arange(config.validation.num_envs))
                runner_state_eval = (train_states, env_state, obsv, train_state_buffer, rng)

                # do evaluation runs
                _, traj_batch_eval = jax.lax.scan(
                    _eval_env, runner_state_eval, None, config.validation.num_steps
                )

                env_states = traj_batch_eval.env_state

                validation_metrics = mh(env_states)

                return validation_metrics

            if mh is None:
                validation_metrics = ValidationSummary()
            else:
                validation_metrics = jax.lax.cond(counter % config.validation_interval == 0, _evaluation_step,
                                                   mh.get_zero_container)

            def callback(metric, live_info, validation_info):
                mean_ep_return = metric.mean_episode_return
                mean_ep_length = metric.mean_episode_length
                timestep = metric.max_timestep
                frac_absorbed = metric.frac_absorbed
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
                    # also log other live info
                    for key, value in live_info.items():
                        wandb_log_dict["Live Info/" + key] = value

                    for key, value in validation_info.items():
                        wandb_log_dict["Validation Info/" + key] = value

                    for key in mean_ep_return_components.keys():
                        group = "Live Return Components"
                        if mean_ep_return_components[key] != 0.0:
                            wandb_log_dict[group + '/' + key] = mean_ep_return_components[key]
                        else:
                            continue
                    wandb_run.log(wandb_log_dict, step=timestep)                

            # --- add validation metrics when applicable ---
            def make_validation_info(_):
                return {
                    "Mean Return": validation_metrics.mean_episode_return,
                    "Mean Length": validation_metrics.mean_episode_length,
                }

            def empty_validation_info(_):
                return {
                    "Mean Return": jnp.nan,
                    "Mean Length": jnp.nan,
                }

            validation_info = jax.lax.cond(
                (counter % config.validation_interval) == 0,
                make_validation_info,
                empty_validation_info,
                operand=None
            )
            ts = train_states[ref_agent]
            live_info = {
                "Learning Rate": ts.adaptive_lr_state.learning_rate
                if config.get("adaptive_lr", False) else config.lr,
            }
            jax.debug.callback(callback, metric, live_info=live_info, validation_info=validation_info)

            # add train state to buffer if needed
            train_state_buffer = jax.lax.cond(
                counter % config.validation_interval == 0,
                lambda x, y: TrainStateBuffer.add(x, y),
                lambda x, y: x,
                train_state_buffer,
                ref_train_state
            )

            runner_state = (train_states, env_state, last_obs, train_state_buffer, rng)
            return runner_state, (metric, validation_metrics)

        rng, _rng = jax.random.split(rng)
        runner_state = (train_states, env_state, obsv, train_state_buffer, _rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config.num_updates
        )

        agent_state = cls._agent_state(train_states=runner_state[0])

        return {"agent_state": agent_state,
                "training_metrics": metrics[0],
                "validation_metrics": metrics[1]}

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
        if wrap_env and not use_mujoco:
            env = cls._wrap_env(env, config)

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
            obs, env_state = env.reset(env_keys, env_id=jnp.arange(n_envs))

        if n_steps is None:
            n_steps = jnp.iinfo(jnp.int32).max

        for i in range(n_steps):

            # SAMPLE ACTION
            rng, step_rng = jax.random.split(rng)
            agent_actions = {}
            new_train_states = dict(train_states)

            keys = jax.random.split(step_rng, len(agent_names) + 1)
            agent_keys = keys[1:]
            total_values = jnp.zeros(n_envs)

            for name, k in zip(agent_names, agent_keys):
                ts = new_train_states[name]
                net = agent_conf.networks[name]

                y, updates = net.apply(
                    {"params": ts.params, "run_stats": ts.run_stats},
                    obs,
                    mutable=["run_stats"]
                )
                pi, value = y
                total_values = total_values + value

                ts = ts.replace(run_stats=updates["run_stats"])
                a = pi.sample(seed=k)

                agent_actions[name] = a
                new_train_states[name] = ts

            train_states = new_train_states
            # init_state_buffer = jax.lax.cond(i < 50, lambda x: buffer_update(x, total_values, obs, env_state.additional_carry), lambda x: x, env_state.additional_carry.init_state_buffer)
            # env_state = env_state.replace(
            #         env_state=env_state.env_state.replace(
            #             env_state=env_state.env_state.env_state.replace(
            #                 additional_carry=env_state.env_state.env_state.additional_carry.replace(init_state_buffer=init_state_buffer)
            #             )
            #         )
            #     )

            # === scatter into full env action ===
            action = jnp.zeros((n_envs, full_action_dim), dtype=jnp.float32)
            for name, a in agent_actions.items():
                idx = agent_action_idx[name]
                action = action.at[:, idx].set(a)

            # STEP ENV
            if use_mujoco:
                obs, reward, absorbing, done, info = env.step(action)
            else:
                obs, reward, absorbing, done, info, env_state = env.step(env_state, action)

            # RENDER
            if use_mujoco:
                env.render(record=True)
            else:
                env.mjx_render(env_state, record=record)

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
    def _wrap_env(env, config):

        if "len_obs_history" in config and config.len_obs_history > 1:
            env = NStepWrapper(env, config.len_obs_history)
        env = RichLogWrapper(env)
        env = VecEnv(env)
        if config.normalize_env:
            env = NormalizeVecRewardDict(env, config.gamma)
        return env
