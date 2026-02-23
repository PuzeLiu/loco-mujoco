from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import flax
import jax
import jax.numpy as jnp
import optax
from flax import struct
from omegaconf import DictConfig, OmegaConf

from loco_mujoco.algorithms.common.dataclasses import TrainState
from loco_mujoco.algorithms.common.world_model import WorldModel, init_target_norm_stats


def _get_world_model_optimizer(config: DictConfig):
    wm_cfg = config.experiment.world_model
    return optax.chain(
        optax.clip_by_global_norm(config.experiment.max_grad_norm),
        optax.adamw(wm_cfg.lr, weight_decay=wm_cfg.weight_decay, eps=1e-5),
    )


@dataclass(frozen=True)
class IPPOWorldConf:
    config: DictConfig
    model: WorldModel
    tx: Any
    obs_ind: jnp.ndarray
    obs_group: Optional[str] = None

    def serialize(self) -> dict:
        conf_dict = OmegaConf.to_container(self.config, resolve=True, throw_on_missing=True)
        return {
            "config": conf_dict,
            "model": flax.serialization.to_state_dict(self.model),
            "obs_ind": self.obs_ind.tolist(),
            "obs_group": self.obs_group,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "IPPOWorldConf":
        config = OmegaConf.create(d["config"])
        wm_cfg = config.experiment.world_model
        mamba_dt_rank = wm_cfg.get("mamba_dt_rank", 1)
        if mamba_dt_rank is None:
            mamba_dt_rank = 1
        mamba_dt_min = wm_cfg.get("mamba_dt_min", 1e-4)
        if mamba_dt_min is None:
            mamba_dt_min = 1e-4
        mamba_dt_max = wm_cfg.get("mamba_dt_max", 1.0)
        if mamba_dt_max is None:
            mamba_dt_max = 1.0
        model = WorldModel(
            input_dim=wm_cfg.input_dim,
            model_dim=wm_cfg.model_dim,
            n_layers=wm_cfg.n_layers,
            n_heads=wm_cfg.n_heads,
            ff_dim=wm_cfg.ff_dim,
            dropout=wm_cfg.dropout,
            mamba_expand=wm_cfg.get("mamba_expand", None),
            mamba_d_state=wm_cfg.get("mamba_d_state", None),
            mamba_d_conv=wm_cfg.get("mamba_d_conv", None),
            mamba_dt_rank=int(mamba_dt_rank),
            mamba_dt_min=float(mamba_dt_min),
            mamba_dt_max=float(mamba_dt_max),
        )
        model = flax.serialization.from_state_dict(model, d["model"])
        obs_ind = jnp.array(d.get("obs_ind", wm_cfg.get("obs_ind", [])), dtype=jnp.int32)
        obs_group = d.get("obs_group", wm_cfg.get("obs_group", None))
        tx = _get_world_model_optimizer(config)
        return cls(config=config, model=model, tx=tx, obs_ind=obs_ind, obs_group=obs_group)


@struct.dataclass
class IPPOWorldState:
    train_state: TrainState
    target_norm_stats: dict = struct.field(pytree_node=True, default_factory=lambda: init_target_norm_stats(dim=6, dtype=jnp.float32))

    def serialize(self) -> dict:
        return {
            "train_state": flax.serialization.to_state_dict(self.train_state),
            "target_norm_stats": flax.serialization.to_state_dict(self.target_norm_stats),
        }

    @classmethod
    def from_dict(cls, d: dict, conf: IPPOWorldConf) -> "IPPOWorldState":
        ts = TrainState(
            apply_fn=conf.model.apply,
            tx=conf.tx,
            **d["train_state"],
        )
        if "target_norm_stats" in d:
            stats_template = init_target_norm_stats(dim=6, dtype=jnp.float32)
            target_norm_stats = flax.serialization.from_state_dict(stats_template, d["target_norm_stats"])
        else:
            target_norm_stats = init_target_norm_stats(dim=6, dtype=jnp.float32)
        return cls(train_state=ts, target_norm_stats=target_norm_stats)


def save_world_model(path: str | Path,
                     conf: IPPOWorldConf,
                     state: IPPOWorldState) -> Path:
    path = Path(path)
    if path.is_dir():
        path = path / "IPPOWorld_saved.pkl"
    data = {
        "world_model_conf": conf.serialize(),
        "world_model_state": state.serialize(),
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return path


def load_world_model(path: str | Path) -> tuple[IPPOWorldConf, IPPOWorldState]:
    path = Path(path)
    with open(path, "rb") as f:
        data = pickle.load(f)
    conf = IPPOWorldConf.from_dict(data["world_model_conf"])
    state = IPPOWorldState.from_dict(data["world_model_state"], conf)
    return conf, state


# Backwards compatibility
WorldModelAgentConf = IPPOWorldConf
WorldModelAgentState = IPPOWorldState
