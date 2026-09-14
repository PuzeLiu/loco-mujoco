from types import ModuleType
from typing import Any, Dict, Tuple, Union

import numpy as np
import jax
import jax.numpy as jnp
from flax import struct
from jax._src.scipy.spatial.transform import Rotation as jnp_R
from scipy.spatial.transform import Rotation as np_R
import mujoco
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model
from omegaconf import ListConfig
from loco_mujoco.core.reward.base import Reward
from loco_mujoco.core.utils import mj_jntname2qposid, mj_jntname2qvelid, mj_jntid2qposid, mj_check_collisions
from loco_mujoco.core.utils.math import quat_scalarfirst2scalarlast


class NoReward(Reward):
    """
    A reward function that returns always 0.

    """

    def __call__(self,
                 state: Union[np.ndarray, jnp.ndarray],
                 action: Union[np.ndarray, jnp.ndarray],
                 next_state: Union[np.ndarray, jnp.ndarray],
                 absorbing: bool,
                 info: Dict[str, Any],
                 env: Any,
                 model: Union[MjModel, Model],
                 data: Union[MjData, Data],
                 carry: Any,
                 backend: ModuleType) -> Tuple[float, Any]:
        """
        Return zero.

        Args:
            state (Union[np.ndarray, jnp.ndarray]): Last state.
            action (Union[np.ndarray, jnp.ndarray]): Applied action.
            next_state (Union[np.ndarray, jnp.ndarray]): Current state.
            absorbing (bool): Whether the state is absorbing.
            info (Dict[str, Any]): Additional information.
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Additional carry.
            backend (ModuleType): Backend module used for computation (either numpy or jax.numpy).

        Returns:
            Tuple[float, Any]: The reward for the current transition and the updated carry.

        """
        return 0.0, carry


class TargetXVelocityReward(Reward):
    """
    Reward function that computes the reward based on the deviation from the root's
    target velocity in the x-direction.

    """
    def __init__(self, env: Any, target_velocity: float, **kwargs):
        """
        Initialize the reward function.

        Args:
            env (Any): The environment instance.
            target_velocity (float): The target velocity.
            **kwargs (Any): Additional keyword arguments.

        """
        super().__init__(env, **kwargs)
        self._target_vel = target_velocity
        root_free_joint_xml_name = self._info_props["root_free_joint_xml_name"]
        self._x_vel_idx = mj_jntname2qvelid(root_free_joint_xml_name, env._model)[0]

    def __call__(self,
                 state: Union[np.ndarray, jnp.ndarray],
                 action: Union[np.ndarray, jnp.ndarray],
                 next_state: Union[np.ndarray, jnp.ndarray],
                 absorbing: bool,
                 info: Dict[str, Any],
                 env: Any,
                 model: Union[MjModel, Model],
                 data: Union[MjData, Data],
                 carry: Any,
                 backend: ModuleType) -> Tuple[float, Any]:
        """
        Compute the reward based on deviation from target velocity in x-direction.

        Args:
            state (Union[np.ndarray, jnp.ndarray]): Last state.
            action (Union[np.ndarray, jnp.ndarray]): Applied action.
            next_state (Union[np.ndarray, jnp.ndarray]): Current state.
            absorbing (bool): Whether the state is absorbing.
            info (Dict[str, Any]): Additional information.
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Additional carry.
            backend (ModuleType): Backend module used for computation (either numpy or jax.numpy).

        Returns:
            Tuple[float, Any]: The reward for the current transition and the updated carry.

        """
        x_vel = backend.squeeze(data.qvel[self._x_vel_idx])
        return backend.exp(-backend.square(x_vel - self._target_vel)), carry


class TargetVelocityGoalReward(Reward):
    """
    Reward function that computes the reward based on the deviation from the goal velocity. The goal velocity is
    provided as an observation in the environment. The reward is computed as the negative exponential of the squared
    difference between the current velocity and the goal velocity. The reward is computed for the x, y, and yaw
    velocities of the root.

    """

    def __init__(self, env: Any, tracking_w_exp_xy=10.0, tracking_w_exp_yaw=10.0,
                 tracking_w_sum_xy=1.0, tracking_w_sum_yaw=1.0, **kwargs):
        """
        Initialize the reward function.

        Args:
            env (Any): The environment instance.
            tracking_w_exp_xy (float, optional): The exponential weight for xy-tracking reward.
            tracking_w_exp_yaw (float, optional): The exponential weight for yaw-tracking reward.
            **kwargs (Any): Additional keyword arguments.

        """

        super().__init__(env, **kwargs)

        self._free_jnt_name = self._info_props["root_free_joint_xml_name"]
        self._vel_idx = np.array(mj_jntname2qvelid(self._free_jnt_name, env._model))
        self._w_exp_xy = tracking_w_exp_xy
        self._w_exp_yaw = tracking_w_exp_yaw
        self._w_sum_xy = tracking_w_sum_xy
        self._w_sum_yaw = tracking_w_sum_yaw

        # find the goal velocity observation
        assert "GoalRandomRootVelocity" in env.obs_container, \
            f"GoalRandomRootVelocity is the required goal for the reward for{self.__class__.__name__}"

        super().__init__(env, **kwargs)

    def __call__(self,
                 state: Union[np.ndarray, jnp.ndarray],
                 action: Union[np.ndarray, jnp.ndarray],
                 next_state: Union[np.ndarray, jnp.ndarray],
                 absorbing: bool,
                 info: Dict[str, Any],
                 env: Any,
                 model: Union[MjModel, Model],
                 data: Union[MjData, Data],
                 carry: Any,
                 backend: ModuleType) -> Tuple[float, Any]:
        """
        Computes a tracking reward based on the deviation from the goal velocity.Tracking is done on the x, y, and yaw
        velocities of the root.

        Args:
            state (Union[np.ndarray, jnp.ndarray]): Last state.
            action (Union[np.ndarray, jnp.ndarray]): Applied action.
            next_state (Union[np.ndarray, jnp.ndarray]): Current state.
            absorbing (bool): Whether the state is absorbing.
            info (Dict[str, Any]): Additional information.
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Additional carry.
            backend (ModuleType): Backend module used for computation (either numpy or jax.numpy).

        Returns:
            Tuple[float, Any]: The reward for the current transition and the updated carry.
        """
        if backend == np:
            R = np_R
        else:
            R = jnp_R

        goal_state = getattr(carry.observation_states, "GoalRandomRootVelocity")

        # get root orientation
        root_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, self._free_jnt_name)

        assert root_jnt_id != -1, f"Joint {self._free_jnt_name} not found in the model."
        root_jnt_qpos_start_id = model.jnt_qposadr[root_jnt_id]
        root_qpos = backend.squeeze(data.qpos[root_jnt_qpos_start_id:root_jnt_qpos_start_id+7])
        root_quat = R.from_quat(quat_scalarfirst2scalarlast(root_qpos[3:7]))

        # get current local vel of root
        lin_vel_global = backend.squeeze(data.qvel[self._vel_idx])[:3]
        ang_vel_global = backend.squeeze(data.qvel[self._vel_idx])[3:]
        lin_vel_local = root_quat.as_matrix().T @ lin_vel_global
        vel_local = backend.concatenate([lin_vel_local[:2], backend.atleast_1d(ang_vel_global[2])]) # construct vel, x, y and yaw

        # calculate tracking reward
        goal_vel = backend.array([goal_state.goal_vel_x, goal_state.goal_vel_y, goal_state.goal_vel_yaw])
        tracking_reward_xy = backend.exp(-self._w_exp_xy * backend.mean(backend.square(vel_local[:2] - goal_vel[:2])))
        tracking_reward_yaw = backend.exp(-self._w_exp_yaw * backend.mean(backend.square(vel_local[2] - goal_vel[2])))
        total_tracking = self._w_sum_xy * tracking_reward_xy + self._w_sum_yaw * tracking_reward_yaw

        return total_tracking, carry


@struct.dataclass
class LocomotionRewardState:
    """
    State of LocomotionReward.
    """
    last_qvel: Union[np.ndarray, jax.Array]
    last_action: Union[np.ndarray, jax.Array]
    time_since_last_touchdown: Union[np.ndarray, jax.Array]
    reward_components: Dict[str, Union[np.ndarray, jax.Array]]


class LocomotionReward(TargetVelocityGoalReward):

    """
    Reward function extending the TargetVelocityGoalReward with typical additional penalties
    and regularization terms for locomotion. This reward is stateful: LocomotionRewardState

    """

    def __init__(self, env: Any, **kwargs):
        """
        Initialize the reward function.

        Args:
            env (Any): The environment instance.
            **kwargs (Any): Additional keyword arguments.

        """
        super().__init__(env, **kwargs)

        model = env._model
        self._free_joint_qpos_ind = np.array(mj_jntname2qposid(self._info_props["root_free_joint_xml_name"], model))
        self._free_joint_qvel_ind = np.array(mj_jntname2qvelid(self._info_props["root_free_joint_xml_name"], model))
        self._free_joint_qpos_mask = np.zeros(model.nq, dtype=bool)
        self._free_joint_qpos_mask[self._free_joint_qpos_ind] = True
        self._free_joint_qvel_mask = np.zeros(model.nv, dtype=bool)
        self._free_joint_qvel_mask[self._free_joint_qvel_ind] = True
        self._foot_names = self._info_props["foot_geom_names"]

        self._floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self._foot_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in self._foot_names]

        # reward coefficients
        self._z_vel_coeff = kwargs.get("z_vel_coeff", 2.0)
        self._roll_pitch_vel_coeff = kwargs.get("roll_pitch_vel_coeff", 5e-2)
        self._roll_pitch_pos_coeff = kwargs.get("roll_pitch_pos_coeff", 2e-1)
        self._nominal_joint_pos_coeff = kwargs.get("nominal_joint_pos_coeff", 0.0)
        self._nominal_joint_pos_names = kwargs.get("nominal_joint_pos_names", None)
        self._joint_position_limit_coeff = kwargs.get("joint_position_limit_coeff", 10.0)
        self._joint_vel_coeff = kwargs.get("joint_vel_coeff", 0.0)
        self._joint_acc_coeff = kwargs.get("joint_acc_coeff", 2e-7)
        self._joint_torque_coeff = kwargs.get("joint_torque_coeff", 2e-5)
        self._action_rate_coeff = kwargs.get("action_rate_coeff", 1e-2)
        self._air_time_max = kwargs.get("air_time_max", 0.0)
        self._air_time_coeff = kwargs.get("air_time_coeff", 0.0)
        self._symmetry_air_coeff = kwargs.get("symmetry_air_coeff", 0.0)
        self._energy_coeff = kwargs.get("energy_coeff", 0.0)

        # get limits and nominal joint positions
        self._limited_joints = np.array(model.jnt_limited, dtype=bool)
        self._limited_joints_qpos_id = model.jnt_qposadr[np.where(self._limited_joints)]
        self._joint_ranges = model.jnt_range[self._limited_joints]
        self._nominal_joint_qpos = env._model.qpos0
        if self._nominal_joint_pos_names is None:
            # take all limited joints
            self._nominal_joint_qpos_id = self._limited_joints_qpos_id
        else:
            self._nominal_joint_qpos_id = np.concatenate([mj_jntname2qposid(name, model)
                                                          for name in self._nominal_joint_pos_names])

    def init_state(self, env: Any,
                   key: Any,
                   model: Union[MjModel, Model],
                   data: Union[MjData, Data],
                   backend: ModuleType):
        """
        Initialize the reward state.

        Args:
            env (Any): The environment instance.
            key (Any): Key for the reward state.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            backend (ModuleType): Backend module used for computation (either numpy or jax.numpy).

        Returns:
            LocomotionRewardState: The initialized reward state.

        """
        return LocomotionRewardState(
            last_qvel=data.qvel,
            last_action=backend.zeros(env.info.action_space.shape[0]),
            time_since_last_touchdown=backend.zeros(len(self._foot_ids)),
            reward_components={
                "tracking/main_goal": 0.0,
                "penalties/z_velocity": 0.0,
                "penalties/roll_pitch_velocity": 0.0,
                "penalties/roll_pitch_position": 0.0,
                "penalties/nominal_joint_position": 0.0,
                "penalties/joint_position_limit": 0.0,
                "penalties/joint_velocity": 0.0,
                "penalties/joint_acceleration": 0.0,
                "penalties/joint_torque": 0.0,
                "penalties/action_rate": 0.0,
                "penalties/air_time": 0.0,
                "penalties/gait_symmetry": 0.0,
                "penalties/energy": 0.0,
            }
        )

    def reset(self,
              env: Any,
              model: Union[MjModel, Model],
              data: Union[MjData, Data],
              carry: Any,
              backend: ModuleType):
        """
        Reset the reward state.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Additional carry.
            backend (ModuleType): Backend module used for computation (either numpy or jax.numpy).

        Returns:
            Tuple[Union[MjData, Data], Any]: The updated data and carry.

        """
        reward_state = self.init_state(env, None, model, data, backend)
        carry = carry.replace(reward_state=reward_state)
        return data, carry

    def __call__(self,
                 state: Union[np.ndarray, jnp.ndarray],
                 action: Union[np.ndarray, jnp.ndarray],
                 next_state: Union[np.ndarray, jnp.ndarray],
                 absorbing: bool,
                 info: Dict[str, Any],
                 env: Any,
                 model: Union[MjModel, Model],
                 data: Union[MjData, Data],
                 carry: Any,
                 backend: ModuleType) -> Tuple[float, Any]:
        """
        Based on the tracking reward, this reward function adds typical penalties and regularization terms
        for locomotion.

        Args:
            state (Union[np.ndarray, jnp.ndarray]): Last state.
            action (Union[np.ndarray, jnp.ndarray]): Applied action.
            next_state (Union[np.ndarray, jnp.ndarray]): Current state.
            absorbing (bool): Whether the state is absorbing.
            info (Dict[str, Any]): Additional information.
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Additional carry.
            backend (ModuleType): Backend module used for computation (either numpy or jax.numpy).

        Returns:
            Tuple[float, Any]: The reward for the current transition and the updated carry.
        """

        if backend == np:
            R = np_R
        else:
            R = jnp_R

        # get current reward state
        reward_state = carry.reward_state

        # get global pose quantities
        global_pose_root = data.qpos[self._free_joint_qpos_ind]
        global_pos_root = global_pose_root[:3]
        global_quat_root = global_pose_root[3:]
        global_rot = R.from_quat(quat_scalarfirst2scalarlast(global_quat_root))

        # get global velocity quantities
        global_vel_root = data.qvel[self._free_joint_qvel_ind]

        # get local velocity quantities
        local_vel_root_lin = global_rot.inv().apply(global_vel_root[:3])
        local_vel_root_ang = global_rot.inv().apply(global_vel_root[3:])

        # velocity reward
        if self._z_vel_coeff > 0.0:
            z_vel_reward = self._z_vel_coeff * -(backend.square(local_vel_root_lin[2]))
        else:
            z_vel_reward = 0.0
        if self._roll_pitch_vel_coeff > 0.0:
            roll_pitch_vel_reward = self._roll_pitch_vel_coeff * -backend.square(local_vel_root_ang[:2]).sum()
        else:
            roll_pitch_vel_reward = 0.0

        # position reward
        if self._roll_pitch_pos_coeff > 0.0:
            euler = global_rot.as_euler("xyz")
            roll_pitch_reward = self._roll_pitch_pos_coeff * -backend.square(euler[:2]).sum()
        else:
            roll_pitch_reward = 0.0

        # nominal joint pos reward
        if self._nominal_joint_pos_coeff > 0.0:
            joint_qpos_reward = (self._nominal_joint_pos_coeff *
                                 -backend.square(data.qpos[self._nominal_joint_qpos_id] -
                                                 self._nominal_joint_qpos[self._nominal_joint_qpos_id]).sum())
        else:
            joint_qpos_reward = 0.0

        # joint position limit reward
        if self._joint_position_limit_coeff > 0.0:
            joint_positions = backend.array(data.qpos[self._limited_joints_qpos_id])
            lower_limit_penalty = -backend.minimum(joint_positions - self._joint_ranges[:, 0], 0.0).sum()
            upper_limit_penalty = backend.maximum(joint_positions - self._joint_ranges[:, 1], 0.0).sum()
            joint_position_limit_reward = self._joint_position_limit_coeff * -(lower_limit_penalty + upper_limit_penalty)
        else:
            joint_position_limit_reward = 0.0

        # joint velocity reward
        joint_vel = data.qvel[~self._free_joint_qvel_mask]
        if self._joint_vel_coeff > 0.0:
            joint_vel_reward = self._joint_vel_coeff * -backend.square(joint_vel).sum()
        else:
            joint_vel_reward = 0.0

        # joint acceleration reward
        if self._joint_acc_coeff > 0.0:
            last_joint_vel = reward_state.last_qvel[~self._free_joint_qvel_mask]
            acceleration_norm = backend.sum(backend.square(joint_vel - last_joint_vel) / env.dt)
            acceleration_reward = self._joint_acc_coeff * -acceleration_norm
        else:
            acceleration_reward = 0.0

        # joint torque reward
        if self._joint_torque_coeff > 0.0:
            torque_norm = backend.sum(backend.square(data.qfrc_actuator[~self._free_joint_qvel_mask]))
            torque_reward = self._joint_torque_coeff * -torque_norm
        else:
            torque_reward = 0.0

        # action rate reward
        if self._action_rate_coeff > 0.0:
            action_rate_norm = backend.sum(backend.square(action - reward_state.last_action))
            action_rate_reward = self._action_rate_coeff * -action_rate_norm
        else:
            action_rate_reward = 0.0

        # air time reward
        if self._air_time_coeff > 0.0 or self._symmetry_air_coeff > 0.0:
            air_time_reward = 0.0
            foots_on_ground = backend.zeros(len(self._foot_ids))
            tslt = reward_state.time_since_last_touchdown.copy()
            for i, f_id in enumerate(self._foot_ids):
                foot_on_ground = mj_check_collisions(f_id, self._floor_id, data, backend)
                if backend == np:
                    foots_on_ground[i] = foot_on_ground
                else:
                    foots_on_ground = foots_on_ground.at[i].set(foot_on_ground)

                if backend == np:
                    if foot_on_ground:
                        air_time_reward += (tslt[i] - self._air_time_max)
                        tslt[i] = 0.0
                    else:
                        tslt[i] += env.dt
                else:
                    tslt_i, air_time_reward = jax.lax.cond(foot_on_ground,
                                                           lambda: (0.0, air_time_reward + tslt[i] - self._air_time_max),
                                                           lambda: (tslt[i] + env.dt, air_time_reward))
                    tslt = tslt.at[i].set(tslt_i)

            air_time_reward = self._air_time_coeff * air_time_reward
        else:
            tslt = reward_state.time_since_last_touchdown.copy()
            air_time_reward = 0.0

        # symmetry reward
        if self._symmetry_air_coeff > 0.0:
            symmetry_air_violations = 0.0
            if backend == np:
                if (not foots_on_ground[0] and not foots_on_ground[1]):
                    symmetry_air_violations += 1
                if not foots_on_ground[2] and not foots_on_ground[3]:
                    symmetry_air_violations += 1
            else:
                symmetry_air_violations = jax.lax.cond(jnp.logical_and(jnp.logical_not(foots_on_ground[0]),
                                                                       jnp.logical_not(foots_on_ground[1])),
                                                       lambda: symmetry_air_violations + 1,
                                                       lambda: symmetry_air_violations)

                symmetry_air_violations = jax.lax.cond(jnp.logical_and(jnp.logical_not(foots_on_ground[2]),
                                                                       jnp.logical_not(foots_on_ground[3])),
                                                       lambda: symmetry_air_violations + 1,
                                                       lambda: symmetry_air_violations)

            symmetry_air_reward = self._symmetry_air_coeff * -symmetry_air_violations
        else:
            symmetry_air_reward = 0.0

        # energy reward
        if self._energy_coeff > 0.0:
            energy = backend.sum(backend.abs(joint_vel) * backend.abs(data.qfrc_actuator[~self._free_joint_qvel_mask]))
            energy_reward = self._energy_coeff * -energy
        else:
            energy_reward = 0.0

        # total reward
        tracking_reward, _ = super().__call__(state, action, next_state, absorbing, info,
                                              env, model, data, carry, backend)
        penality_rewards = (z_vel_reward + roll_pitch_vel_reward + roll_pitch_reward + joint_qpos_reward
                            + joint_position_limit_reward + joint_vel_reward + acceleration_reward
                            + torque_reward + action_rate_reward + air_time_reward
                            + symmetry_air_reward + energy_reward)
        total_reward = tracking_reward + penality_rewards
        total_reward = backend.maximum(total_reward, 0.0)

        reward_components = {
            "tracking/main_goal": tracking_reward,
            "penalties/z_velocity": z_vel_reward,
            "penalties/roll_pitch_velocity": roll_pitch_vel_reward,
            "penalties/roll_pitch_position": roll_pitch_reward,
            "penalties/nominal_joint_position": joint_qpos_reward,
            "penalties/joint_position_limit": joint_position_limit_reward,
            "penalties/joint_velocity": joint_vel_reward,
            "penalties/joint_acceleration": acceleration_reward,
            "penalties/joint_torque": torque_reward,
            "penalties/action_rate": action_rate_reward,
            "penalties/air_time": air_time_reward,
            "penalties/gait_symmetry": symmetry_air_reward,
            "penalties/energy": energy_reward,
        }

        reward_state = reward_state.replace(
            last_qvel=data.qvel,
            last_action=action,
            time_since_last_touchdown=tslt,
            reward_components=reward_components
        )

        carry = carry.replace(reward_state=reward_state)

        return total_reward, carry


@struct.dataclass
class HumanoidLocomotionRewardState:
    """
    State of HumanoidLocomotionReward.
    """
    gait_process: float
    last_qvel: Union[np.ndarray, jax.Array]
    last_action: Union[np.ndarray, jax.Array]
    time_since_last_touchdown: Union[np.ndarray, jax.Array]
    last_left_step_length: Union[float, np.ndarray, jax.Array]
    last_right_step_length: Union[float, np.ndarray, jax.Array]
    left_step_seen: Union[bool, np.ndarray, jax.Array]
    right_step_seen: Union[bool, np.ndarray, jax.Array]
    feet_step_length_symmetry_error: Union[float, np.ndarray, jax.Array]
    feet_step_length_symmetry_valid: Union[bool, np.ndarray, jax.Array]
    leg_joint_symmetry_left_qpos_history: Union[np.ndarray, jax.Array]
    leg_joint_symmetry_right_qpos_history: Union[np.ndarray, jax.Array]
    leg_joint_symmetry_left_qvel_history: Union[np.ndarray, jax.Array]
    leg_joint_symmetry_right_qvel_history: Union[np.ndarray, jax.Array]
    leg_joint_symmetry_history_index: Union[int, np.ndarray, jax.Array]
    leg_joint_symmetry_history_count: Union[int, np.ndarray, jax.Array]
    leg_joint_cycle_symmetry_error: Union[float, np.ndarray, jax.Array]
    leg_joint_cycle_symmetry_valid: Union[bool, np.ndarray, jax.Array]
    reward_components: Dict[str, Union[np.ndarray, jax.Array]]


class HumanoidLocomotionReward(Reward):
    """
    Reward function extending the TargetVelocityGoalReward with typical additional penalties
    and regularization terms for locomotion. This reward is stateful: LocomotionRewardState
    """

    def __init__(self, env: Any, **kwargs):
        """
        Initialize the reward function.

        Args:
            env (Any): The environment instance.
            **kwargs (Any): Additional keyword arguments.
        """
        super().__init__(env, **kwargs)

        model = env._model
        self._free_jnt_name = self._info_props["root_free_joint_xml_name"]
        self._alternate_gait_start_phase = bool(
            env.env_cfg.get("alternate_gait_start_phase", False)
        )
        feet_stride_symmetry_frame_name = kwargs.get(
            "feet_stride_symmetry_frame", env.env_cfg.get("base_site", None)
        )
        self._feet_stride_symmetry_frame_site_id = (
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_SITE,
                str(feet_stride_symmetry_frame_name),
            )
            if feet_stride_symmetry_frame_name is not None
            else -1
        )
        if (
            feet_stride_symmetry_frame_name is not None
            and self._feet_stride_symmetry_frame_site_id < 0
        ):
            raise ValueError(
                "Missing feet stride symmetry frame site: "
                f"{feet_stride_symmetry_frame_name!r}."
            )

        # Initialize joint indices and masks
        self._free_joint_qpos_ind = np.array(mj_jntname2qposid(self._free_jnt_name, model))
        self._free_joint_qvel_ind = np.array(mj_jntname2qvelid(self._free_jnt_name, model))
        
        # self._free_joint_qpos_mask = np.zeros(model.nq, dtype=bool)
        # self._free_joint_qpos_mask[self._free_joint_qpos_ind] = True
        
        # self._free_joint_qvel_mask = np.zeros(model.nv, dtype=bool)
        # self._free_joint_qvel_mask[self._free_joint_qvel_ind] = True
        
        self.qpos_joint_adr = []
        self.qvel_joint_adr = []
        self.control_joint_ind = np.array(env.env_cfg.agent.agent_lower_body.action_idx)
        for jn in env.env_cfg.robot.joint_names:
            self.qpos_joint_adr.append(env.model.jnt_qposadr[env.model.joint(jn).id])
            self.qvel_joint_adr.append(env.model.jnt_dofadr[env.model.joint(jn).id])
        self.qpos_joint_adr = np.array(self.qpos_joint_adr)[self.control_joint_ind]
        self.qvel_joint_adr = np.array(self.qvel_joint_adr)[self.control_joint_ind]
        self._free_joint_qpos_mask = np.ones(model.nq, dtype=bool)
        self._free_joint_qvel_mask = np.ones(model.nv, dtype=bool)
        self._free_joint_qpos_mask[self.qpos_joint_adr] = False
        self._free_joint_qvel_mask[self.qvel_joint_adr] = False

        # Initialize floor and foot geometry IDs
        self._floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        foot_names = self._info_props["foot_geom_names"]
        
        # Get left and right foot names and IDs
        self._left_foot_names = [name for name in foot_names if "left" in name]
        self._right_foot_names = [name for name in foot_names if "right" in name]
        
        self._left_foot_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) 
            for name in self._left_foot_names
        ]
        self._right_foot_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) 
            for name in self._right_foot_names
        ]
        
        self._left_foot_body_ids = [model.geom_bodyid[foot_id] for foot_id in self._left_foot_ids]
        self._right_foot_body_ids = [model.geom_bodyid[foot_id] for foot_id in self._right_foot_ids]
        
        # Initialize foot sensor addresses
        # Adapted from: https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/locomotion/h1/joystick_gait_tracking.py
        foot_sensor_adrs = []
        for foot_sensor in ['left_foot_global_linvel', 'right_foot_global_linvel']:
            sensor_id = model.sensor(foot_sensor).id
            sensor_adr = model.sensor_adr[sensor_id]
            sensor_dim = model.sensor_dim[sensor_id]
            foot_sensor_adrs.append(list(range(sensor_adr, sensor_adr + sensor_dim)))
        
        self._left_foot_sensor_adr = np.array(foot_sensor_adrs[0])
        self._right_foot_sensor_adr = np.array(foot_sensor_adrs[1])

        # Initialize foot site IDs. H1 assets use the *_mimic names, while
        # other humanoid assets may expose the shorter legacy names.
        def resolve_foot_site(primary_name, fallback_name):
            site_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SITE, primary_name
            )
            if site_id < 0:
                site_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_SITE, fallback_name
                )
            if site_id < 0:
                raise ValueError(
                    f"Missing foot site: expected {primary_name!r} or "
                    f"{fallback_name!r}."
                )
            return site_id

        self._left_foot_site_id = resolve_foot_site(
            "left_foot", "left_foot_mimic"
        )
        self._right_foot_site_id = resolve_foot_site(
            "right_foot", "right_foot_mimic"
        )

        # Extract reward coefficients from kwargs
        self._survival = kwargs.get("survival", 0.0)

        # Velocity tracking weights and coefficients
        self._tracking_w_exp_linvel_x = kwargs.get("tracking_w_exp_linvel_x", 0.0)
        self._tracking_w_sum_linvel_x = kwargs.get("tracking_w_sum_linvel_x", 0.0)
        self._tracking_w_exp_linvel_y = kwargs.get("tracking_w_exp_linvel_y", 0.0)
        self._tracking_w_sum_linvel_y = kwargs.get("tracking_w_sum_linvel_y", 0.0)
        self._tracking_w_exp_angvel = kwargs.get("tracking_w_exp_angvel", 0.0)
        self._tracking_w_sum_angvel = kwargs.get("tracking_w_sum_angvel", 0.0)
        self._tracking_world_direction_coeff = kwargs.get("tracking_world_direction_coeff", 0.0)
        self._tracking_world_direction_vel_exp = kwargs.get("tracking_world_direction_vel_exp", 4.0)
        self._tracking_world_direction_pos_exp = kwargs.get("tracking_world_direction_pos_exp", 4.0)
        self._lateral_velocity_penalty_coeff = kwargs.get(
            "lateral_velocity_penalty_coeff", 0.0
        )
        self._yaw_velocity_penalty_coeff = kwargs.get(
            "yaw_velocity_penalty_coeff", 0.0
        )
        self._cross_track_penalty_coeff = kwargs.get(
            "cross_track_penalty_coeff", 0.0
        )
        self._transition_direction_constraints_enabled = bool(kwargs.get(
            "transition_direction_constraints_enabled", False
        ))
        self._reset_to_stand_constraints_enabled = bool(kwargs.get(
            "reset_to_stand_constraints_enabled", False
        ))
        self._stand_to_walk_transition_steps = int(kwargs.get(
            "stand_to_walk_transition_steps", 0
        ))
        self._walk_to_stand_transition_steps = int(kwargs.get(
            "walk_to_stand_transition_steps", 0
        ))
        self._transition_lateral_velocity_scale = float(kwargs.get(
            "transition_lateral_velocity_scale", 1.0
        ))
        self._transition_yaw_velocity_scale = float(kwargs.get(
            "transition_yaw_velocity_scale", 1.0
        ))
        self._transition_cross_track_scale = float(kwargs.get(
            "transition_cross_track_scale", 1.0
        ))
        self._transition_yaw_deviation_scale = float(kwargs.get(
            "transition_yaw_deviation_scale", 1.0
        ))

        # Nominal posture tracking weights and coefficients
        self._nominal_joint_pos_exp = kwargs.get("tracking_nominal_joint_pos_exp", 0.0)
        self._nominal_joint_pos_coeff = kwargs.get("tracking_nominal_joint_pos_coeff", 0.0)
        self._nominal_joint_pos_names = kwargs.get("tracking_nominal_joint_pos_names", None)

        self._joint_deviation_l1_coeff = kwargs.get("joint_deviation_l1_coeff", 0.0)   
        self._base_height_coeff = kwargs.get("base_height_coeff", 0.0)
        # self._base_height_target = kwargs.get("base_height_target", 0.0)
        self.orientation_coeff = kwargs.get("orientation_coeff", 0.0)

        # Torque and energy coefficients
        self._joint_torque_coeff = kwargs.get("joint_torque_coeff", 0.0)
        self._energy_coeff = kwargs.get("energy_coeff", 0.0)

        # Velocity and acceleration penalties
        self._z_vel_coeff = kwargs.get("z_vel_coeff", 0.0)
        self._roll_pitch_vel_coeff = kwargs.get("roll_pitch_vel_coeff", 0.0)
        self._joint_vel_coeff = kwargs.get("joint_vel_coeff", 0.0)
        self._joint_acc_coeff = kwargs.get("joint_acc_coeff", 0.0)
        self._root_acc_coeff = kwargs.get("root_acc_coeff", 0.0)
        self._action_rate_coeff = kwargs.get("action_rate_coeff", 0.0)

        # Joint position limit coefficients
        self._joint_position_limit_scale = kwargs.get("joint_position_limit_scale", 1.0)
        self._joint_position_limit_coeff = kwargs.get("joint_position_limit_coeff", 0.0)

        # Feet-related coefficients
        self._feet_slip_coeff = kwargs.get("feet_slip_coeff", 0.0)
        self._feet_friction_utilization_coeff = kwargs.get(
            "feet_friction_utilization_coeff", 0.0
        )
        self._feet_friction_utilization_threshold = float(
            kwargs.get("feet_friction_utilization_threshold", 0.35)
        )
        self._feet_friction_min_normal_force = float(
            kwargs.get("feet_friction_min_normal_force", 25.0)
        )
        self._feet_yaw_diff_coeff = kwargs.get("feet_yaw_diff_coeff", 0.0)
        self._feet_yaw_mean_coeff = kwargs.get("feet_yaw_mean_coeff", 0.0)
        self._feet_roll_coeff = kwargs.get("feet_roll_coeff", 0.0)
        self._feet_distance_coeff = kwargs.get("feet_distance_coeff", 0.0)
        self._feet_distance_target = kwargs.get("feet_distance_target", 0.0)
        self._feet_distance_mode = kwargs.get("feet_distance_mode", "target")
        if self._feet_distance_mode not in {"target", "minimum_lateral"}:
            raise ValueError(
                "feet_distance_mode must be 'target' or 'minimum_lateral', got "
                f"{self._feet_distance_mode!r}."
            )
        self._feet_swing_coeff = kwargs.get("feet_swing_coeff", 0.0)
        self._feet_swing_period = kwargs.get("feet_swing_period", 0.2)
        self._feet_swing_height = kwargs.get("feet_swing_height", 0.08)
        self._feet_swing_height_coeff = kwargs.get("feet_swing_height_coeff", 0.0)
        self._feet_stride_symmetry_coeff = kwargs.get(
            "feet_stride_symmetry_coeff", 0.0
        )
        self._feet_step_length_symmetry_coeff = kwargs.get(
            "feet_step_length_symmetry_coeff", 0.0
        )
        self._leg_joint_cycle_symmetry_coeff = kwargs.get(
            "leg_joint_cycle_symmetry_coeff", 0.0
        )
        self._leg_joint_cycle_symmetry_velocity_scale = float(
            kwargs.get("leg_joint_cycle_symmetry_velocity_scale", 0.02)
        )
        symmetry_coeff_values = np.asarray(
            self._leg_joint_cycle_symmetry_coeff, dtype=np.float32
        ).reshape(-1)
        leg_joint_cycle_symmetry_enabled = np.any(
            np.abs(symmetry_coeff_values) > 0.0
        )
        if leg_joint_cycle_symmetry_enabled:
            gait_frequency_range = np.asarray(
                env.env_cfg.goal.params.gait_frequency_range,
                dtype=np.float32,
            ).reshape(-1)
            min_gait_frequency = float(np.min(gait_frequency_range))
            if min_gait_frequency <= 0.0:
                raise ValueError(
                    "A positive minimum goal.params.gait_frequency_range is "
                    "required when leg_joint_cycle_symmetry_coeff is enabled"
                )
            max_half_cycle_steps = int(np.ceil(
                0.5 / (float(env.dt) * min_gait_frequency)
            ))
            self._leg_joint_cycle_symmetry_history_steps = (
                max_half_cycle_steps + 1
            )
        else:
            self._leg_joint_cycle_symmetry_history_steps = 2
        left_leg_symmetry_names = kwargs.get(
            "leg_joint_cycle_symmetry_left_names",
            [
                "left_hip_yaw", "left_hip_roll", "left_hip_pitch",
                "left_knee", "left_ankle",
            ],
        )
        right_leg_symmetry_names = kwargs.get(
            "leg_joint_cycle_symmetry_right_names",
            [
                "right_hip_yaw", "right_hip_roll", "right_hip_pitch",
                "right_knee", "right_ankle",
            ],
        )
        if len(left_leg_symmetry_names) != len(right_leg_symmetry_names):
            raise ValueError(
                "Left and right leg joint symmetry name lists must have equal length"
            )
        self._leg_joint_symmetry_left_qpos_id = np.concatenate([
            mj_jntname2qposid(name, model) for name in left_leg_symmetry_names
        ])
        self._leg_joint_symmetry_right_qpos_id = np.concatenate([
            mj_jntname2qposid(name, model) for name in right_leg_symmetry_names
        ])
        self._leg_joint_symmetry_left_qvel_id = np.concatenate([
            mj_jntname2qvelid(name, model) for name in left_leg_symmetry_names
        ])
        self._leg_joint_symmetry_right_qvel_id = np.concatenate([
            mj_jntname2qvelid(name, model) for name in right_leg_symmetry_names
        ])
        mirror_signs = np.asarray(
            kwargs.get(
                "leg_joint_cycle_symmetry_mirror_signs",
                [-1.0, -1.0, 1.0, 1.0, 1.0],
            ),
            dtype=np.float32,
        )
        if mirror_signs.shape != (len(left_leg_symmetry_names),):
            raise ValueError(
                "leg_joint_cycle_symmetry_mirror_signs must match the configured "
                "number of joints per leg"
            )
        self._leg_joint_cycle_symmetry_mirror_signs = mirror_signs
        self._standing_leg_joint_symmetry_coeff = kwargs.get(
            "standing_leg_joint_symmetry_coeff", 0.0
        )
        self._torso_joint_zero_coeff = kwargs.get(
            "torso_joint_zero_coeff", 0.0
        )
        torso_joint_zero_names = kwargs.get(
            "torso_joint_zero_names", ["torso"]
        )
        self._torso_joint_zero_qpos_id = np.concatenate([
            mj_jntname2qposid(name, model) for name in torso_joint_zero_names
        ])
        self._stand_command_threshold = kwargs.get("stand_command_threshold", 1.0e-6)
        self._hip_pos_coeff = kwargs.get("hip_pos_coeff", 0.0)
        hip_pos_names = kwargs.get("hip_pos_names", [])
        self._hip_pos_qpos_id = (
            np.concatenate([mj_jntname2qposid(name, model) for name in hip_pos_names])
            if hip_pos_names
            else np.array([], dtype=np.int32)
        )

        # Air time and impact coefficients
        self._air_time_max = kwargs.get("air_time_max", 0.0)
        self._air_time_coeff = kwargs.get("air_time_coeff", 0.0)
        self._no_fly_coeff = kwargs.get("no_fly_coeff", 0.0)
        self._symmetry_air_coeff = kwargs.get("symmetry_air_coeff", 0.0)
        self._impact_threshold = kwargs.get("impact_threshold", 0.0)
        self._impact_coeff = kwargs.get("impact_coeff", 0.0)

        self._tracking_base_pos_coeff = kwargs.get("tracking_base_pos_coeff", 0.0)
        self._tracking_base_pos_exp = kwargs.get("tracking_base_pos_exp", 4.0)
        self._yaw_deviation_coeff = kwargs.get("yaw_deviation_coeff", 0.0)
        self._only_positive_rewards = kwargs.get("only_positive_rewards", False)

        # Initialize joint limits and nominal positions
        self._limited_joints = np.array(model.jnt_limited, dtype=bool)
        self._limited_joints_qpos_id = model.jnt_qposadr[np.where(self._limited_joints)]
        self._joint_ranges = model.jnt_range[self._limited_joints]
        self._nominal_joint_qpos = env._init_state_handler.qpos_init
        
        if self._nominal_joint_pos_names is None:
            # Take all limited joints
            self._nominal_joint_qpos_id = self._limited_joints_qpos_id
        else:
            self._nominal_joint_qpos_id = np.concatenate([
                mj_jntname2qposid(name, model) for name in self._nominal_joint_pos_names
            ])

        # Initialize COM position and stand still
        self._stand_no_step_coeff = kwargs.get("stand_no_step_coeff", 0.0)
        self._root_centering_coeff = kwargs.get("root_centering_coeff", 0.0)
        self._root_centering_target_x = kwargs.get("root_centering_target_x", 0.0)
        self._root_centering_target_y = kwargs.get("root_centering_target_y", 0.0)

    def init_state(self, env: Any, key: Any, model: Union[MjModel, Model], 
                   data: Union[MjData, Data], backend: ModuleType):
        """
        Initialize the reward state.

        Args:
            env (Any): The environment instance.
            key (Any): Key for the reward state.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            backend (ModuleType): Backend module used for computation (either numpy or jax.numpy).

        Returns:
            LocomotionRewardState: The initialized reward state.
        """
        reward_components = {
            "survival": 0.,
            "tracking_linvel_x": 0.,
            "tracking_linvel_y": 0.,
            "tracking_angvel": 0.,
            "tracking_world_direction": 0.,
            "lateral_velocity_penalty": 0.,
            "yaw_velocity_penalty": 0.,
            "cross_track_penalty": 0.,
            "tracking_joint_qpos": 0.,
            "tracking_feet_swing": 0.,
            "feet_swing_height": 0.,
            "feet_stride_symmetry": 0.,
            "feet_step_length_symmetry": 0.,
            "leg_joint_cycle_symmetry": 0.,
            "standing_leg_joint_symmetry": 0.,
            "torso_joint_zero": 0.,
            "hip_pos": 0.,
            "joint_deviation_l1": 0.,
            "base_height": 0.,
            "orientation": 0.,
            "torque": 0.,
            "energy": 0.,
            "z_vel": 0.,
            "roll_pitch_vel": 0.,
            "joint_vel": 0.,
            "acceleration": 0.,
            "root_acceleration": 0.,
            "action_rate": 0.,
            "joint_position_limit": 0.,
            "feet_slip": 0.,
            "feet_friction_utilization": 0.,
            "feet_yaw_diff": 0.,
            "feet_yaw_mean": 0.,
            "feet_roll": 0.,
            "feet_distance": 0.,
            "air_time": 0.,
            "no_fly": 0.,
            "impact": 0.,
            "tracking_base_pos": 0.,
            "yaw_deviation": 0.,
            "stand_no_step": 0.,
            "root_centering": 0.,
        }

        return HumanoidLocomotionRewardState(
            gait_process=0.0,
            last_qvel=data.qvel, 
            last_action=backend.zeros(env.info.action_space.shape[0]),
            time_since_last_touchdown=backend.zeros(2, dtype=backend.float32),
            last_left_step_length=backend.array(0.0, dtype=backend.float32),
            last_right_step_length=backend.array(0.0, dtype=backend.float32),
            left_step_seen=backend.array(False, dtype=bool),
            right_step_seen=backend.array(False, dtype=bool),
            feet_step_length_symmetry_error=backend.array(0.0, dtype=backend.float32),
            feet_step_length_symmetry_valid=backend.array(False, dtype=bool),
            leg_joint_symmetry_left_qpos_history=backend.zeros(
                (
                    self._leg_joint_cycle_symmetry_history_steps,
                    len(self._leg_joint_symmetry_left_qpos_id),
                ),
                dtype=backend.float32,
            ),
            leg_joint_symmetry_right_qpos_history=backend.zeros(
                (
                    self._leg_joint_cycle_symmetry_history_steps,
                    len(self._leg_joint_symmetry_right_qpos_id),
                ),
                dtype=backend.float32,
            ),
            leg_joint_symmetry_left_qvel_history=backend.zeros(
                (
                    self._leg_joint_cycle_symmetry_history_steps,
                    len(self._leg_joint_symmetry_left_qvel_id),
                ),
                dtype=backend.float32,
            ),
            leg_joint_symmetry_right_qvel_history=backend.zeros(
                (
                    self._leg_joint_cycle_symmetry_history_steps,
                    len(self._leg_joint_symmetry_right_qvel_id),
                ),
                dtype=backend.float32,
            ),
            leg_joint_symmetry_history_index=backend.array(0, dtype=backend.int32),
            leg_joint_symmetry_history_count=backend.array(0, dtype=backend.int32),
            leg_joint_cycle_symmetry_error=backend.array(0.0, dtype=backend.float32),
            leg_joint_cycle_symmetry_valid=backend.array(False, dtype=bool),
            reward_components=reward_components
        )

    def _feet_positions_local(self, data, backend):
        R = np_R if backend == np else jnp_R
        if self._feet_stride_symmetry_frame_site_id >= 0:
            global_pos_root = data.site_xpos[
                self._feet_stride_symmetry_frame_site_id
            ]
            global_rot = R.from_matrix(
                data.site_xmat[
                    self._feet_stride_symmetry_frame_site_id
                ].reshape(3, 3)
            )
        else:
            global_pose_root = data.qpos[self._free_joint_qpos_ind]
            global_pos_root = global_pose_root[:3]
            global_rot = R.from_quat(
                quat_scalarfirst2scalarlast(global_pose_root[3:])
            )
        left_foot_pos_local = global_rot.inv().apply(
            data.site_xpos[self._left_foot_site_id] - global_pos_root
        )
        right_foot_pos_local = global_rot.inv().apply(
            data.site_xpos[self._right_foot_site_id] - global_pos_root
        )
        return backend.stack([left_foot_pos_local, right_foot_pos_local])

    def _initial_gait_phase(self, carry, backend):
        if not self._alternate_gait_start_phase or carry.env_id is None:
            return backend.array(0.0)
        return 0.5 * backend.mod(backend.asarray(carry.env_id), 2)

    def _update_leg_joint_cycle_symmetry(
        self, reward_state, data, gait_frequency, straight_command, env, backend
    ):
        active = straight_command & (gait_frequency > 1.0e-8)
        history_size = self._leg_joint_cycle_symmetry_history_steps
        history_index = reward_state.leg_joint_symmetry_history_index
        history_count = reward_state.leg_joint_symmetry_history_count
        half_cycle_steps = backend.rint(
            0.5 / backend.maximum(env.dt * gait_frequency, 1.0e-8)
        ).astype(backend.int32)
        half_cycle_steps = backend.clip(half_cycle_steps, 1, history_size - 1)
        delayed_index = backend.mod(history_index - half_cycle_steps, history_size)

        left_qpos = (
            data.qpos[self._leg_joint_symmetry_left_qpos_id]
            - self._nominal_joint_qpos[self._leg_joint_symmetry_left_qpos_id]
        )
        right_qpos = (
            data.qpos[self._leg_joint_symmetry_right_qpos_id]
            - self._nominal_joint_qpos[self._leg_joint_symmetry_right_qpos_id]
        )
        left_qvel = data.qvel[self._leg_joint_symmetry_left_qvel_id]
        right_qvel = data.qvel[self._leg_joint_symmetry_right_qvel_id]
        signs = backend.asarray(self._leg_joint_cycle_symmetry_mirror_signs)

        delayed_left_qpos = reward_state.leg_joint_symmetry_left_qpos_history[
            delayed_index
        ]
        delayed_right_qpos = reward_state.leg_joint_symmetry_right_qpos_history[
            delayed_index
        ]
        delayed_left_qvel = reward_state.leg_joint_symmetry_left_qvel_history[
            delayed_index
        ]
        delayed_right_qvel = reward_state.leg_joint_symmetry_right_qvel_history[
            delayed_index
        ]
        left_qpos_error = left_qpos - signs * delayed_right_qpos
        right_qpos_error = right_qpos - signs * delayed_left_qpos
        left_qvel_error = left_qvel - signs * delayed_right_qvel
        right_qvel_error = right_qvel - signs * delayed_left_qvel
        qpos_mse = 0.5 * (
            backend.mean(backend.square(left_qpos_error))
            + backend.mean(backend.square(right_qpos_error))
        )
        qvel_mse = 0.5 * (
            backend.mean(backend.square(left_qvel_error))
            + backend.mean(backend.square(right_qvel_error))
        )
        valid = active & (history_count >= half_cycle_steps)
        penalty = (
            qpos_mse
            + self._leg_joint_cycle_symmetry_velocity_scale * qvel_mse
        ) * valid.astype(backend.float32)
        error = backend.sqrt(qpos_mse) * valid.astype(backend.float32)

        def write_history(history, value):
            if backend == np:
                updated = history.copy()
                updated[int(history_index)] = value
                return updated
            return history.at[history_index].set(value)

        left_qpos_history = backend.where(
            active,
            write_history(
                reward_state.leg_joint_symmetry_left_qpos_history, left_qpos
            ),
            reward_state.leg_joint_symmetry_left_qpos_history,
        )
        right_qpos_history = backend.where(
            active,
            write_history(
                reward_state.leg_joint_symmetry_right_qpos_history, right_qpos
            ),
            reward_state.leg_joint_symmetry_right_qpos_history,
        )
        left_qvel_history = backend.where(
            active,
            write_history(
                reward_state.leg_joint_symmetry_left_qvel_history, left_qvel
            ),
            reward_state.leg_joint_symmetry_left_qvel_history,
        )
        right_qvel_history = backend.where(
            active,
            write_history(
                reward_state.leg_joint_symmetry_right_qvel_history, right_qvel
            ),
            reward_state.leg_joint_symmetry_right_qvel_history,
        )
        next_history_index = backend.where(
            active, backend.mod(history_index + 1, history_size), 0
        )
        next_history_count = backend.where(
            active, backend.minimum(history_count + 1, history_size), 0
        )
        return (
            penalty,
            error,
            valid,
            left_qpos_history,
            right_qpos_history,
            left_qvel_history,
            right_qvel_history,
            next_history_index,
            next_history_count,
        )

    @staticmethod
    def _update_step_length_symmetry(
        reward_state, feet_on_ground, feet_pos_local, straight_command, backend
    ):
        left_touchdown = feet_on_ground[0] & (
            reward_state.time_since_last_touchdown[0] > 1.0e-6
        )
        right_touchdown = feet_on_ground[1] & (
            reward_state.time_since_last_touchdown[1] > 1.0e-6
        )
        current_step_length = backend.abs(
            feet_pos_local[0, 0] - feet_pos_local[1, 0]
        )
        last_left_step_length = backend.where(
            left_touchdown,
            current_step_length,
            reward_state.last_left_step_length,
        )
        last_right_step_length = backend.where(
            right_touchdown,
            current_step_length,
            reward_state.last_right_step_length,
        )
        left_step_seen = reward_state.left_step_seen | left_touchdown
        right_step_seen = reward_state.right_step_seen | right_touchdown
        valid = left_step_seen & right_step_seen & straight_command
        error = backend.abs(last_left_step_length - last_right_step_length)
        penalty = backend.square(error) * valid.astype(backend.float32)
        return (
            penalty,
            error,
            valid,
            last_left_step_length,
            last_right_step_length,
            left_step_seen,
            right_step_seen,
        )

    def reset(self, env: Any, model: Union[MjModel, Model], data: Union[MjData, Data], 
              carry: Any, backend: ModuleType):
        """
        Reset the reward state.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Additional carry.
            backend (ModuleType): Backend module used for computation (either numpy or jax.numpy).

        Returns:
            Tuple[Union[MjData, Data], Any]: The updated data and carry.
        """
        reward_state = self.init_state(env, None, model, data, backend).replace(
            gait_process=self._initial_gait_phase(carry, backend)
        )
        carry = carry.replace(reward_state=reward_state)
        return data, carry

    def __call__(self, state: Union[np.ndarray, jnp.ndarray], action: Union[np.ndarray, jnp.ndarray],
                 next_state: Union[np.ndarray, jnp.ndarray], absorbing: bool, info: Dict[str, Any],
                 env: Any, model: Union[MjModel, Model], data: Union[MjData, Data], 
                 carry: Any, backend: ModuleType) -> Tuple[float, Any]:
        """
        Based on the tracking reward, this reward function adds typical penalties and regularization terms
        for locomotion.

        Args:
            state (Union[np.ndarray, jnp.ndarray]): Last state.
            action (Union[np.ndarray, jnp.ndarray]): Applied action.
            next_state (Union[np.ndarray, jnp.ndarray]): Current state.
            absorbing (bool): Whether the state is absorbing.
            info (Dict[str, Any]): Additional information.
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Additional carry.
            backend (ModuleType): Backend module used for computation (either numpy or jax.numpy).

        Returns:
            Tuple[float, Any]: The reward for the current transition and the updated carry.
        """
        # Select rotation backend
        if backend == np:
            R = np_R
        else:
            R = jnp_R

        # Get current states
        reward_state = carry.reward_state
        goal_state = getattr(carry.observation_states, "GoalPatternLoco")  # getattr(carry.observation_states, "GoalRandomRootVelocityAndPhase")

        # Extract global pose and velocity information
        global_pose_root = data.qpos[self._free_joint_qpos_ind]
        global_pos_root = global_pose_root[:3]
        global_quat_root = global_pose_root[3:]
        global_rot = R.from_quat(quat_scalarfirst2scalarlast(global_quat_root))
        global_vel_root = data.qvel[self._free_joint_qvel_ind]

        # Transform to local coordinates
        local_vel_root_lin = global_rot.inv().apply(global_vel_root[:3])
        local_vel_root_ang = global_rot.inv().apply(global_vel_root[3:])
        global_vel_root_ang = global_vel_root[3:]

        # ==================== REWARD COMPONENTS ====================
        
        # Survival reward
        survival_reward = 1.0

        # Goal tracking rewards
        sampled_goal_vel = backend.array([
            goal_state.goal_vel_x,
            goal_state.goal_vel_y,
            goal_state.goal_vel_yaw,
        ])
        goal_vel = sampled_goal_vel
        locomotion_timed_stand = (
            bool(env.env_cfg.get("locomotion_only", False))
            and (
                bool(getattr(env, "stand_phase_enabled", False))
                or bool(getattr(env, "walk_to_stand_enabled", False))
            )
        )
        if locomotion_timed_stand:
            # A transition observation is consumed by the next action, so the
            # reward at each command boundary still follows the previous obs.
            in_command_stand = env._is_locomotion_reward_stand_phase(carry)
            goal_vel = backend.where(
                in_command_stand,
                backend.zeros_like(goal_vel),
                goal_vel,
            )
        
        tracking_reward_linvel_x = backend.exp(
            -backend.square(local_vel_root_lin[0] - goal_vel[0]) * self._tracking_w_exp_linvel_x
        )
        tracking_reward_linvel_y = backend.exp(
            -backend.square(local_vel_root_lin[1] - goal_vel[1]) * self._tracking_w_exp_linvel_y
        )
        tracking_reward_angvel = backend.exp(
            -backend.square(local_vel_root_ang[2] - goal_vel[2]) * self._tracking_w_exp_angvel
        )

        # Convert the reset-time body-frame command into a fixed world-frame
        # direction. The path starts at reset for direct walking and at the
        # command transition position for stand-to-walk episodes.
        command_speed_xy = backend.linalg.norm(goal_vel[:2])
        sampled_command_speed_xy = backend.linalg.norm(sampled_goal_vel[:2])
        moving_command = command_speed_xy >= self._stand_command_threshold
        pure_forward_command = (
            backend.abs(goal_vel[0]) >= self._stand_command_threshold
        ) & (
            backend.abs(goal_vel[1]) < self._stand_command_threshold
        ) & (
            backend.abs(goal_vel[2]) < self._stand_command_threshold
        )
        forward_tracking_gate = backend.where(
            pure_forward_command,
            tracking_reward_linvel_x,
            backend.asarray(1.0, dtype=tracking_reward_linvel_x.dtype),
        )
        tracking_reward_linvel_y *= forward_tracking_gate
        tracking_reward_angvel *= forward_tracking_gate
        safe_command_speed = backend.maximum(sampled_command_speed_xy, 1.0e-6)
        normalized_command_direction_local = backend.array([
            sampled_goal_vel[0] / safe_command_speed,
            sampled_goal_vel[1] / safe_command_speed,
            0.0,
        ])
        command_direction_local = backend.where(
            sampled_command_speed_xy >= self._stand_command_threshold,
            normalized_command_direction_local,
            backend.array([1.0, 0.0, 0.0]),
        )
        initial_rot = R.from_matrix(carry.base_site_rot0)
        command_direction_world = initial_rot.apply(command_direction_local)[:2]
        command_normal_world = backend.array([
            -command_direction_world[1], command_direction_world[0]
        ])
        lateral_velocity = backend.sum(global_vel_root[:2] * command_normal_world)
        direction_origin = getattr(
            carry, "locomotion_direction_origin", carry.base_site_pos0
        )
        displacement_from_start = global_pos_root[:2] - direction_origin[:2]
        cross_track_error = backend.sum(displacement_from_start * command_normal_world)
        tracking_world_direction_alignment = backend.exp(
            -self._tracking_world_direction_vel_exp * backend.square(lateral_velocity)
            -self._tracking_world_direction_pos_exp * backend.square(cross_track_error)
        )
        tracking_xy_velocity_reward = backend.exp(
            -self._tracking_w_exp_linvel_x
            * backend.square(local_vel_root_lin[0] - goal_vel[0])
            -self._tracking_w_exp_linvel_y
            * backend.square(local_vel_root_lin[1] - goal_vel[1])
        )
        tracking_world_direction_reward = (
            tracking_world_direction_alignment
            * tracking_xy_velocity_reward
            * moving_command.astype(backend.float32)
        )
        moving_command_float = moving_command.astype(backend.float32)
        transition_active = backend.array(False)
        if (
            self._transition_direction_constraints_enabled
            and bool(env.env_cfg.get("locomotion_only", False))
        ):
            stand_end_step = getattr(carry, "stand_end_step", backend.array(0))
            sampled_command_still = (
                sampled_command_speed_xy < self._stand_command_threshold
            ) & (
                backend.abs(sampled_goal_vel[2])
                < self._stand_command_threshold
            )
            reset_to_stand_active = backend.logical_and(
                self._reset_to_stand_constraints_enabled,
                backend.logical_or(
                    (
                        (stand_end_step > 0)
                        & (carry.cur_step_in_episode <= stand_end_step)
                    ),
                    sampled_command_still,
                ),
            )
            stand_to_walk_elapsed = carry.cur_step_in_episode - stand_end_step
            stand_to_walk_active = (
                (stand_end_step > 0)
                & (stand_to_walk_elapsed > 0)
                & (
                    stand_to_walk_elapsed
                    <= self._stand_to_walk_transition_steps
                )
                & (sampled_command_speed_xy >= self._stand_command_threshold)
            )
            walk_to_stand_step = getattr(
                carry, "walk_to_stand_step", backend.array(0)
            )
            walk_to_stand_elapsed = (
                carry.cur_step_in_episode - walk_to_stand_step
            )
            walk_to_stand_active = (
                (walk_to_stand_step > 0)
                & (walk_to_stand_elapsed > 0)
                & (
                    walk_to_stand_elapsed
                    <= self._walk_to_stand_transition_steps
                )
            )
            transition_active = backend.logical_or(
                reset_to_stand_active,
                backend.logical_or(
                    stand_to_walk_active,
                    walk_to_stand_active,
                ),
            )
        lateral_velocity_gate = backend.where(
            transition_active,
            backend.asarray(self._transition_lateral_velocity_scale),
            moving_command_float,
        )
        yaw_velocity_gate = backend.where(
            transition_active,
            backend.asarray(self._transition_yaw_velocity_scale),
            moving_command_float,
        )
        cross_track_gate = backend.where(
            transition_active,
            backend.asarray(self._transition_cross_track_scale),
            moving_command_float,
        )
        lateral_velocity_penalty = (
            backend.square(lateral_velocity) * lateral_velocity_gate
        )
        yaw_velocity_penalty = (
            backend.square(local_vel_root_ang[2]) * yaw_velocity_gate
        )
        cross_track_penalty = (
            backend.square(cross_track_error) * cross_track_gate
        )

        # Base height reward
        base_height_target = goal_state.goal_height
        base_height = global_pos_root[2] - 0  # Assuming flat ground at z=0
        base_height_reward = backend.square(base_height - base_height_target)

        # Base position tracking reward
        goal_pos_xyz = backend.array([goal_state.goal_pos_x, goal_state.goal_pos_y, goal_state.goal_height])
        base_pos_xyz = global_pos_root - carry.env_offset
        dist = backend.linalg.norm(base_pos_xyz - goal_pos_xyz)
        tracking_base_pos_reward = backend.exp(-self._tracking_base_pos_exp * dist ** 2)

        # Orientation reward
        projected_gravity = global_rot.inv().apply(backend.array([0, 0, -1]))
        orientation_reward = backend.sum(backend.square(projected_gravity[:2]))  # Penalize deviation from vertical

        # Joint torque reward
        torque_reward = backend.sum(backend.square(data.qfrc_actuator[~self._free_joint_qvel_mask]))

        # Torque tiredness reward
        torques = data.qfrc_actuator[~self._free_joint_qvel_mask]

        # Energy reward
        energy_reward = backend.sum(backend.clip(
            data.qvel[~self._free_joint_qvel_mask] * data.qfrc_actuator[~self._free_joint_qvel_mask], 
            a_min=0.0
        ))

        # Velocity penalties
        z_vel_reward = backend.square(local_vel_root_lin[2])
        roll_pitch_vel_reward = backend.square(local_vel_root_ang[:2]).sum()

        # Joint motion penalties
        joint_vel = data.qvel[~self._free_joint_qvel_mask]
        joint_vel_reward = backend.square(joint_vel).sum()

        last_joint_vel = reward_state.last_qvel[~self._free_joint_qvel_mask]
        acceleration_reward = (backend.square((joint_vel - last_joint_vel) / env.dt)).sum()

        # Root acceleration penalty
        root_acceleration_reward = backend.square(
            (global_vel_root - reward_state.last_qvel[self._free_joint_qvel_ind]) / env.dt
        ).sum()

        # Action rate penalty
        action_rate_reward = (backend.square(action[self.control_joint_ind] - reward_state.last_action[self.control_joint_ind])).sum()

        # Joint position limit penalty
        joint_positions = backend.array(data.qpos[self._limited_joints_qpos_id])
        scale_factor = 0.5 * (1 - self._joint_position_limit_scale)
        range_diff = self._joint_ranges[:, 1] - self._joint_ranges[:, 0]
        
        lower = self._joint_ranges[:, 0] + scale_factor * range_diff
        upper = self._joint_ranges[:, 1] - scale_factor * range_diff
        joint_position_limit_reward = ((joint_positions < lower) + (joint_positions > upper)).sum() * 1.0

        # ==================== FEET-RELATED REWARDS ====================
        
        def get_feet_contact_states():
            """Check if the foot is in contact with the floor."""
            left_contacts = [
                mj_check_collisions(f_id, self._floor_id, data, backend) 
                for f_id in self._left_foot_ids
            ]
            right_contacts = [
                mj_check_collisions(f_id, self._floor_id, data, backend) 
                for f_id in self._right_foot_ids
            ]
            
            if backend == np:
                left_foot_on_ground = any(left_contacts)
                right_foot_on_ground = any(right_contacts)
                foots_on_ground = np.array([left_foot_on_ground, right_foot_on_ground])
            else:
                # JAX-compatible version
                left_foot_on_ground = (
                    jnp.logical_or.reduce(jnp.array(left_contacts)) if left_contacts 
                    else jnp.array(False)
                )
                right_foot_on_ground = (
                    jnp.logical_or.reduce(jnp.array(right_contacts)) if right_contacts 
                    else jnp.array(False)
                )
                foots_on_ground = jnp.array([left_foot_on_ground, right_foot_on_ground])
            
            return foots_on_ground

        # Feet slip reward
        left_foot_body_id = self._left_foot_body_ids[0]
        right_foot_body_id = self._right_foot_body_ids[0]
        
        left_foot_vel = data.sensordata[self._left_foot_sensor_adr]
        right_foot_vel = data.sensordata[self._right_foot_sensor_adr]
        feet_on_ground = get_feet_contact_states()
        
        feet_slip_reward = (
            backend.square(left_foot_vel[:3] * feet_on_ground[0]) + 
            backend.square(right_foot_vel[:3] * feet_on_ground[1])
        ).sum()

        def friction_utilization_penalty(foot_body_id, foot_on_ground):
            contact_force = data.cfrc_ext[foot_body_id, 3:6]
            normal_force = backend.abs(contact_force[2])
            tangential_force = backend.linalg.norm(contact_force[:2])
            valid_contact = foot_on_ground & (
                normal_force >= self._feet_friction_min_normal_force
            )
            utilization = tangential_force / backend.maximum(
                normal_force,
                self._feet_friction_min_normal_force,
            )
            excess = backend.maximum(
                utilization - self._feet_friction_utilization_threshold,
                backend.array(0.0),
            )
            return backend.square(excess) * valid_contact.astype(
                backend.float32
            )

        feet_friction_utilization_reward = (
            friction_utilization_penalty(
                left_foot_body_id, feet_on_ground[0]
            )
            + friction_utilization_penalty(
                right_foot_body_id, feet_on_ground[1]
            )
        )

        # Feet yaw difference reward
        left_foot_yaw = R.from_matrix(data.site_xmat[self._left_foot_site_id]).as_euler('xyz')[2]
        left_foot_yaw = (left_foot_yaw + backend.pi) % (2 * backend.pi) - backend.pi
        
        right_foot_yaw = R.from_matrix(data.site_xmat[self._right_foot_site_id]).as_euler('xyz')[2]
        right_foot_yaw = (right_foot_yaw + backend.pi) % (2 * backend.pi) - backend.pi
        
        feet_yaw_diff_reward = backend.square(
            (left_foot_yaw - right_foot_yaw + backend.pi) % (2 * backend.pi) - backend.pi
        )

        # Feet yaw mean reward
        feet_yaw_mean = (
            (left_foot_yaw * 0.5 + right_foot_yaw * 0.5) +
            backend.pi * (backend.abs(left_foot_yaw - right_foot_yaw) > backend.pi)
        )
        base_yaw = global_rot.as_euler('xyz')[2]
        feet_yaw_mean_reward = backend.square(
            (base_yaw - feet_yaw_mean + backend.pi) % (2 * backend.pi) - backend.pi
        )

        # Feet roll reward
        left_foot_roll = R.from_matrix(data.site_xmat[self._left_foot_site_id]).as_euler('xyz')[0]
        left_foot_roll = (left_foot_roll + backend.pi) % (2 * backend.pi) - backend.pi
        
        right_foot_roll = R.from_matrix(data.site_xmat[self._right_foot_site_id]).as_euler('xyz')[0]
        right_foot_roll = (right_foot_roll + backend.pi) % (2 * backend.pi) - backend.pi
        
        feet_roll_reward = backend.square(left_foot_roll) + backend.square(right_foot_roll)

        feet_pos_local = self._feet_positions_local(data, backend)
        straight_command = (
            backend.abs(goal_vel[0]) >= self._stand_command_threshold
        ) & (
            backend.abs(goal_vel[1]) < self._stand_command_threshold
        ) & (
            backend.abs(goal_vel[2]) < self._stand_command_threshold
        )

        # Feet distance reward
        left_foot_pos = data.site_xpos[self._left_foot_site_id]
        right_foot_pos = data.site_xpos[self._right_foot_site_id]
        if self._feet_distance_mode == "minimum_lateral":
            lateral_width = backend.abs(
                feet_pos_local[0, 1] - feet_pos_local[1, 1]
            )
            lateral_width_deficit = backend.maximum(
                self._feet_distance_target - lateral_width,
                backend.array(0.0),
            )
            feet_distance_reward = backend.square(lateral_width_deficit)
        else:
            feet_distance = (
                backend.cos(base_yaw) * (left_foot_pos[1] - right_foot_pos[1]) -
                backend.sin(base_yaw) * (left_foot_pos[0] - right_foot_pos[0])
            )
            feet_distance_reward = backend.linalg.norm(
                self._feet_distance_target - feet_distance
            )

        # Feet swing reward
        gait_frequency = goal_state.gait_frequency
        gait_process = backend.fmod(reward_state.gait_process + env.dt * gait_frequency, 1.0)
        
        phase_left = gait_process
        phase_right = backend.fmod(gait_process + 0.5, 1.0)
        is_stance_left = phase_left < 0.5
        is_stance_right = phase_right < 0.5
        # When goal xy velocity is ~0, force stance for both feet so the reward
        # encourages keeping both feet planted instead of swinging in place.
        standing_still = (
            backend.linalg.norm(goal_vel[:2]) < self._stand_command_threshold
        ) & (backend.abs(goal_vel[2]) < self._stand_command_threshold)
        is_stance_left = is_stance_left | standing_still
        is_stance_right = is_stance_right | standing_still
        feet_swing_reward = (
            (~(feet_on_ground[0] ^ is_stance_left)).astype(backend.float32)
            + (~(feet_on_ground[1] ^ is_stance_right)).astype(backend.float32)
        )
        feet_swing_reward *= forward_tracking_gate

        # Unitree-style gait shaping: regulate swing-foot clearance, suppress
        # sliding during contact, and keep hip yaw/roll near the nominal pose.
        feet_swing_height_reward = (
            backend.square(left_foot_pos[2] - self._feet_swing_height)
            * (~feet_on_ground[0]).astype(backend.float32)
            + backend.square(right_foot_pos[2] - self._feet_swing_height)
            * (~feet_on_ground[1]).astype(backend.float32)
        )
        feet_symmetry_error_xy = feet_pos_local[0, :2] + feet_pos_local[1, :2]
        feet_x_symmetry_reward = backend.where(
            standing_still,
            backend.square(feet_pos_local[0, 0])
            + backend.square(feet_pos_local[1, 0]),
            backend.square(feet_symmetry_error_xy[0]),
        )
        feet_stride_symmetry_reward = (
            feet_x_symmetry_reward
            + backend.square(feet_symmetry_error_xy[1])
        )
        (
            feet_step_length_symmetry_reward,
            feet_step_length_symmetry_error,
            feet_step_length_symmetry_valid,
            last_left_step_length,
            last_right_step_length,
            left_step_seen,
            right_step_seen,
        ) = self._update_step_length_symmetry(
            reward_state,
            feet_on_ground,
            feet_pos_local,
            straight_command,
            backend,
        )
        (
            leg_joint_cycle_symmetry_reward,
            leg_joint_cycle_symmetry_error,
            leg_joint_cycle_symmetry_valid,
            leg_joint_symmetry_left_qpos_history,
            leg_joint_symmetry_right_qpos_history,
            leg_joint_symmetry_left_qvel_history,
            leg_joint_symmetry_right_qvel_history,
            leg_joint_symmetry_history_index,
            leg_joint_symmetry_history_count,
        ) = self._update_leg_joint_cycle_symmetry(
            reward_state,
            data,
            gait_frequency,
            straight_command,
            env,
            backend,
        )
        standing_leg_joint_symmetry_error = (
            data.qpos[self._leg_joint_symmetry_left_qpos_id]
            - backend.asarray(self._leg_joint_cycle_symmetry_mirror_signs)
            * data.qpos[self._leg_joint_symmetry_right_qpos_id]
        )
        standing_leg_joint_symmetry_reward = (
            backend.mean(backend.square(standing_leg_joint_symmetry_error))
            * standing_still.astype(backend.float32)
        )
        torso_joint_zero_reward = backend.mean(
            backend.square(data.qpos[self._torso_joint_zero_qpos_id])
        )
        hip_pos_reward = backend.square(
            data.qpos[self._hip_pos_qpos_id] - self._nominal_joint_qpos[self._hip_pos_qpos_id]
        ).sum()
        
        # left_swing = (
        #     (backend.abs(gait_process - 0.25) < 0.5 * self._feet_swing_period) & 
        #     (gait_frequency > 1.0e-8)
        # )
        # right_swing = (
        #     (backend.abs(gait_process - 0.75) < 0.5 * self._feet_swing_period) & 
        #     (gait_frequency > 1.0e-8)
        # )
        
        # feet_swing_reward = (
        #     (left_swing & ~feet_on_ground[0]).astype(backend.float32) +
        #     (right_swing & ~feet_on_ground[1]).astype(backend.float32)
        # )

        # Standing no-step penalty
        stand_no_step_reward = standing_still.astype(backend.float32) * (
            (1.0 - feet_on_ground[0].astype(backend.float32)) +
            (1.0 - feet_on_ground[1].astype(backend.float32))
        )

        # Nominal joint position rewards
        joint_qpos_reward = backend.exp(
            -1 * self._nominal_joint_pos_exp *
            backend.square(
                data.qpos[self._nominal_joint_qpos_id] - 
                self._nominal_joint_qpos[self._nominal_joint_qpos_id]
            ).sum()
        )

        joint_deviation_l1_penalty = backend.sum(backend.abs(
            data.qpos[self._nominal_joint_qpos_id] - 
            self._nominal_joint_qpos[self._nominal_joint_qpos_id]
        ))

        # ==================== AIR TIME AND IMPACT REWARDS ====================
        
        # Air time reward
        air_time_reward = 0.0
        tslt = reward_state.time_since_last_touchdown.copy()
        
        for i, _ in enumerate(["left", "right"]):
            foot_on_ground = feet_on_ground[i]
            if backend == np:
                if foot_on_ground:
                    if tslt[i] > 1e-6:  # > 0, to avoid numerical issues
                        air_time_reward += (tslt[i] - self._air_time_max)
                    tslt[i] = 0.0
                else:
                    tslt[i] += env.dt
            else:
                tslt_i, air_time_reward = jax.lax.cond(
                    foot_on_ground,
                    lambda: (0.0, air_time_reward + (tslt[i] - self._air_time_max) * (tslt[i] > 1e-6)),
                    lambda: (tslt[i] + env.dt, air_time_reward)
                )
                tslt = tslt.at[i].set(tslt_i)

        # No fly reward (penalize when both feet are off the ground)
        flying = backend.logical_and(tslt[0] > 0.0, tslt[1] > 0.0)
        no_fly_reward = flying * 1.0

        # Impact reward (penalize high impact forces at the feet)
        left_foot_contact_forces = data.cfrc_ext[self._left_foot_body_ids, 3:6]
        right_foot_contact_forces = data.cfrc_ext[self._right_foot_body_ids, 3:6]
        
        left_foot_contact_force_norm = backend.linalg.norm(left_foot_contact_forces, axis=1)
        right_foot_contact_force_norm = backend.linalg.norm(right_foot_contact_forces, axis=1)
        
        left_foot_impact = left_foot_contact_force_norm > self._impact_threshold
        right_foot_impact = right_foot_contact_force_norm > self._impact_threshold
        
        impact_reward = left_foot_impact * 1.0 + right_foot_impact * 1.0
        impact_reward = backend.mean(impact_reward)

        # Yaw deviation reward
        cur_rot = data.site_xmat[env.base_site_id].reshape(3, 3)
        init_rot = carry.base_site_rot0
        rel_rot = init_rot.T @ cur_rot
        relative_yaw = backend.arctan2(rel_rot[1, 0], rel_rot[0, 0])
        yaw_deviation_scale = backend.where(
            transition_active,
            backend.asarray(self._transition_yaw_deviation_scale),
            backend.asarray(1.0),
        )
        yaw_deviation_reward = relative_yaw ** 2 * yaw_deviation_scale

        # Com root centering
        support_center_xy = 0.5 * (left_foot_pos[:2] + right_foot_pos[:2])
        root_center_target_xy = backend.array([
            self._root_centering_target_x,
            self._root_centering_target_y
        ])
        root_offset_xy = (global_pos_root[:2] - support_center_xy) - root_center_target_xy
        root_centering_reward = backend.sum(backend.square(root_offset_xy))
        root_centering_reward = root_centering_reward * standing_still.astype(backend.float32)

        # Symmetry air reward (currently unused)
        symmetry_air_reward = 0.0

        # ==================== SCALE REWARDS BY COEFFICIENTS ====================
        max_curriculum_step = env.env_cfg.curriculum.end_step
        def curriculum_coeff(value):
            if not isinstance(value, ListConfig):
                return value
            if len(value) == 2:
                coeff_scale = jnp.linspace(
                    value[0], value[1], max_curriculum_step + 1
                )
            elif len(value) == max_curriculum_step + 1:
                coeff_scale = jnp.asarray(value)
            else:
                raise ValueError(
                    "Curriculum reward coefficients must contain either two "
                    "endpoints or one value per curriculum stage"
                )
            step = backend.clip(carry.curriculum.step, 0, max_curriculum_step)
            return coeff_scale[step]

        action_rate_coeff = curriculum_coeff(self._action_rate_coeff)
        base_height_coeff = curriculum_coeff(self._base_height_coeff)
        joint_acc_coeff = curriculum_coeff(self._joint_acc_coeff)
        feet_slip_coeff = curriculum_coeff(self._feet_slip_coeff)
        feet_friction_utilization_coeff = curriculum_coeff(
            self._feet_friction_utilization_coeff
        )
        orientation_coeff = curriculum_coeff(self.orientation_coeff)
        root_centering_coeff = curriculum_coeff(self._root_centering_coeff)
        tracking_world_direction_coeff = curriculum_coeff(
            self._tracking_world_direction_coeff
        )
        feet_swing_height_coeff = curriculum_coeff(
            self._feet_swing_height_coeff
        )
        feet_stride_symmetry_coeff = curriculum_coeff(
            self._feet_stride_symmetry_coeff
        )
        feet_distance_coeff = curriculum_coeff(self._feet_distance_coeff)
        feet_step_length_symmetry_coeff = curriculum_coeff(
            self._feet_step_length_symmetry_coeff
        )
        leg_joint_cycle_symmetry_coeff = curriculum_coeff(
            self._leg_joint_cycle_symmetry_coeff
        )
        standing_leg_joint_symmetry_coeff = curriculum_coeff(
            self._standing_leg_joint_symmetry_coeff
        )
        torso_joint_zero_coeff = curriculum_coeff(
            self._torso_joint_zero_coeff
        )
        hip_pos_coeff = curriculum_coeff(self._hip_pos_coeff)

        survival_reward *= (self._survival * env.dt)
        tracking_reward_linvel_x *= (self._tracking_w_sum_linvel_x * env.dt)
        tracking_reward_linvel_y *= (self._tracking_w_sum_linvel_y * env.dt)
        tracking_reward_angvel *= (self._tracking_w_sum_angvel * env.dt)
        tracking_world_direction_reward *= (tracking_world_direction_coeff * env.dt)
        lateral_velocity_penalty *= (
            self._lateral_velocity_penalty_coeff * env.dt
        )
        yaw_velocity_penalty *= (self._yaw_velocity_penalty_coeff * env.dt)
        cross_track_penalty *= (self._cross_track_penalty_coeff * env.dt)
        joint_qpos_reward *= (self._nominal_joint_pos_coeff * env.dt)
        joint_deviation_l1_penalty *= (self._joint_deviation_l1_coeff * env.dt)
        base_height_reward *= (base_height_coeff * env.dt)
        orientation_reward *= (orientation_coeff * env.dt)
        torque_reward *= (self._joint_torque_coeff * env.dt)
        energy_reward *= (self._energy_coeff * env.dt)
        z_vel_reward *= (self._z_vel_coeff * env.dt)
        roll_pitch_vel_reward *= (self._roll_pitch_vel_coeff * env.dt)
        joint_vel_reward *= (self._joint_vel_coeff * env.dt)
        acceleration_reward *= (joint_acc_coeff * env.dt)
        root_acceleration_reward *= (self._root_acc_coeff * env.dt)
        action_rate_reward *= (action_rate_coeff * env.dt)
        joint_position_limit_reward *= (self._joint_position_limit_coeff * env.dt)
        feet_slip_reward *= (feet_slip_coeff * env.dt)
        feet_friction_utilization_reward *= (
            feet_friction_utilization_coeff * env.dt
        )
        feet_yaw_diff_reward *= (self._feet_yaw_diff_coeff * env.dt)
        feet_yaw_mean_reward *= (self._feet_yaw_mean_coeff * env.dt)
        feet_roll_reward *= (self._feet_roll_coeff * env.dt)
        feet_distance_reward *= (feet_distance_coeff * env.dt)
        feet_swing_reward *= (self._feet_swing_coeff * env.dt)
        feet_swing_height_reward *= (feet_swing_height_coeff * env.dt)
        feet_stride_symmetry_reward *= (feet_stride_symmetry_coeff * env.dt)
        feet_step_length_symmetry_reward *= (
            feet_step_length_symmetry_coeff * env.dt
        )
        leg_joint_cycle_symmetry_reward *= (
            leg_joint_cycle_symmetry_coeff * env.dt
        )
        standing_leg_joint_symmetry_reward *= (
            standing_leg_joint_symmetry_coeff * env.dt
        )
        torso_joint_zero_reward *= (torso_joint_zero_coeff * env.dt)
        hip_pos_reward *= (hip_pos_coeff * env.dt)
        air_time_reward *= (self._air_time_coeff * env.dt)
        no_fly_reward *= (self._no_fly_coeff * env.dt)
        impact_reward *= (self._impact_coeff * env.dt)
        tracking_base_pos_reward *= (self._tracking_base_pos_coeff * env.dt)
        yaw_deviation_reward *= (self._yaw_deviation_coeff * env.dt)
        stand_no_step_reward *= (self._stand_no_step_coeff * env.dt)
        root_centering_reward *= (root_centering_coeff * env.dt)

        # ==================== COMBINE REWARDS ====================
        
        tracking_reward = (
            tracking_reward_linvel_x + tracking_reward_linvel_y + tracking_reward_angvel +
            tracking_world_direction_reward +
            joint_qpos_reward + feet_swing_reward + tracking_base_pos_reward
        )
        
        penalty_rewards = (
            base_height_reward + orientation_reward + torque_reward + 
            energy_reward + z_vel_reward + roll_pitch_vel_reward + joint_vel_reward +
            lateral_velocity_penalty + yaw_velocity_penalty + cross_track_penalty +
            acceleration_reward + root_acceleration_reward + action_rate_reward + 
            joint_position_limit_reward + feet_slip_reward + 
            feet_friction_utilization_reward +
            feet_yaw_diff_reward + feet_yaw_mean_reward + feet_roll_reward +
            feet_distance_reward + feet_swing_height_reward +
            feet_stride_symmetry_reward + feet_step_length_symmetry_reward +
            leg_joint_cycle_symmetry_reward +
            standing_leg_joint_symmetry_reward + torso_joint_zero_reward +
            hip_pos_reward +
            air_time_reward + no_fly_reward + impact_reward +
            joint_deviation_l1_penalty + yaw_deviation_reward + stand_no_step_reward + root_centering_reward
        )
        
        total_reward = survival_reward + tracking_reward + penalty_rewards
        if self._only_positive_rewards:
            total_reward = backend.maximum(total_reward, 0.0)
        
        # Handle NaN values
        total_reward = backend.nan_to_num(total_reward, nan=0.0)

        # ==================== UPDATE REWARD STATE ====================
        
        # Update reward state with new values
        reward_state = reward_state.replace(
            gait_process=gait_process,
            last_qvel=data.qvel, 
            last_action=action, 
            time_since_last_touchdown=tslt,
            last_left_step_length=last_left_step_length,
            last_right_step_length=last_right_step_length,
            left_step_seen=left_step_seen,
            right_step_seen=right_step_seen,
            feet_step_length_symmetry_error=feet_step_length_symmetry_error,
            feet_step_length_symmetry_valid=feet_step_length_symmetry_valid,
            leg_joint_symmetry_left_qpos_history=leg_joint_symmetry_left_qpos_history,
            leg_joint_symmetry_right_qpos_history=leg_joint_symmetry_right_qpos_history,
            leg_joint_symmetry_left_qvel_history=leg_joint_symmetry_left_qvel_history,
            leg_joint_symmetry_right_qvel_history=leg_joint_symmetry_right_qvel_history,
            leg_joint_symmetry_history_index=leg_joint_symmetry_history_index,
            leg_joint_symmetry_history_count=leg_joint_symmetry_history_count,
            leg_joint_cycle_symmetry_error=leg_joint_cycle_symmetry_error,
            leg_joint_cycle_symmetry_valid=leg_joint_cycle_symmetry_valid,
        )
        
        # Update reward components dictionary
        updated_reward_components = {
            "survival": survival_reward,
            "tracking_linvel_x": tracking_reward_linvel_x,
            "tracking_linvel_y": tracking_reward_linvel_y,
            "tracking_angvel": tracking_reward_angvel,
            "tracking_world_direction": tracking_world_direction_reward,
            "lateral_velocity_penalty": lateral_velocity_penalty,
            "yaw_velocity_penalty": yaw_velocity_penalty,
            "cross_track_penalty": cross_track_penalty,
            "tracking_joint_qpos": joint_qpos_reward,
            "tracking_feet_swing": feet_swing_reward,
            "feet_swing_height": feet_swing_height_reward,
            "feet_stride_symmetry": feet_stride_symmetry_reward,
            "feet_step_length_symmetry": feet_step_length_symmetry_reward,
            "leg_joint_cycle_symmetry": leg_joint_cycle_symmetry_reward,
            "standing_leg_joint_symmetry": standing_leg_joint_symmetry_reward,
            "torso_joint_zero": torso_joint_zero_reward,
            "hip_pos": hip_pos_reward,
            "base_height": base_height_reward,
            "joint_deviation_l1": joint_deviation_l1_penalty,
            "orientation": orientation_reward,
            "torque": torque_reward,
            "energy": energy_reward,
            "z_vel": z_vel_reward,
            "roll_pitch_vel": roll_pitch_vel_reward,
            "joint_vel": joint_vel_reward,
            "acceleration": acceleration_reward,
            "root_acceleration": root_acceleration_reward,
            "action_rate": action_rate_reward,
            "joint_position_limit": joint_position_limit_reward,
            "feet_slip": feet_slip_reward,
            "feet_friction_utilization": feet_friction_utilization_reward,
            "feet_yaw_diff": feet_yaw_diff_reward,
            "feet_yaw_mean": feet_yaw_mean_reward,
            "feet_roll": feet_roll_reward,
            "feet_distance": feet_distance_reward,
            "air_time": air_time_reward,
            "no_fly": no_fly_reward,
            "impact": impact_reward,
            "tracking_base_pos": tracking_base_pos_reward,
            "yaw_deviation": yaw_deviation_reward,
            "stand_no_step": stand_no_step_reward,
            "root_centering": root_centering_reward,
        }
        
        reward_state = reward_state.replace(reward_components=updated_reward_components)
        carry = carry.replace(reward_state=reward_state)
        
        return total_reward, carry
