from typing import List, Union, Tuple
import mujoco
from mujoco import MjSpec

import loco_mujoco
from loco_mujoco.core import ObservationType, Observation
from loco_mujoco.environments.humanoids.base_robot_humanoid import BaseRobotHumanoid
from loco_mujoco.core.utils import info_property


class UnitreeH1(BaseRobotHumanoid):

    """
    Description
    ------------

    Mujoco environment of the Unitree H1 robot.


    Default Observation Space
    -----------------

    ============ ================== ================ ==================================== ============================== ===
    Index in Obs Name               ObservationType  Min                                  Max                            Dim
    ============ ================== ================ ==================================== ============================== ===
    0 - 4        q_root             FreeJointPosNoXY [-inf, -inf, -inf, -inf, -inf]       [inf, inf, inf, inf, inf]      5
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    5            q_torso         JointPos         [-2.35]                              [2.35]                         1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    6            q_left_shoulder_pitch        JointPos         [-2.87]                              [2.87]                         1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    7            q_left_shoulder_roll        JointPos         [-0.34]                              [3.11]                         1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    8            q_left_shoulder_yaw        JointPos         [-1.3]                               [4.45]                         1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    9            q_left_elbow       JointPos         [-1.25]                              [2.61]                         1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    10           q_right_shoulder_pitch        JointPos         [-2.87]                              [2.87]                         1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    11           q_right_shoulder_roll        JointPos         [-3.11]                              [0.34]                         1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    12           q_right_shoulder_yaw        JointPos         [-4.45]                              [1.3]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    13           q_right_elbow      JointPos         [-1.25]                              [2.61]                         1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    14           q_right_hip_pitch    JointPos         [-1.57]                              [1.57]                         1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    15           q_right_hip_roll  JointPos         [-0.43]                              [0.43]                         1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    16           q_right_hip_yaw   JointPos         [-0.43]                              [0.43]                         1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    17           q_right_knee     JointPos         [-0.26]                              [2.05]                         1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    18           q_right_ankle    JointPos         [-0.87]                              [0.52]                         1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    19           q_left_hip_pitch    JointPos         [-1.57]                              [1.57]                         1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    20           q_left_hip_roll  JointPos         [-0.43]                              [0.43]                         1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    21           q_left_hip_yaw   JointPos         [-0.43]                              [0.43]                         1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    22           q_left_knee     JointPos         [-0.26]                              [2.05]                         1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    23           q_left_ankle    JointPos         [-0.87]                              [0.52]                         1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    24 - 29      dq_root            FreeJointVel     [-inf, -inf, -inf, -inf, -inf, -inf] [inf, inf, inf, inf, inf, inf] 6
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    30           dq_torso        JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    31           dq_left_shoulder_pitch       JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    32           dq_left_shoulder_roll       JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    33           dq_left_shoulder_yaw       JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    34           dq_left_elbow      JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    35           dq_right_shoulder_pitch       JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    36           dq_right_shoulder_roll       JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    37           dq_right_shoulder_yaw       JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    38           dq_right_elbow     JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    39           dq_right_hip_pitch   JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    40           dq_right_hip_roll JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    41           dq_right_hip_yaw  JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    42           dq_right_knee    JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    43           dq_right_ankle   JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    44           dq_left_hip_pitch   JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    45           dq_left_hip_roll JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    46           dq_left_hip_yaw  JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    47           dq_left_knee    JointVel         [-inf]                               [inf]                          1
    ------------ ------------------ ---------------- ------------------------------------ ------------------------------ ---
    48           dq_left_ankle   JointVel         [-inf]                               [inf]                          1
    ============ ================== ================ ==================================== ============================== ===

    Default Action Space
    -----------------

    Control function type: **DefaultControl**

    See control function interface for more details.

    =============== ==== ===
    Index in Action Min  Max
    =============== ==== ===
    0               -1.0 1.0
    --------------- ---- ---
    1               -1.0 1.0
    --------------- ---- ---
    2               -1.0 1.0
    --------------- ---- ---
    3               -1.0 1.0
    --------------- ---- ---
    4               -1.0 1.0
    --------------- ---- ---
    5               -1.0 1.0
    --------------- ---- ---
    6               -1.0 1.0
    --------------- ---- ---
    7               -1.0 1.0
    --------------- ---- ---
    8               -1.0 1.0
    --------------- ---- ---
    9               -1.0 1.0
    --------------- ---- ---
    10              -1.0 1.0
    --------------- ---- ---
    11              -1.0 1.0
    --------------- ---- ---
    12              -1.0 1.0
    --------------- ---- ---
    13              -1.0 1.0
    --------------- ---- ---
    14              -1.0 1.0
    --------------- ---- ---
    15              -1.0 1.0
    --------------- ---- ---
    16              -1.0 1.0
    --------------- ---- ---
    17              -1.0 1.0
    --------------- ---- ---
    18              -1.0 1.0
    =============== ==== ===


    Methods
    ------------

    """

    mjx_enabled = False

    def __init__(self, disable_arms: bool = False, disable_back_joint: bool = False,
                 spec: Union[str, MjSpec] = None,
                 observation_spec: List[Observation] = None,
                 actuation_spec: List[str] = None,
                 **kwargs) -> None:
        """
        Constructor.

        Args:
            disable_arms (bool): Whether to disable arm joints.
            disable_back_joint (bool): Whether to disable the back joint.
            spec (Union[str, MjSpec]): Specification of the environment. Can be a path to the XML file or an MjSpec object.
                If none is provided, the default XML file is used.
            observation_spec (List[Observation], optional): List defining the observation space. Defaults to None.
            actuation_spec (List[str], optional): List defining the action space. Defaults to None.
            **kwargs: Additional parameters for the environment.
        """

        self._disable_arms = disable_arms
        self._disable_back_joint = disable_back_joint

        if spec is None:
            spec = self.get_default_xml_file_path()

        # load the model specification
        spec = mujoco.MjSpec.from_file(spec) if not isinstance(spec, MjSpec) else spec

        # get the observation and action specification
        if observation_spec is None:
            # get default
            observation_spec = self._get_observation_specification(spec)
        else:
            # parse
            observation_spec = self.parse_observation_spec(observation_spec)
        if actuation_spec is None:
            actuation_spec = self._get_action_specification(spec)

        # modify the specification if needed
        if self.mjx_enabled:
            spec = self._modify_spec_for_mjx(spec)
        if disable_arms or disable_back_joint:
            joints_to_remove, actuators_to_remove, equ_constraints_to_remove = self._get_spec_modifications()
            obs_to_remove = ["q_" + j for j in joints_to_remove] + ["dq_" + j for j in joints_to_remove]
            observation_spec = [elem for elem in observation_spec if elem.name not in obs_to_remove]
            actuation_spec = [ac for ac in actuation_spec if ac not in actuators_to_remove]
            spec = self._delete_from_spec(spec, joints_to_remove,
                                          actuators_to_remove, equ_constraints_to_remove)
            if disable_arms:
                spec = self._reorient_arms(spec)

        super().__init__(spec=spec, actuation_spec=actuation_spec, observation_spec=observation_spec, **kwargs)

    def _get_spec_modifications(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Specifies which joints, actuators, and equality constraints should be removed from the Mujoco specification.

        Returns:
            Tuple[List[str], List[str], List[str]]: A tuple containing lists of joints to remove, actuators to remove,
            and equality constraints to remove.
        """

        joints_to_remove = []
        actuators_to_remove = []
        equ_constr_to_remove = []

        if self._disable_arms:
            joints_to_remove += ["left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "right_shoulder_pitch",
                                 "right_shoulder_roll", "right_shoulder_yaw", "right_elbow"]
            actuators_to_remove += ["left_shoulder_pitch_actuator", "left_shoulder_roll_actuator", "left_shoulder_yaw_actuator",
                                    "left_elbow", "right_shoulder_pitch_actuator", "right_shoulder_roll_actuator",
                                    "right_shoulder_yaw_actuator", "right_elbow"]

        if self._disable_back_joint:
            joints_to_remove += ["torso"]
            actuators_to_remove += ["torso_actuator"]

        return joints_to_remove, actuators_to_remove, equ_constr_to_remove


    @staticmethod
    def _reorient_arms(spec: MjSpec) -> MjSpec:
        """
        Reorients the arms to prevent collision with the hips. Used when disable_arms is set to True.

        Args:
            spec (MjSpec): Mujoco specification.

        Returns:
            MjSpec: Modified Mujoco specification.
        """
        # modify the arm orientation
        left_shoulder_pitch_link = spec.find_body("left_shoulder_pitch_link")
        left_shoulder_pitch_link.quat = [1.0, 0.25, 0.1, 0.0]
        right_elbow_link = spec.find_body("right_elbow_link")
        right_elbow_link.quat = [1.0, 0.0, 0.25, 0.0]
        right_shoulder_pitch_link = spec.find_body("right_shoulder_pitch_link")
        right_shoulder_pitch_link.quat = [1.0, -0.25, 0.1, 0.0]
        left_elbow_link = spec.find_body("left_elbow_link")
        left_elbow_link.quat = [1.0, 0.0, 0.25, 0.0]

        return spec

    @staticmethod
    def _get_observation_specification(spec: MjSpec) -> List[Observation]:
        """
        Returns the observation specification of the environment.

        Args:
            spec (MjSpec): Specification of the environment.

        Returns:
            List[Observation]: List of observations.
        """
        observation_spec = [# ------------- JOINT POS -------------
                            ObservationType.FreeJointPosNoXY("q_root", xml_name="root"),
                            ObservationType.JointPos("q_torso", xml_name="torso"),
                            ObservationType.JointPos("q_left_shoulder_pitch", xml_name="left_shoulder_pitch"),
                            ObservationType.JointPos("q_left_shoulder_roll", xml_name="left_shoulder_roll"),
                            ObservationType.JointPos("q_left_shoulder_yaw", xml_name="left_shoulder_yaw"),
                            ObservationType.JointPos("q_left_elbow", xml_name="left_elbow"),
                            ObservationType.JointPos("q_right_shoulder_pitch", xml_name="right_shoulder_pitch"),
                            ObservationType.JointPos("q_right_shoulder_roll", xml_name="right_shoulder_roll"),
                            ObservationType.JointPos("q_right_shoulder_yaw", xml_name="right_shoulder_yaw"),
                            ObservationType.JointPos("q_right_elbow", xml_name="right_elbow"),
                            ObservationType.JointPos("q_right_hip_pitch", xml_name="right_hip_pitch"),
                            ObservationType.JointPos("q_right_hip_roll", xml_name="right_hip_roll"),
                            ObservationType.JointPos("q_right_hip_yaw", xml_name="right_hip_yaw"),
                            ObservationType.JointPos("q_right_knee", xml_name="right_knee"),
                            ObservationType.JointPos("q_right_ankle", xml_name="right_ankle"),
                            ObservationType.JointPos("q_left_hip_pitch", xml_name="left_hip_pitch"),
                            ObservationType.JointPos("q_left_hip_roll", xml_name="left_hip_roll"),
                            ObservationType.JointPos("q_left_hip_yaw", xml_name="left_hip_yaw"),
                            ObservationType.JointPos("q_left_knee", xml_name="left_knee"),
                            ObservationType.JointPos("q_left_ankle", xml_name="left_ankle"),

                            # ------------- JOINT VEL -------------
                            ObservationType.FreeJointVel("dq_root", xml_name="root"),
                            ObservationType.JointVel("dq_torso", xml_name="torso"),
                            ObservationType.JointVel("dq_left_shoulder_pitch", xml_name="left_shoulder_pitch"),
                            ObservationType.JointVel("dq_left_shoulder_roll", xml_name="left_shoulder_roll"),
                            ObservationType.JointVel("dq_left_shoulder_yaw", xml_name="left_shoulder_yaw"),
                            ObservationType.JointVel("dq_left_elbow", xml_name="left_elbow"),
                            ObservationType.JointVel("dq_right_shoulder_pitch", xml_name="right_shoulder_pitch"),
                            ObservationType.JointVel("dq_right_shoulder_roll", xml_name="right_shoulder_roll"),
                            ObservationType.JointVel("dq_right_shoulder_yaw", xml_name="right_shoulder_yaw"),
                            ObservationType.JointVel("dq_right_elbow", xml_name="right_elbow"),
                            ObservationType.JointVel("dq_right_hip_pitch", xml_name="right_hip_pitch"),
                            ObservationType.JointVel("dq_right_hip_roll", xml_name="right_hip_roll"),
                            ObservationType.JointVel("dq_right_hip_yaw", xml_name="right_hip_yaw"),
                            ObservationType.JointVel("dq_right_knee", xml_name="right_knee"),
                            ObservationType.JointVel("dq_right_ankle", xml_name="right_ankle"),
                            ObservationType.JointVel("dq_left_hip_pitch", xml_name="left_hip_pitch"),
                            ObservationType.JointVel("dq_left_hip_roll", xml_name="left_hip_roll"),
                            ObservationType.JointVel("dq_left_hip_yaw", xml_name="left_hip_yaw"),
                            ObservationType.JointVel("dq_left_knee", xml_name="left_knee"),
                            ObservationType.JointVel("dq_left_ankle", xml_name="left_ankle")]

        return observation_spec

    @staticmethod
    def _get_action_specification(spec: MjSpec) -> List[str]:
        """
        Getter for the action space specification.

        Args:
            spec (MjSpec): Specification of the environment.

        Returns:
            List[str]: List of action names.
        """
        action_spec = ["torso", "left_shoulder_pitch", "left_shoulder_roll",
                       "left_shoulder_yaw", "left_elbow", "right_shoulder_pitch", "right_shoulder_roll",
                       "right_shoulder_yaw", "right_elbow", "right_hip_pitch",
                       "right_hip_roll", "right_hip_yaw", "right_knee",
                       "right_ankle", "left_hip_pitch", "left_hip_roll",
                       "left_hip_yaw", "left_knee", "left_ankle"]

        return action_spec

    @classmethod
    def get_default_xml_file_path(cls) -> str:
        """
        Returns the default XML file path for the Unitree H1 environment.
        """
        # return (loco_mujoco.PATH_TO_MODELS / "unitree_h1" / "h1.xml").as_posix()
        return "/home/steven/code/juggling/humanoid_juggling/humanoid_juggling/envs/assets/mjcf/juggling_scene.xml"

    @info_property
    def upper_body_xml_name(self) -> str:
        """
        Returns the name of the upper body specified in the XML file.
        """
        return "torso_link"

    @info_property
    def root_free_joint_xml_name(self) -> str:
        """
        Returns the name of the free joint of the root specified in the XML file.
        """
        return "root"

    @info_property
    def root_height_healthy_range(self) -> Tuple[float, float]:
        """
        Returns the healthy range of the root height. This is only used when HeightBasedTerminalStateHandler is used.
        """
        return (0.6, 1.5)

    @info_property
    def foot_geom_names(self) -> List[str]:
        """
        Returns the names of the foot geometries.

        Returns:
            List[str]: The names of the foot geometries.
        """
        # print("---------using new foot geom names for H1---------------")
        return ["left_foot1_col", "left_foot2_col",
                "right_foot1_col", "right_foot2_col"
                ]