import math
import torch
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.utils import configclass

import omni.isaac.lab_tasks.manager_based.defnder.mdp as mdp
from omni.isaac.lab_assets.droneDeFNder import DEFNDER_CFG


##
# Scene definition
##


@configclass
class DeFNderSceneCfg(InteractiveSceneCfg):
    """Configuration for a defnder scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation
    robot: ArticulationCfg = DEFNDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    # no commands for this MDP
    null = mdp.NullCommandCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=-1.0)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # (3) Primary task: joint position tracking
    distance_to_target = RewTerm(
        func=mdp.joint_pos_target_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["aza_axis", "ela_axis", "saza_axis", "sela_axis"])},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Configuration for the curriculum."""

    pass


@configclass
class ActionsCfg:
    """Action specifications for the environment."""

    defnder_joint_velocity = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=["aza_axis", "ela_axis", "saza_axis", "sela_axis"])


@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        target_joint_pos = ObsTerm(func=mdp.joint_pos_target)
      
        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # on reset
    reset_aza_axis = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["aza_axis"]),
            "position_range": (-math.pi, math.pi),
            "velocity_range": (-math.pi / 2, math.pi / 2),
        },
    )

    reset_ela_axis = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["ela_axis"]),
            "position_range": (-0.52, 1.04),
            "velocity_range": (-math.pi / 2, math.pi / 2),
        },
    )

    reset_saza_axis = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["saza_axis"]),
            "position_range": (-0.2, 0.2),
            "velocity_range": (-math.pi / 2, math.pi / 2),
        }
    )

    reset_sela_axis = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["sela_axis"]),
            "position_range": (-0.2, 0.2),
            "velocity_range": (-math.pi / 2, math.pi / 2),
        }
    )


@configclass
class DefnderEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the defnder environment."""

    # Scene settings
    scene : DeFNderSceneCfg = DeFNderSceneCfg(num_envs=1, env_spacing=2.5)
    # Basic settings
    observations : ObservationsCfg = ObservationsCfg()
    actions : ActionsCfg = ActionsCfg()
    events : EventCfg = EventCfg()
    # MDP settings
    curriculum: CurriculumCfg = CurriculumCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    # No command generator
    commands: CommandsCfg = CommandsCfg()
    # Target positions for joints
    target_aza_axis: float = 0.0
    target_ela_axis: float = 0.0

    def __post_init__(self):
        """Post initialization."""
        # viewer settings
        self.viewer.eye = (4.5, 0.0, 6.0)
        self.viewer.lookat = (0.0, 0.0, 2.0)
        # step settings
        self.decimation = 1  # env step every 1 sim steps: 100Hz / 1 = 100Hz
        # simulation settings
        self.sim.dt = 0.01  # sim step every 10ms: 100Hz
        self.episode_length_s = 60  # 60 seconds
        self.target_aza_axis = torch.distributions.Uniform(-math.pi, math.pi).sample().item()
        self.target_ela_axis = torch.distributions.Uniform(-0.52, 1.04).sample().item()
        self.target_saza_axis = torch.distributions.Uniform(-0.2, 0.2).sample().item()
        self.target_sela_axis = torch.distributions.Uniform(-0.2, 0.2).sample().item()