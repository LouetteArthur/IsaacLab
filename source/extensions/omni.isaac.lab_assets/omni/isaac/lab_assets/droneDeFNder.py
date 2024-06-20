"""Configuration for the DeFNder robot."""
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg
import os.path as osp


ASSET_PATH = osp.join(osp.dirname(__file__), "assets")

DEFNDER_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=osp.join(ASSET_PATH, "urdf/dfnder_medium.urdf"),
        usd_dir=osp.join(ASSET_PATH, "usd"),
        usd_file_name="dfnder_medium.usd",
        merge_fixed_joints=True,
        fix_base=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0), joint_pos={"aza_axis": 0.0, "ela_axis": 0.0, "saza_axis": 0.15, "sela_axis": 0.0}
    ),
    actuators={
        "aza_axis": ImplicitActuatorCfg(
            joint_names_expr=["aza_axis"], effort_limit=5000.0, velocity_limit=1.5707963267948966, stiffness=0, damping=1000),
        "ela_axis": ImplicitActuatorCfg(
            joint_names_expr=["ela_axis"], effort_limit=5000.0, velocity_limit=1.5707963267948966, stiffness=0, damping=1000),
        "saza_axis": ImplicitActuatorCfg(
            joint_names_expr=["saza_axis"], effort_limit=5000.0, velocity_limit=1.5707963267948966, stiffness=0, damping=1000),
        "sela_axis": ImplicitActuatorCfg(
            joint_names_expr=["sela_axis"], effort_limit=5000.0, velocity_limit=1.5707963267948966, stiffness=0, damping=1000),
    },
)

DRONE_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=osp.join(ASSET_PATH, "urdf/drone.urdf"),
        usd_dir=osp.join(ASSET_PATH, "usd"),
        usd_file_name="drone.usd",
        merge_fixed_joints=True,
        fix_base=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 100.0), joint_pos={"x_axis": 0.0, "y_axis": 0.0, "z_axis": 0.0}
    ),
    actuators={
        "x_axis": ImplicitActuatorCfg(
            joint_names_expr=["x_axis"], effort_limit=5000.0, velocity_limit=10,),
        "y_axis": ImplicitActuatorCfg(
            joint_names_expr=["y_axis"], effort_limit=5000.0, velocity_limit=10,),
        "z_axis": ImplicitActuatorCfg(
            joint_names_expr=["z_axis"], effort_limit=5000.0, velocity_limit=10,),
    },
)
