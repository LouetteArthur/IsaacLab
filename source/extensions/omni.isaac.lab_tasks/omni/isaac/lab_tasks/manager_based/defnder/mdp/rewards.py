from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    target: Articulation = env.scene[target_cfg.name]

    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    target_joint_pos = wrap_to_pi(target.data.joint_pos[:, target_cfg.joint_ids])
    # compute the reward
    # print(f"INFO:Current joint pos: {joint_pos}")
    # print(f"INFO:Target joint pos: {target_joint_pos}")
    reward = torch.sum(torch.square(joint_pos - target_joint_pos), dim=1)
    # print("Reward:", reward)
    return reward


def joint_pos_target(env: ManagerBasedRLEnv, target_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the joint position that is targeted"""
    target: Articulation = env.scene[target_cfg.name]
    target_joint_pos = wrap_to_pi(target.data.joint_pos[:, target_cfg.joint_ids])
    return target_joint_pos