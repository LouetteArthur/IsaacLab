from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    # retrieve the target joint positions
    target = torch.tensor([env.cfg.target_aza_axis, env.cfg.target_ela_axis, env.cfg.target_saza_axis, env.cfg.target_sela_axis], device=env.device)
    return torch.sum(torch.square(joint_pos - target), dim=1)


def joint_pos_target(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the joint position that is targeted"""
    return torch.tensor([[env.cfg.target_aza_axis, env.cfg.target_ela_axis, env.cfg.target_saza_axis, env.cfg.target_sela_axis]], device=env.device)