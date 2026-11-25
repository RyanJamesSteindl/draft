"""Hexapod base task: ``make_base_env_cfg`` with no curriculum (as ``quadruped/env.py``)."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from draft.tasks.hexapod.base_env import make_base_env_cfg


def make_env_cfg(**base) -> ManagerBasedRlEnvCfg:
    """The base environment; kwargs pass straight to ``make_base_env_cfg``."""
    cfg, _ctx = make_base_env_cfg(**base)
    return cfg
