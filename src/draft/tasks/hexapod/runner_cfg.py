"""PPO runner config: the quadruped's, unchanged except the run's name."""

from __future__ import annotations

from draft.tasks.quadruped.runner_cfg import QuadrupedLocomotionRunnerCfg


def make_runner_cfg() -> QuadrupedLocomotionRunnerCfg:
    return QuadrupedLocomotionRunnerCfg(
        run_name="hexapod_locomotion", wandb_project="hexapod_locomotion")
