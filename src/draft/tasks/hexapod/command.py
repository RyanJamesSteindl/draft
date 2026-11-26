"""Body-height and velocity command with a speed-tied gait clock (hexapod-sim M3).

``HeightVelocityCommand`` is mjlab's ``UniformVelocityCommand`` plus:

- **Body height.** A target trunk height, sampled uniformly from
  ``height_range`` each time the velocity is resampled. The command the policy
  sees is ``[vx, vy, wz, h]``; the mjlab rewards read only the first three.
- **Speed cap by height.** ``vx`` and ``vy`` are sampled as fractions of the
  range in ``ranges`` and then scaled by ``cap(h) / ranges.lin_vel_x[1]``, where
  ``cap(h) = speed_fraction * cap_share * top_speed(h)``. ``top_speed`` comes
  from a table (the hexapod-sim M2 speed tool: tripod top speed against body
  height); ``speed_fraction`` is raised in stages by :func:`speed_stages`.
  The planar speed ``|vx, vy|`` is then clipped to ``cap(h)`` as well.
- **Gait clock.** A stride phase integrated every step at
  ``f = clip(duty * |v_xy| / (stroke_share * stroke(h)), f_min, f_top(h))``: the
  stride lengthens up to ``stroke_share`` of the stance stroke the M2 tool gives
  at that height, then the step rate rises, never above the M2 rate at top
  speed. Integrated rather than ``t * f``, so changing ``f`` does not make the
  phase jump. Reset to 0 with each episode.

The tables are plain numbers passed through the run config, so the task does
not depend on the analysis code that produced them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp.velocity_command import (
    UniformVelocityCommand, UniformVelocityCommandCfg,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def height_above_terrain(env: "ManagerBasedRlEnv", sensor_name: str = "terrain_scan",
                         asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Trunk height above the ground under it: root z minus the median of the
    torso terrain scan's hits (rays that miss are ignored).

    The terrain is rough and many tiles are raised (pyramid stairs, slopes), so
    world z is not body height. Fixed 27 Nov after the first M3 pilot, whose
    world-z reward drove the trunk down onto raised tiles.
    """
    z = env.scene[asset_cfg.name].data.root_link_pos_w[:, 2]
    data = env.scene.sensors[sensor_name].data
    hit_z = data.hit_pos_w[..., 2]
    hit_z = torch.where(data.distances >= 0, hit_z, torch.full_like(hit_z, float("nan")))
    ground = torch.nanmedian(hit_z, dim=-1).values
    return z - torch.nan_to_num(ground, nan=0.0)


def _interp(x: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear ``ys(x)``, clamped at the ends (``xs`` increasing)."""
    x = x.clamp(xs[0], xs[-1])
    i = torch.searchsorted(xs, x.contiguous()).clamp(1, len(xs) - 1)
    x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


@dataclass(kw_only=True)
class HeightVelocityCommandCfg(UniformVelocityCommandCfg):
    height_range: tuple[float, float]
    #: Ray-cast sensor whose hits give the ground under the trunk.
    terrain_sensor: str = "terrain_scan"
    table_heights: Sequence[float]
    table_top_speed: Sequence[float]
    table_stroke: Sequence[float]
    table_step_rate: Sequence[float]
    #: Share of the M2 top speed the command may reach (agreed 27 Nov: 0.8).
    cap_share: float = 0.8
    #: Raised by the speed_stages curriculum; 1.0 means the full cap.
    speed_fraction: float = 1.0
    duty: float = 0.5
    stroke_share: float = 0.75
    min_step_rate: float = 1.2

    def build(self, env: "ManagerBasedRlEnv") -> "HeightVelocityCommand":
        return HeightVelocityCommand(self, env)


class HeightVelocityCommand(UniformVelocityCommand):
    cfg: HeightVelocityCommandCfg

    def __init__(self, cfg: HeightVelocityCommandCfg, env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        t = lambda v: torch.tensor(list(v), dtype=torch.float32, device=self.device)
        self._h, self._top = t(cfg.table_heights), t(cfg.table_top_speed)
        self._stroke, self._rate = t(cfg.table_stroke), t(cfg.table_step_rate)
        if not bool((self._h[1:] > self._h[:-1]).all()):
            raise ValueError("table_heights must be increasing")
        self.height_command = torch.full((self.num_envs,), sum(cfg.height_range) / 2,
                                         device=self.device)
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.step_rate = torch.full((self.num_envs,), cfg.min_step_rate, device=self.device)
        self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return torch.cat((self.vel_command_b, self.height_command.unsqueeze(-1)), dim=-1)

    def speed_cap(self, h: torch.Tensor) -> torch.Tensor:
        """Largest commanded planar speed at body height ``h`` right now."""
        return self.cfg.speed_fraction * self.cfg.cap_share * _interp(h, self._h, self._top)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        lo, hi = self.cfg.height_range
        self.height_command[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(lo, hi)
        # The parent samples within `ranges` (forward-only envs included); rescale
        # so ranges.lin_vel_x[1] maps to this height's cap.
        super()._resample_command(env_ids)
        cap = self.speed_cap(self.height_command[env_ids])
        v = self.vel_command_b[env_ids, :2] * (cap / self.cfg.ranges.lin_vel_x[1]).unsqueeze(-1)
        # Planar speed, not just vx, stays under the cap (vy rides on vx otherwise).
        v *= (cap / v.norm(dim=-1).clamp(min=1e-6)).clamp(max=1.0).unsqueeze(-1)
        self.vel_command_b[env_ids, :2] = v
        self.vel_command_w[env_ids, :2] = v

    def _update_metrics(self) -> None:
        super()._update_metrics()
        steps = self.cfg.resampling_time_range[1] / self._env.step_dt
        z = height_above_terrain(self._env, self.cfg.terrain_sensor)
        self.metrics["error_height"] += torch.abs(z - self.height_command) / steps

    def reset(self, env_ids):
        extras = super().reset(env_ids)
        self.phase[env_ids] = 0.0
        return extras

    def compute(self, dt: float) -> None:
        super().compute(dt)
        c = self.cfg
        h = self.height_command
        v = torch.norm(self.vel_command_b[:, :2], dim=-1)
        f_top = _interp(h, self._h, self._rate)
        f = c.duty * v / (c.stroke_share * _interp(h, self._h, self._stroke))
        # In place, so the buffers stay ordinary tensors under inference_mode (replay).
        self.step_rate.copy_(torch.minimum(f.clamp(min=c.min_step_rate), f_top.clamp(min=c.min_step_rate)))
        self.phase.add_(self.step_rate * dt).remainder_(1.0)


def speed_stages(env: "ManagerBasedRlEnv", env_ids: torch.Tensor, command_name: str,
                 stages: Sequence[tuple[int, float]]) -> dict[str, torch.Tensor]:
    """Curriculum: set ``speed_fraction`` from ``(env_step, fraction)`` stages.

    Fixed by step count (as mjlab's ``commands_vel``), so every variant sees
    the same schedule.
    """
    del env_ids
    term = env.command_manager.get_term(command_name)
    for step, frac in stages:
        if env.common_step_counter >= step:
            term.cfg.speed_fraction = float(frac)
    return {"speed_fraction": torch.tensor(term.cfg.speed_fraction)}


def commanded_height_exp(env: "ManagerBasedRlEnv", command_name: str, std: float,
                         asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """``exp(-((z - h_cmd) / std)^2)``, z the trunk height above the terrain."""
    term = env.command_manager.get_term(command_name)
    z = height_above_terrain(env, term.cfg.terrain_sensor, asset_cfg)
    h = term.height_command
    return torch.exp(-torch.square((z - h) / std))


def below_commanded_height(env: "ManagerBasedRlEnv", command_name: str, share: float,
                           floor: float,
                           asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminate when trunk height above the mean foot is below ``share * h_cmd``
    (and never above ``floor``), as ``root_height_above_feet_below``."""
    asset = env.scene[asset_cfg.name]
    root_z = asset.data.root_link_pos_w[:, 2]
    feet_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2].mean(dim=-1)
    h = env.command_manager.get_term(command_name).height_command
    return (root_z - feet_z) < torch.clamp(share * h, min=floor)
