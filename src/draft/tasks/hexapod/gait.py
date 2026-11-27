"""Tripod gait: a clock the policy observes, and a reward for following it.

The stride phase runs from 0 to 1 at ``frequency_hz``, counted from the start of
each episode. Tripod A (left front, right mid, left rear) should be on the
ground in the first half of the stride and tripod B (right front, left mid,
right rear) in the second, each lifting while the other stands. Feet are not
scored within ``tolerance`` of a switch, so touchdown and lift-off need not be
instantaneous. When the command is (near) zero, all six feet should be down.

The policy sees ``sin, cos`` of the phase, so it knows where in the stride it is.

``duty`` (default 0.5, M1's schedule) is each tripod's share of the stride on
the ground. Above 0.5 the tripods overlap: both are down for ``duty - 0.5`` of
the stride around each changeover, centred on it. ``min_support`` (default 0,
off) scores a step 0 whenever fewer feet than that are down while moving.

``touchdown_ground_speed`` is a cost on feet close to the ground but not yet
(or no longer) in contact: their ground-frame velocity should match a small
backward sweep, ``-alpha * v_cmd``, so a foot lands already moving with the
ground (swing-leg retraction; Seyfarth et al. 2003) instead of braking the
body, and its downward speed should not exceed ``v_land``.

``frequency_hz=None`` takes the phase from the command term instead
(``command.py``'s ``HeightVelocityCommand``, which integrates a step rate tied
to the commanded speed and height); ``clock_command`` names that term.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from draft.tasks.hexapod.legs import LEGS

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

TRIPOD_A = ("left_front", "right_mid", "left_rear")
TRIPOD_B = ("right_front", "left_mid", "right_rear")
#: 1 where a foot (in LEGS order, the contact sensor's order) belongs to tripod A.
_IN_A = torch.tensor([leg in TRIPOD_A for leg in LEGS])


def gait_phase(env: "ManagerBasedRlEnv", frequency_hz: float | None,
               clock_command: str = "twist") -> torch.Tensor:
    """Stride phase in [0, 1), per env: elapsed time x ``frequency_hz``, or the
    command term's integrated phase when ``frequency_hz`` is None."""
    if frequency_hz is None:
        return env.command_manager.get_term(clock_command).phase
    t = env.episode_length_buf.float() * env.step_dt
    return torch.remainder(t * frequency_hz, 1.0)


def gait_clock(env: "ManagerBasedRlEnv", frequency_hz: float | None,
               clock_command: str = "twist") -> torch.Tensor:
    """Observation: ``[sin 2*pi*phase, cos 2*pi*phase]``, shape [B, 2]."""
    ang = 2.0 * math.pi * gait_phase(env, frequency_hz, clock_command)
    return torch.stack((torch.sin(ang), torch.cos(ang)), dim=-1)


def tripod_contact_match(
    env: "ManagerBasedRlEnv",
    sensor_name: str,
    frequency_hz: float | None,
    command_name: str,
    command_threshold: float,
    tolerance: float = 0.05,
    duty: float = 0.5,
    min_support: int = 0,
) -> torch.Tensor:
    """Reward in [0, 1]: fraction of feet whose contact matches the tripod schedule."""
    found = env.scene.sensors[sensor_name].data.found
    contact = found.reshape(found.shape[0], -1) > 0                     # [B, 6]
    phase = gait_phase(env, frequency_hz, command_name).unsqueeze(-1)                 # [B, 1]
    in_a = _IN_A.to(contact.device).unsqueeze(0)                        # [1, 6]
    # Stance window per foot: starts at -o (tripod A) or 0.5 - o (B), lasts `duty`.
    o = (duty - 0.5) / 2
    start = torch.where(in_a, torch.tensor(-o), torch.tensor(0.5 - o)).to(phase.device)
    x = torch.remainder(phase - start, 1.0)                              # [B, 6] time since touchdown
    want_down = x < duty

    cmd = env.command_manager.get_command(command_name)
    moving = (torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])) > command_threshold
    want_down = torch.where(moving.unsqueeze(-1), want_down, torch.ones_like(want_down))

    # Within `tolerance` of a foot's own touchdown or lift-off either state is fine.
    to_event = torch.minimum(torch.minimum(x, 1.0 - x), torch.abs(x - duty))
    near_switch = (to_event < tolerance) & moving.unsqueeze(-1)
    match = (contact == want_down) | near_switch
    score = match.float().mean(dim=-1)
    if min_support > 0:
        short = moving & (contact.sum(dim=-1) < min_support)
        score = torch.where(short, torch.zeros_like(score), score)
    return score


def touchdown_ground_speed(
    env: "ManagerBasedRlEnv",
    sensor_name: str,
    height_sensor_name: str,
    command_name: str,
    foot_sites: tuple[str, ...],
    foot_radius: float,
    window: float = 0.03,
    alpha: float = 0.1,
    v_land: float = 0.4,
) -> torch.Tensor:
    """Cost: feet within ``window`` of the ground and not in contact should move
    with the ground, ``v_xy = -alpha * v_cmd`` (world axes), and come down no
    faster than ``v_land``. Summed over feet; use a negative weight."""
    asset = env.scene["robot"]
    ids = [asset.site_names.index(n) for n in foot_sites]
    found = env.scene.sensors[sensor_name].data.found
    contact = found.reshape(found.shape[0], -1) > 0                              # [B, F]
    h = env.scene.sensors[height_sensor_name].data.heights - foot_radius          # [B, F]
    near = (~contact) & (h < window)
    v = asset.data.site_lin_vel_w[:, ids, :]                                      # [B, F, 3]
    cmd = env.command_manager.get_command(command_name)[:, :2]                    # body frame
    q = asset.data.root_link_quat_w
    yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    c, s_ = torch.cos(yaw), torch.sin(yaw)
    cmd_w = torch.stack((c * cmd[:, 0] - s_ * cmd[:, 1], s_ * cmd[:, 0] + c * cmd[:, 1]), dim=-1)
    target = -alpha * cmd_w.unsqueeze(1)                                          # [B, 1, 2]
    e_xy = torch.square(v[..., :2] - target).sum(-1)
    e_z = torch.square(torch.clamp(-v[..., 2] - v_land, min=0.0))
    cost = (near.float() * (e_xy + e_z)).sum(-1)
    n = near.sum().clamp(min=1)
    env.extras.setdefault("log", {})["Metrics/touchdown_speed_error"] = (
        (near.float() * e_xy.sqrt()).sum() / n)
    return cost
