"""Tripod gait: a clock the policy observes, and a reward for following it.

The stride phase runs from 0 to 1 at ``frequency_hz``, counted from the start of
each episode. Tripod A (left front, right mid, left rear) should be on the
ground in the first half of the stride and tripod B (right front, left mid,
right rear) in the second, each lifting while the other stands. Feet are not
scored within ``tolerance`` of a switch, so touchdown and lift-off need not be
instantaneous. When the command is (near) zero, all six feet should be down.

The policy sees ``sin, cos`` of the phase, so it knows where in the stride it is.
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


def gait_phase(env: "ManagerBasedRlEnv", frequency_hz: float) -> torch.Tensor:
    """Stride phase in [0, 1), per env, from the episode's elapsed time."""
    t = env.episode_length_buf.float() * env.step_dt
    return torch.remainder(t * frequency_hz, 1.0)


def gait_clock(env: "ManagerBasedRlEnv", frequency_hz: float) -> torch.Tensor:
    """Observation: ``[sin 2*pi*phase, cos 2*pi*phase]``, shape [B, 2]."""
    ang = 2.0 * math.pi * gait_phase(env, frequency_hz)
    return torch.stack((torch.sin(ang), torch.cos(ang)), dim=-1)


def tripod_contact_match(
    env: "ManagerBasedRlEnv",
    sensor_name: str,
    frequency_hz: float,
    command_name: str,
    command_threshold: float,
    tolerance: float = 0.05,
) -> torch.Tensor:
    """Reward in [0, 1]: fraction of feet whose contact matches the tripod schedule."""
    found = env.scene.sensors[sensor_name].data.found
    contact = found.reshape(found.shape[0], -1) > 0                     # [B, 6]
    phase = gait_phase(env, frequency_hz).unsqueeze(-1)                 # [B, 1]
    in_a = _IN_A.to(contact.device).unsqueeze(0)                        # [1, 6]
    first_half = phase < 0.5
    want_down = torch.where(in_a, first_half, ~first_half)              # [B, 6]

    cmd = env.command_manager.get_command(command_name)
    moving = (torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])) > command_threshold
    want_down = torch.where(moving.unsqueeze(-1), want_down, torch.ones_like(want_down))

    # Within `tolerance` of a switch (phase 0 or 0.5) either state is fine.
    d = torch.remainder(phase, 0.5)
    near_switch = (torch.minimum(d, 0.5 - d) < tolerance) & moving.unsqueeze(-1)
    match = (contact == want_down) | near_switch
    return match.float().mean(dim=-1)
