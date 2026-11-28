"""Hexapod EntityCfg: the quadruped's, with the stance read from a hexapod leg.

Actuators (DcMotor torque-speed groups, PD gains per N·m), foot-only collision
and the robot spec are the quadruped task's own, imported unchanged, so the
hexapod trains under the same actuator model as the M0 control.
"""

from __future__ import annotations

from pathlib import Path

from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from draft.tasks.hexapod.legs import default_stance
from draft.tasks.quadruped.quadruped_entity import (
    _TERRAIN_COLLISION,  # matches .*_lower_leg_link_sphere: all six feet
    _actuator_cfgs_from_xml,
    _build_robot_spec,
)
from draft.tasks.quadruped.speed_limits import _resolve_robot_xml


def make_hexapod_entity_cfg(xml_path: Path | str, init_height: float,
                            geom_group: int | None = None,
                            joint_pos: dict[str, float] | None = None) -> EntityCfg:
    """As ``make_quadruped_entity_cfg``, for six legs. ``joint_pos`` overrides the nominal
    crouch (the default pose: the reset pose and the actions' zero)."""
    xml_path = _resolve_robot_xml(Path(xml_path))
    captured = xml_path
    return EntityCfg(
        spec_fn=lambda: _build_robot_spec(captured, geom_group),
        articulation=EntityArticulationInfoCfg(
            actuators=_actuator_cfgs_from_xml(xml_path),
            soft_joint_pos_limit_factor=0.9,
        ),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, init_height),
            joint_pos=joint_pos if joint_pos is not None else default_stance(xml_path),
            joint_vel={".*": 0.0},
        ),
        collisions=(_TERRAIN_COLLISION,),
    )
