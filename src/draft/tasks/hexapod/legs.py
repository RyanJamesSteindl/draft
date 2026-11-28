"""Six-leg names and the geometry the hexapod task reads from a model.

The quadruped task hard-codes four legs ``fl/fr/rl/rr`` (``quadruped/base_env.py``,
``quadruped/quadruped_entity.py``, ``quadruped/speed_limits.py``). These are the
same measurements made on a hexapod's legs, named ``{left,right}_{front,mid,rear}``
so Draft's ``left_``/``right_`` mirror builds the right side. Each function
mirrors the quadruped one named in its docstring; only the leg names differ.

Kept free of mjlab, like ``quadruped/speed_limits.py``.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import mujoco
import numpy as np

from draft.tasks.quadruped.speed_limits import REF_LEG_LENGTH, _resolve_robot_xml

#: Order sets the order of per-foot observations and sensors.
LEGS = ("left_front", "right_front", "left_mid", "right_mid", "left_rear", "right_rear")
#: The leg whose dimensions stand for all six (they are identical by construction).
REF_LEG = "left_front"

FOOT_GEOMS = tuple(f"{leg}_lower_leg_link_sphere" for leg in LEGS)
FOOT_SITES = tuple(f"{leg}_foot" for leg in LEGS)
UPPER_LEG_BODIES = tuple(f"{leg}_upper_leg_link" for leg in LEGS)

# As quadruped_entity: swing room kept from each soft limit, and the nominal crouch.
_STANCE_HEADROOM: float = 0.20
_NOMINAL_CROUCH: float = 1.0
# As quadruped_entity: foot-tip speed -> body speed factor, for the fallback.
_GAIT_EFFICIENCY: float = 0.25


def _id(model: mujoco.MjModel, obj: mujoco.mjtObj, name: str) -> int:
    i = mujoco.mj_name2id(model, obj, name)
    if i < 0:  # mj_name2id returns -1, which indexes the LAST element silently
        raise ValueError(f"{name} not found in the model; is this a hexapod design?")
    return i


def _load(xml_path: Path | str) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(_resolve_robot_xml(Path(xml_path))))


def measure_leg_length(xml_path: Path | str) -> float:
    """Thigh + shank of the reference leg (as ``measure_quadruped_leg_length``)."""
    model = _load(xml_path)
    lower = _id(model, mujoco.mjtObj.mjOBJ_BODY, f"{REF_LEG}_lower_leg_link")
    foot = _id(model, mujoco.mjtObj.mjOBJ_SITE, f"{REF_LEG}_foot")
    return float(np.linalg.norm(model.body_pos[lower]) + np.linalg.norm(model.site_pos[foot]))


def measure_foot_sphere_radius(xml_path: Path | str) -> float:
    """Foot sphere radius of the reference leg (as ``quadruped_entity``'s)."""
    model = _load(xml_path)
    return float(model.geom_size[_id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                     f"{REF_LEG}_lower_leg_link_sphere"), 0])


def measure_max_body_speed(xml_path: Path | str) -> float:
    """Knee no-load speed x shank x 0.25 (as ``quadruped_entity``'s).

    Reads ``parameters_from_gui.yaml`` beside the XML, which
    ``scripts/generate_design.py`` writes; else ``1.35 * sqrt(s)``.
    """
    import yaml
    xml_path = _resolve_robot_xml(Path(xml_path))
    params_yaml = xml_path.parent / "parameters_from_gui.yaml"
    if not params_yaml.exists():
        return 1.35 * math.sqrt(measure_leg_length(xml_path) / REF_LEG_LENGTH)
    params = yaml.safe_load(params_yaml.read_text())
    knee_cls = params.get("knee_mot", "L")
    knee_vel = float(params.get(f"{knee_cls}_motor_velocity", 35.0))
    lower_leg = float(params.get("lower_leg_link_length", REF_LEG_LENGTH / 2))
    return knee_vel * lower_leg * _GAIT_EFFICIENCY


def _soft_range(model: mujoco.MjModel, joint: str, factor: float = 0.9) -> tuple[float, float]:
    j = _id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    lo, hi = (float(x) for x in model.jnt_range[j])
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo) * factor
    return mid - half, mid + half


def stance_from_feet(xml_path: Path | str, feet: dict[str, list[float]], height: float) -> dict[str, float]:
    """Joint angles that put each foot at ``feet[leg]`` (x, y in the trunk frame, metres) with the
    trunk ``height`` above flat ground, by damped least squares from the nominal crouch.

    Used when a project supplies stance centres (hexapod-sim's foot circle, 28 Nov 2026) instead of
    the nominal crouch. Raises if a foot is out of reach by more than 5 mm.
    """
    model = _load(xml_path)
    data = mujoco.MjData(model)
    crouch = default_stance(xml_path)
    out: dict[str, float] = {}
    torso = model.body(1)
    for leg in LEGS:
        names = [f"{leg}_hip_yaw", f"{leg}_hip_pitch", f"{leg}_knee"]
        jid = [_id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
        qadr = [model.jnt_qposadr[j] for j in jid]
        dadr = [model.jnt_dofadr[j] for j in jid]
        lo = np.array([model.jnt_range[j, 0] for j in jid]); hi = np.array([model.jnt_range[j, 1] for j in jid])
        site = _id(model, mujoco.mjtObj.mjOBJ_SITE, f"{leg}_foot")
        radius = float(model.geom_size[_id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{leg}_lower_leg_link_sphere"), 0])
        target = np.array([feet[leg][0], feet[leg][1], -height + radius])
        data.qpos[:] = model.qpos0
        q = np.array([0.0, crouch[".*hip_pitch"], crouch[".*knee"]])
        jac = np.zeros((3, model.nv))
        err = np.inf
        for _ in range(200):
            data.qpos[qadr] = q
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            e = target - (data.site_xpos[site] - data.xpos[torso.id])
            err = float(np.linalg.norm(e))
            if err < 1e-5:
                break
            mujoco.mj_jacSite(model, data, jac, None, site)
            J = jac[:, dadr]
            q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), e), lo, hi)
        if err > 5e-3:
            raise ValueError(f"{leg}: stance foot {feet[leg]} at height {height} is {err * 1000:.0f} mm out of reach")
        out.update({n: float(v) for n, v in zip(names, q)})
    return out


def default_stance(xml_path: Path | str) -> dict[str, float]:
    """The nominal crouch ``knee = -k``, ``hip_pitch = k/2`` (as ``quadruped_entity``'s)."""
    model = _load(xml_path)
    knee_lo, knee_hi = _soft_range(model, f"{REF_LEG}_knee")
    hip_lo, hip_hi = _soft_range(model, f"{REF_LEG}_hip_pitch")
    k_min = max(0.0, -knee_hi + _STANCE_HEADROOM, 2.0 * (hip_lo + _STANCE_HEADROOM))
    k_max = min(-knee_lo - _STANCE_HEADROOM, 2.0 * (hip_hi - _STANCE_HEADROOM))
    if k_max <= k_min:
        k = max(0.5 * (k_min + k_max), 0.0)
        warnings.warn(f"knee/hip ranges leave no stance headroom; using k={k:.2f}",
                      UserWarning, stacklevel=2)
    else:
        k = float(min(max(_NOMINAL_CROUCH, k_min), k_max))
    return {".*hip_pitch": 0.5 * k, ".*knee": -k, ".*hip_roll": 0.0}
