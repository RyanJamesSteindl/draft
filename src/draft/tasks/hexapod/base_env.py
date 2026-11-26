"""Base environment for the hexapod locomotion task.

``BaseCtx`` and ``make_base_env_cfg`` are ``quadruped/base_env.py``'s, copied
with only the leg-specific parts changed: the six-leg foot geoms, foot sites and
upper-leg bodies (``legs.py``), leg geometry measured on a hexapod leg, and the
hexapod entity. Terrain, sensors, rewards, terminations, randomisation and the
Froude scaling are unchanged, and the leg-free helpers and constants are
imported from the quadruped module, so the hexapod trains under the same recipe
as the M0 quadruped control. Diff the two functions to see every change.
"""

from __future__ import annotations

import math
from dataclasses import fields, replace
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
    ContactMatch, ContactSensorCfg, ObjRef, RayCastSensorCfg, RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise

import draft.tasks.hexapod.command as cmd_mod
import draft.tasks.hexapod.gait as gait_mod
import draft.tasks.quadruped.mdp as q_mdp
from draft.tasks.hexapod.hexapod_entity import make_hexapod_entity_cfg
from draft.tasks.hexapod.legs import (
    FOOT_GEOMS, FOOT_SITES, UPPER_LEG_BODIES,
    measure_foot_sphere_radius, measure_leg_length, measure_max_body_speed,
)
from draft.tasks.quadruped.base_env import (  # leg-free: shared unchanged
    _ACTION_SCALE, _BASE_NOISE, _BASE_PUSH_SCALE, _BASE_STEP_H, _BASE_TERRAIN_LEVEL,
    _BASE_WAVE, _FOOT_SCAN_RING_M, _JOINT_VEL_NOISE_FRAC, _VIEW_DISTANCE_PER_LEG,
    DEFAULT_TERRAIN_MIX, TERRAIN_SCAN_FIXES, _lin_vel_z_l2, apply_terrain_ceiling,
    apply_terrain_mix,
)
from draft.tasks.quadruped.quadruped_entity import (
    REF_INIT_HEIGHT, REF_LEG_LENGTH, measure_max_joint_speed,
)
from draft.tasks.quadruped.runner_cfg import QuadrupedLocomotionRunnerCfg

#: Feet the quadruped recipe's per-foot reward weights were calibrated on.
_QUADRUPED_FEET = 4

#: ``free``: no gait is asked for. ``tripod``: see gait.py.
GAITS = ("free", "tripod")


class BaseCtx:
    """Scale factors + reusable ranges shared with the task variants.

    The task is scaled off the measured geometry of ``task_geometry_xml``
    (default: the robot itself). Pointing a twin and its original at one model
    makes the task identical, leaving the robot as the only variable.
    """

    def __init__(self, robot_xml: Path | str, task_geometry_xml: Path | str | None = None,
                 v_max_override: float | None = None):
        geom_xml = task_geometry_xml if task_geometry_xml is not None else robot_xml
        self.leg_length = measure_leg_length(geom_xml)
        self.s = self.leg_length / REF_LEG_LENGTH
        self.s_sqrt = math.sqrt(self.s)
        self.v_max = measure_max_body_speed(geom_xml)
        self.omega_nl = measure_max_joint_speed(geom_xml)
        self.foot_radius = measure_foot_sphere_radius(geom_xml)
        self.hfield_scale = max(0.1, 2.0 * self.foot_radius)
        self.init_height = REF_INIT_HEIGHT * self.s
        if v_max_override is not None:
            # measure_max_body_speed is a heuristic; a design may state its
            # measured ceiling instead.
            self.v_max = float(v_max_override)
        self.terrain_mix = DEFAULT_TERRAIN_MIX  # set by make_base_env_cfg
        self.command_clamped = False           # set by make_base_env_cfg
        self.dimensional_weights = False       # set by make_base_env_cfg
        self.terrain_scan_fix = "all"          # set by make_base_env_cfg
        self.episode_length_s = 0.0            # set by make_base_env_cfg
        self.joint_vel_noise = 0.0             # set by make_base_env_cfg
        s_sqrt = self.s_sqrt
        self.push_base_range: dict[str, tuple[float, float]] = {
            "x":     (-0.5 * s_sqrt, 0.5 * s_sqrt),
            "y":     (-0.5 * s_sqrt, 0.5 * s_sqrt),
            "z":     (-0.4 * s_sqrt, 0.4 * s_sqrt),
            "roll":  (-0.52 / s_sqrt, 0.52 / s_sqrt),
            "pitch": (-0.52 / s_sqrt, 0.52 / s_sqrt),
            "yaw":   (-0.78 / s_sqrt, 0.78 / s_sqrt),
        }


def make_base_env_cfg(
    *,
    robot_xml: Path | str,
    n_envs: int = 4096,
    episode_length_s: float = 20.0,
    play: bool = False,
    task_geometry_xml: Path | str | None = None,
    terrain_mix: str = DEFAULT_TERRAIN_MIX,
    clamp_command_to_vmax: bool = False,
    v_max_override: float | None = None,
    terrain_scale_with_size: bool = False,
    dimensional_weights: bool = False,
    terrain_scan_fix: str = "all",
    per_foot_weights_as_quadruped: bool = False,
    gait: str = "free",
    gait_frequency_hz: float | str = 1.8,
    gait_reward_weight: float = 1.0,
    height_range: tuple[float, float] | None = None,
    speed_table: dict | None = None,
    speed_cap_share: float = 0.8,
    speed_stages: list[tuple[int, float]] | None = None,
) -> tuple[ManagerBasedRlEnvCfg, BaseCtx]:
    """Shared gentle-terrain natural-gait env cfg + a context object.

    ``task_geometry_xml``: see :class:`BaseCtx`. ``terrain_mix``: see
    :data:`TERRAIN_MIXES`. ``clamp_command_to_vmax`` caps the velocity command at
    ``ctx.v_max`` (the Froude range need not be reachable by the actuators).
    ``dimensional_weights`` rescales the ``air_time`` and ``action_rate_l2``
    weights to cancel their cost's dimensions. ``terrain_scan_fix``: see
    :data:`TERRAIN_SCAN_FIXES`. ``per_foot_weights_as_quadruped`` scales
    the rewards summed over feet (air_time, foot_clearance) by
    ``_QUADRUPED_FEET / len(FOOT_SITES)``, so six feet earn the same total
    incentive four feet do; off, each foot earns the quadruped's per-foot
    weight. ``gait``: ``free`` lets the gait emerge; ``tripod`` adds a stride
    clock (``gait_frequency_hz``) to the observations and rewards feet for
    following the tripod schedule (``gait.py``, weight ``gait_reward_weight``).

    M3 (all off by default, so M1 configs are unchanged): ``height_range``
    adds a body-height command (``command.py``) that the policy observes, the
    trunk-height reward follows it (weight 1.0, std 0.08 m x s) and the
    too-low termination becomes half the commanded height (floor 0.12 m x s).
    ``speed_table`` (``heights``, ``top_speed``, ``stroke``, ``step_rate``;
    from the hexapod-sim M2 speed tool) caps the planar speed command at
    ``speed_cap_share`` of the top speed at the commanded height.
    ``speed_stages``: ``[iteration, fraction of the cap]`` pairs, raised in
    order. ``gait_frequency_hz: speed`` ties the tripod clock to the commanded
    speed and height (needs ``height_range``).
    Settings that change the task must match across
    any runs being compared.
    """
    if terrain_scan_fix not in TERRAIN_SCAN_FIXES:
        raise ValueError(f"unknown terrain_scan_fix {terrain_scan_fix!r}; "
                         f"known: {TERRAIN_SCAN_FIXES}")
    ctx = BaseCtx(robot_xml, task_geometry_xml, v_max_override)
    ctx.terrain_scan_fix = terrain_scan_fix
    ts = ctx.s if terrain_scale_with_size else 1.0
    s, s_sqrt = ctx.s, ctx.s_sqrt

    cfg = make_velocity_env_cfg()
    # init_height and the nominal stance also come from the task geometry.
    cfg.scene.entities = {"robot": make_hexapod_entity_cfg(
        robot_xml, init_height=ctx.init_height,
        geom_group=None if terrain_scan_fix == "none" else 1)}

    # ── Terrain: generator under the gentle ceiling, fixed level band ─────────
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "generator"
    gen = replace(ROUGH_TERRAINS_CFG)
    gen.curriculum = True
    gen.sub_terrains = apply_terrain_mix({
        # terrain_scale_with_size (off by default) scales roughness with leg
        # length; in practice it made the tall designs shuffle.
        name: apply_terrain_ceiling(st, _BASE_STEP_H * ts, _BASE_NOISE * ts,
                                    _BASE_WAVE * ts, ctx.hfield_scale)
        for name, st in gen.sub_terrains.items()
    }, terrain_mix)
    ctx.terrain_mix = terrain_mix
    cfg.scene.terrain.terrain_generator = gen
    # Fixed level band (see _BASE_TERRAIN_LEVEL); the terrain challenge resets it to 0.
    cfg.scene.terrain.max_init_terrain_level = _BASE_TERRAIN_LEVEL
    cfg.sim.nconmax = None

    # ── Sensors: terrain scan (torso) + foot height scan (feet) + contacts ────
    for sensor in cfg.scene.sensors:
        if sensor.name == "terrain_scan":
            assert isinstance(sensor, RayCastSensorCfg) and isinstance(sensor.frame, ObjRef)
            sensor.frame.name = "torso_link"
        if sensor.name == "foot_height_scan":
            assert isinstance(sensor, TerrainHeightSensorCfg)
            sensor.frame = tuple(ObjRef(type="site", name=s_, entity="robot") for s_ in FOOT_SITES)
            sensor.pattern = RingPatternCfg.single_ring(
                radius=_FOOT_SCAN_RING_M, num_samples=4)
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=FOOT_GEOMS, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"), reduce="netforce", num_slots=1, track_air_time=True,
    )
    upper_leg_ground_cfg = ContactSensorCfg(
        name="upper_leg_ground_contact",
        primary=ContactMatch(mode="body", pattern=UPPER_LEG_BODIES, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",), reduce="none", num_slots=1,
    )
    existing = tuple(s_ for s_ in cfg.scene.sensors
                     if s_.name not in {"feet_ground_contact", "upper_leg_ground_contact",
                                        "torso_ground_contact"})
    cfg.scene.sensors = existing + (feet_ground_cfg, upper_leg_ground_cfg)
    upright_scan = "terrain_scan"
    if terrain_scan_fix == "reward":
        # Policy scan sees groups 0 and 1 (robot too); upright reads a terrain-only copy.
        scan = next(s_ for s_ in cfg.scene.sensors if s_.name == "terrain_scan")
        scan.include_geom_groups = (0, 1)
        cfg.scene.sensors = cfg.scene.sensors + (
            replace(scan, name="terrain_scan_ground", include_geom_groups=(0,),
                    debug_vis=False),)
        upright_scan = "terrain_scan_ground"

    # joint_vel obs noise, sized to this design's joint-speed ceiling.
    _jv = _JOINT_VEL_NOISE_FRAC * ctx.omega_nl
    for group in cfg.observations.values():
        if "joint_vel" in group.terms:
            group.terms["joint_vel"].noise = Unoise(n_min=-_jv, n_max=_jv)
    ctx.joint_vel_noise = _jv

    # ── Actions ───────────────────────────────────────────────────────────────
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = _ACTION_SCALE

    # ── Velocity command: moderate walking range (variants may curriculum it) ─
    twist = cfg.commands["twist"]
    twist.ranges.lin_vel_x = (-1.0 * s_sqrt, 1.0 * s_sqrt)
    twist.ranges.lin_vel_y = (-0.6 * s_sqrt, 0.6 * s_sqrt)
    twist.ranges.ang_vel_z = (-0.5 / s_sqrt, 0.5 / s_sqrt)
    command_scale = 1.0
    if clamp_command_to_vmax:
        # Never ask for more body speed than the knee actuator can deliver.
        v = ctx.v_max
        wide = twist.ranges.lin_vel_x[1]
        twist.ranges.lin_vel_x = (max(-v, twist.ranges.lin_vel_x[0]),
                                  min(v, twist.ranges.lin_vel_x[1]))
        twist.ranges.lin_vel_y = (max(-0.6 * v, twist.ranges.lin_vel_y[0]),
                                  min(0.6 * v, twist.ranges.lin_vel_y[1]))
        # How much the clamp bit; the tracking-reward width follows it.
        command_scale = twist.ranges.lin_vel_x[1] / wide if wide > 0 else 1.0
    ctx.height_command = height_range is not None
    if height_range is not None:
        if speed_table is None:
            raise ValueError("height_range needs speed_table (the M2 limits by height)")
        stages = [(int(it) * QuadrupedLocomotionRunnerCfg.num_steps_per_env, float(fr))
                  for it, fr in (speed_stages or [(0, 1.0)])]
        old = cfg.commands["twist"]
        cfg.commands["twist"] = cmd_mod.HeightVelocityCommandCfg(
            **{f.name: getattr(old, f.name) for f in fields(old)},
            height_range=(float(height_range[0]), float(height_range[1])),  # metres, as M2
            table_heights=speed_table["heights"], table_top_speed=speed_table["top_speed"],
            table_stroke=speed_table["stroke"], table_step_rate=speed_table["step_rate"],
            cap_share=speed_cap_share, speed_fraction=1.0 if play else stages[0][1])
        ctx.speed_stages = stages
    elif gait_frequency_hz == "speed":
        raise ValueError("gait_frequency_hz: speed needs height_range (the clock lives there)")
    ctx.command_clamped = clamp_command_to_vmax
    ctx.dimensional_weights = dimensional_weights
    ctx.command_scale = command_scale

    # ── Rewards: the proven gait set (weights calibrated at s=1) ──────────────
    cfg.rewards["track_linear_velocity"].weight = 1.0
    # Tolerance is half the command range, so it scales with any clamp; otherwise
    # it can exceed the range and standing still would score well.
    cfg.rewards["track_linear_velocity"].params["std"] = (
        math.sqrt(0.25) * s_sqrt * command_scale)
    cfg.rewards["track_angular_velocity"].weight = 1.0
    cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(0.25) / s_sqrt

    # Upright relative to the terrain surface normal; stricter for taller robots.
    cfg.rewards["upright"].weight = 1.0
    cfg.rewards["upright"].params["std"] = math.sqrt(0.2) / s_sqrt
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso_link",)
    cfg.rewards["upright"].params["terrain_sensor_names"] = (upright_scan,)

    # Short, deliberate steps: reward airtime from 0 up to a modest Froude cap.
    cfg.rewards["air_time"].weight = 0.4 / (s_sqrt if dimensional_weights else 1.0)
    cfg.rewards["air_time"].params["sensor_name"] = "feet_ground_contact"
    cfg.rewards["air_time"].params["threshold_min"] = 0.0
    cfg.rewards["air_time"].params["threshold_max"] = 0.35 * s_sqrt
    # A speed threshold, so it scales like a speed.
    cfg.rewards["air_time"].params["command_threshold"] = 0.1 * s_sqrt

    cfg.rewards["foot_clearance"].weight = -2.0 / s**1.5
    # mjlab's feet_air_time and feet_clearance sum over feet; their weights
    # were calibrated on four.
    ctx.per_foot_scale = (_QUADRUPED_FEET / len(FOOT_SITES)
                          if per_foot_weights_as_quadruped else 1.0)
    cfg.rewards["air_time"].weight *= ctx.per_foot_scale
    cfg.rewards["foot_clearance"].weight *= ctx.per_foot_scale

    # ── Gait: free, or a tripod clock + contact-schedule reward ──────────────
    if gait not in GAITS:
        raise ValueError(f"unknown gait {gait!r}; known: {GAITS}")
    ctx.gait = gait
    if gait == "tripod":
        freq = None if gait_frequency_hz == "speed" else float(gait_frequency_hz)
        for group in ("actor", "critic"):
            cfg.observations[group].terms["gait_clock"] = ObservationTermCfg(
                func=gait_mod.gait_clock, params={"frequency_hz": freq})
        cfg.rewards["tripod_contact"] = RewardTermCfg(
            func=gait_mod.tripod_contact_match, weight=gait_reward_weight,
            params={"sensor_name": "feet_ground_contact",
                    "frequency_hz": freq, "command_name": "twist",
                    # As air_time's: below this the robot should stand still.
                    "command_threshold": 0.1 * s_sqrt})
    cfg.rewards["foot_clearance"].params["target_height"] = 0.10 * s
    cfg.rewards["foot_clearance"].params["height_sensor_name"] = "foot_height_scan"
    cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = FOOT_SITES

    cfg.rewards["dof_pos_limits"].weight = -1.0
    # dimensional_weights (off by default) makes the two weights that do not
    # cancel their cost's dimensions do so: action_rate_l2 cost ~ s^-1 (weight
    # x s), air_time cost ~ s^+0.5 (weight / sqrt(s)).
    cfg.rewards["action_rate_l2"].weight = -0.05 * (s if dimensional_weights else 1.0)
    cfg.rewards["lin_vel_z"] = RewardTermCfg(func=_lin_vel_z_l2, weight=-2.0 / s)
    cfg.rewards["upper_leg_contact"] = RewardTermCfg(
        func=mdp.self_collision_cost, weight=-1.0,
        params={"sensor_name": "upper_leg_ground_contact"},
    )
    # Mild trunk-height reward so tall designs stand rather than crawl.
    cfg.rewards["base_height"] = RewardTermCfg(
        func=q_mdp.base_height_exp, weight=0.4,
        params={"target_height": 0.95 * ctx.init_height, "std": 0.25 * ctx.init_height},
    )
    if height_range is not None:
        cfg.rewards["base_height"] = RewardTermCfg(
            func=cmd_mod.commanded_height_exp, weight=1.0,
            params={"command_name": "twist", "std": 0.08 * s})
    for key in ("pose", "angular_momentum", "body_ang_vel",
                "foot_swing_height", "foot_slip", "soft_landing"):
        cfg.rewards.pop(key, None)

    # ── Terminations: state-based falls (contacts can't fire) ─────────────────
    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=mdp.bad_orientation, params={"limit_angle": 1.0},
    )
    # Trunk above feet, not above world z=0, so terrain depressions are not falls.
    cfg.terminations["base_too_low"] = TerminationTermCfg(
        func=q_mdp.root_height_above_feet_below,
        params={"minimum_height": 0.5 * ctx.init_height,
                "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITES)},
    )
    if height_range is not None:
        cfg.terminations["base_too_low"] = TerminationTermCfg(
            func=cmd_mod.below_commanded_height,
            params={"command_name": "twist", "share": 0.5, "floor": 0.12 * s,
                    "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITES)},
        )
    # out_of_terrain_bounds kept from the base velocity cfg (terrain present).

    # ── Domain randomisation + modest stance-shaping push ─────────────────────
    ctx.base_curriculum = dict(cfg.curriculum)  # {terrain_levels, command_vel}
    cfg.events["foot_friction"] = EventTermCfg(
        mode="startup", func=dr.geom_friction,
        params={"asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOMS),
                "operation": "abs", "ranges": (0.4, 1.1), "shared_random": True},
    )
    cfg.events["base_com"] = EventTermCfg(
        mode="startup", func=dr.body_com_offset,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
                "operation": "add",
                "ranges": {0: (-0.02 * s, 0.02 * s), 1: (-0.02 * s, 0.02 * s),
                           2: (-0.02 * s, 0.02 * s)}},
    )
    ctx.modest_push_range = {k: (v[0] * _BASE_PUSH_SCALE, v[1] * _BASE_PUSH_SCALE)
                             for k, v in ctx.push_base_range.items()}
    cfg.events["push_robot"].params["velocity_range"] = ctx.modest_push_range
    # Push interval scales as sqrt(s), like gait-cycle time.
    cfg.events["push_robot"].interval_range_s = (2.0 * s_sqrt, 5.0 * s_sqrt)
    ctx.push_event = cfg.events["push_robot"]

    # ── Curriculum: none; the base is a fixed task ────────────────────────────
    # ``ctx.base_curriculum`` keeps mjlab's terms for the challenge tasks to reuse.
    cfg.curriculum = {}
    if height_range is not None:
        cfg.curriculum["speed_stages"] = CurriculumTermCfg(
            func=cmd_mod.speed_stages,
            params={"command_name": "twist", "stages": ctx.speed_stages})

    # ── Viewer / envs / episode length ────────────────────────────────────────
    # Camera tracks the torso from above at an oblique angle, distance in leg lengths.
    cfg.viewer.origin_type = cfg.viewer.OriginType.ASSET_BODY
    cfg.viewer.entity_name = "robot"
    cfg.viewer.body_name = "torso_link"
    cfg.viewer.distance = _VIEW_DISTANCE_PER_LEG * ctx.leg_length
    cfg.viewer.elevation = -30.0
    cfg.viewer.azimuth = 120.0
    cfg.scene.num_envs = n_envs
    # Episode length scales as sqrt(s): a fixed budget of gait cycles.
    cfg.episode_length_s = episode_length_s * s_sqrt
    ctx.episode_length_s = cfg.episode_length_s
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.terminations.pop("out_of_terrain_bounds", None)
        cfg.curriculum = {}
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
    else:
        cfg.episode_length_s = ctx.episode_length_s

    return cfg, ctx
