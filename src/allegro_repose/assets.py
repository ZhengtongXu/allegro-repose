"""Scene assets: the Allegro hand (Isaac Lab setup on the Menagerie model) and the cube.

The Menagerie MJCF is loaded unmodified from ``assets/menagerie_wonik_allegro``. The Isaac Lab
Allegro configuration is applied to the in-memory ``MjSpec``:

* bodies and joints renamed to the Isaac Lab names (``palm_link``, ``index_link_0`` ...,
  ``index_joint_0`` ...);
* Isaac Lab joint limits, implicit PD actuators (stiffness 3, damping 0.1, effort 0.5 N m),
  joint friction 0.01 N m, no other joint damping, and a small armature for MuJoCo's explicit
  integration of the PD stiffness (``hand.armature``);
* gravity compensation on every hand body (Isaac Lab spawns the hand with gravity disabled).

The palm is mounted where the Isaac Lab asset places its palm (``palm_pose``).
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from allegro_repose.config import resolve_path

CUBE_BODY = "cube"
CUBE_GEOM = "cube_geom"

# Menagerie prefix -> Isaac Lab finger name; Menagerie segment -> Isaac Lab link index.
FINGERS = {"ff": "index", "mf": "middle", "rf": "ring", "th": "thumb"}
SEGMENTS = {"base": 0, "proximal": 1, "medial": 2, "distal": 3}

# Isaac Lab's Direct task actuation order (ALLEGRO_ACTUATED_JOINT_NAMES); the MJCF order differs.
ISAACLAB_JOINT_ORDER = (
    "index_joint_0", "middle_joint_0", "ring_joint_0", "thumb_joint_0",
    "index_joint_1", "index_joint_2", "index_joint_3",
    "middle_joint_1", "middle_joint_2", "middle_joint_3",
    "ring_joint_1", "ring_joint_2", "ring_joint_3",
    "thumb_joint_1", "thumb_joint_2", "thumb_joint_3",
)
FINGERTIP_BODIES = ("index_link_3", "middle_link_3", "ring_link_3", "thumb_link_3")

# Palm frame of the Isaac Lab Allegro asset in its root frame (fixed mount joint of the asset).
_ISAAC_PALM_IN_ROOT_POS = (-0.008219, -0.02063, 0.08086)
_ISAAC_PALM_IN_ROOT_QUAT = (0.6532803, 0.2705985, -0.6532827, 0.27059752)
# Menagerie palm axes expressed in the Isaac Lab palm frame: the fingers point along Menagerie +z
# and Isaac Lab +x, the finger row runs along +y / -y, the palm normal along +x / +z. The origins
# are matched at the middle finger's base joint (index and ring then agree within 2 mm; the thumb
# base of the Isaac Lab asset sits 4.4 cm closer to the wrist).
_MENAGERIE_TO_ISAAC_PALM_R = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])
_ISAAC_MIDDLE_BASE = np.array([0.0537375, 0.0087771, -0.0113])
_MENAGERIE_MIDDLE_BASE = np.array([0.0, 0.0, 0.0007])


def _quat_to_mat(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q / np.linalg.norm(q))
    return m.reshape(3, 3)


def _mat_to_quat(m: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(m, dtype=np.float64).flatten())
    return q


def palm_pose(hand_cfg: dict) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """World pose (pos, wxyz) of the Menagerie palm frame for Isaac Lab's root pose."""
    r_root = _quat_to_mat(hand_cfg["root_quat"])
    r_rp = _quat_to_mat(_ISAAC_PALM_IN_ROOT_QUAT)
    t_mi = _ISAAC_MIDDLE_BASE - _MENAGERIE_TO_ISAAC_PALM_R @ _MENAGERIE_MIDDLE_BASE
    rot = r_root @ r_rp @ _MENAGERIE_TO_ISAAC_PALM_R
    pos = np.asarray(hand_cfg["root_pos"]) + r_root @ (np.asarray(_ISAAC_PALM_IN_ROOT_POS) + r_rp @ t_mi)
    return tuple(float(x) for x in pos), tuple(float(x) for x in _mat_to_quat(rot))


def isaaclab_name(menagerie_name: str) -> str:
    """ff_base -> index_link_0, ffj2 -> index_joint_2, ff_tip -> index_tip, palm -> palm_link."""
    if menagerie_name == "palm":
        return "palm_link"
    finger = FINGERS[menagerie_name[:2]]
    rest = menagerie_name[2:]
    if rest.startswith("j"):
        return f"{finger}_joint_{int(rest[1:])}"
    seg = rest.lstrip("_")
    return f"{finger}_tip" if seg == "tip" else f"{finger}_link_{SEGMENTS[seg]}"


def make_hand_spec_fn(hand_cfg: dict):
    mjcf = resolve_path(hand_cfg["mjcf"])
    limits = {k: (math.radians(lo), math.radians(hi)) for k, (lo, hi) in hand_cfg["joint_limits_deg"].items()}
    kp, kd = float(hand_cfg["stiffness"]), float(hand_cfg["damping"])
    effort, friction = float(hand_cfg["effort_limit"]), float(hand_cfg["joint_friction"])
    armature = float(hand_cfg.get("armature", 0.0))

    def spec_fn() -> mujoco.MjSpec:
        spec = mujoco.MjSpec.from_file(str(mjcf))
        for body in spec.bodies[1:]:
            body.name = isaaclab_name(body.name)
            if hand_cfg.get("gravity_compensation", True):
                body.gravcomp = 1.0
        for ex in spec.excludes:  # palm / finger-base contact exclusions refer to bodies by name
            ex.bodyname1, ex.bodyname2 = isaaclab_name(ex.bodyname1), isaaclab_name(ex.bodyname2)
        palm = spec.body("palm_link")
        palm.quat = [1.0, 0.0, 0.0, 0.0]  # the scene places the palm frame itself (palm_pose)
        for joint in spec.joints:
            joint.name = isaaclab_name(joint.name)
            lo, hi = limits[joint.name]
            joint.range = [lo, hi]
            joint.damping = [0.0, 0.0, 0.0]  # (MuJoCo 3.10: polynomial coefficients)
            joint.frictionloss = friction
            joint.armature = armature
        for act in spec.actuators:
            act.name = act.target = isaaclab_name(act.target)
            lo, hi = limits[act.target]
            act.ctrlrange = [lo, hi]
            act.gainprm[0] = kp
            act.biasprm[0], act.biasprm[1], act.biasprm[2] = 0.0, -kp, -kd
            act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
            act.forcerange = [-effort, effort]
        return spec

    return spec_fn


GOAL_MARKER_BODY = "goal_marker"
# Visual only. Isaac Lab draws its marker at (-0.2, -0.45, 0.68), which hides the hand in our camera view;
# here it sits beside the hand at the same height.
GOAL_MARKER_POS = (0.20, -0.19, 0.56)


def _face_markers(body, half: float) -> None:
    """Coloured face markers (visual only) so the orientation is visible in renders."""
    colors = [(0.9, 0.1, 0.1), (0.1, 0.8, 0.1), (0.1, 0.3, 0.9)]
    for axis in range(3):
        for sign in (1, -1):
            pos = [0.0, 0.0, 0.0]
            pos[axis] = sign * (half + 0.0005)
            size = [half * 0.6] * 3
            size[axis] = 0.0005
            c = colors[axis] if sign > 0 else tuple(0.5 * x for x in colors[axis])
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size, rgba=[*c, 1.0],
                          contype=0, conaffinity=0, mass=0.0, density=0.0, group=2)


def make_cube_spec_fn(cube_cfg: dict, goal_marker: bool = False):
    """The cube; with ``goal_marker`` also a visual-only mocap copy of it that shows the goal
    orientation (for rendering)."""
    half = float(cube_cfg["half_size"])
    mass = float(cube_cfg["density"]) * (2 * half) ** 3

    def spec_fn() -> mujoco.MjSpec:
        spec = mujoco.MjSpec()
        body = spec.worldbody.add_body(name=CUBE_BODY)
        body.add_freejoint(name="cube_joint")
        i = mass * (2 * half) ** 2 / 6.0  # solid box
        body.explicitinertial = True
        body.mass = mass
        body.inertia = [i, i, i]
        g = body.add_geom(name=CUBE_GEOM, type=mujoco.mjtGeom.mjGEOM_BOX, size=[half] * 3,
                          rgba=[0.85, 0.55, 0.15, 1.0], mass=0.0, density=0.0)
        g.friction = list(cube_cfg["friction"])
        g.condim = int(cube_cfg["condim"])
        g.priority = 1
        _face_markers(body, half)
        if goal_marker:
            ghost = spec.worldbody.add_body(name=GOAL_MARKER_BODY, mocap=True, pos=list(GOAL_MARKER_POS))
            ghost.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[half] * 3, rgba=[0.85, 0.55, 0.15, 0.6],
                           contype=0, conaffinity=0, mass=0.0, density=0.0, group=2)
            _face_markers(ghost, half)
        return spec

    return spec_fn


def cube_mass(cube_cfg: dict) -> float:
    return float(cube_cfg["density"]) * (2 * float(cube_cfg["half_size"])) ** 3
