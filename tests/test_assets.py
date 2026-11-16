"""The vendored Menagerie files are unmodified, and the Isaac Lab setup is applied in memory."""

import hashlib
import math

import mujoco
import numpy as np
import yaml

from allegro_repose.assets import ISAACLAB_JOINT_ORDER, make_hand_spec_fn, palm_pose
from allegro_repose.config import resolve_path

ASSET_DIR = resolve_path("assets/menagerie_wonik_allegro")


def _task():
    with open(resolve_path("configs/task/allegro_repose.yaml")) as f:
        return yaml.safe_load(f)


def test_vendored_files_match_checksums():
    for line in (ASSET_DIR / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        path = resolve_path(name)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name


def test_hand_has_isaaclab_setup():
    hand = _task()["hand"]
    m = make_hand_spec_fn(hand)().compile()
    names = [m.joint(i).name for i in range(m.njnt)]
    assert sorted(names) == sorted(ISAACLAB_JOINT_ORDER)
    for i, n in enumerate(names):
        lo, hi = hand["joint_limits_deg"][n]
        assert math.isclose(m.jnt_range[i, 0], math.radians(lo), abs_tol=1e-9)
        assert math.isclose(m.jnt_range[i, 1], math.radians(hi), abs_tol=1e-9)
    assert (m.actuator_gainprm[:, 0] == hand["stiffness"]).all()
    assert (m.actuator_biasprm[:, 1] == -hand["stiffness"]).all()
    assert (m.actuator_biasprm[:, 2] == -hand["damping"]).all()
    assert (m.actuator_forcerange == [-hand["effort_limit"], hand["effort_limit"]]).all()
    assert (m.dof_frictionloss == hand["joint_friction"]).all() and (m.dof_damping == 0).all()
    assert (m.body_gravcomp[1:] == 1).all()
    assert m.nexclude == 5  # the palm / finger-base exclusions survive the renaming


def test_palm_faces_up_and_fingers_point_along_minus_y():
    _, quat = palm_pose(_task()["hand"])
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(quat))
    mat = mat.reshape(3, 3)
    assert mat[:, 0] @ [0, 0, 1] > 0.99   # palm normal (Menagerie +x) up
    assert mat[:, 2] @ [0, -1, 0] > 0.99  # fingers (Menagerie +z) along -y
