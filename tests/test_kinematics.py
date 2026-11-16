"""Fingertip forward kinematics (torch) against MuJoCo's own body poses and velocities."""

import mujoco
import numpy as np
import torch
import yaml

from allegro_repose.assets import FINGERTIP_BODIES, make_hand_spec_fn, palm_pose
from allegro_repose.config import resolve_path
from allegro_repose.task.kinematics import HandKinematics


def test_fingertip_kinematics_match_mujoco():
    with open(resolve_path("configs/task/allegro_repose.yaml")) as f:
        hand = yaml.safe_load(f)["hand"]
    pos, quat = palm_pose(hand)
    world = mujoco.MjSpec()
    frame = world.worldbody.add_frame(pos=pos, quat=quat)
    frame.attach_body(make_hand_spec_fn(hand)().body("palm_link"), "", "")
    m = world.compile()
    d = mujoco.MjData(m)
    names = [m.joint(i).name for i in range(m.njnt)]
    kin = HandKinematics(make_hand_spec_fn(hand)().compile(), names, list(FINGERTIP_BODIES), pos, quat, "cpu")
    rng = np.random.default_rng(0)
    for _ in range(5):
        d.qpos[:] = rng.uniform(m.jnt_range[:, 0], m.jnt_range[:, 1])
        d.qvel[:] = rng.normal(size=m.nv)
        mujoco.mj_forward(m, d)
        p, q, v, w = kin(torch.tensor(d.qpos, dtype=torch.float32)[None], torch.tensor(d.qvel, dtype=torch.float32)[None])
        for k, b in enumerate(FINGERTIP_BODIES):
            bid = m.body(b).id
            np.testing.assert_allclose(p[0, k].numpy(), d.xpos[bid], atol=1e-5)
            qq = d.xquat[bid] * np.sign(d.xquat[bid] @ q[0, k].numpy())
            np.testing.assert_allclose(q[0, k].numpy(), qq, atol=1e-5)
            vel = np.zeros(6)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, bid, vel, 0)  # world frame [rot, lin]
            np.testing.assert_allclose(w[0, k].numpy(), vel[:3], atol=1e-4)
            np.testing.assert_allclose(v[0, k].numpy(), vel[3:], atol=1e-4)
