"""Fingertip forward kinematics of the fixed-base hand, in torch (differentiable, batched).

The Isaac Lab observation contains fingertip poses and velocities. With a fixed wrist they are
functions of the joint state only, so they are computed here from (q, qd) instead of being read
from the simulator, so the observation is a pure function of the physical state (``task/core.py``).
``tests/test_kinematics.py`` checks
the result against MuJoCo's ``xpos``/``xquat`` and body velocities.

All fingers are chains of the same depth, so they are computed together (link by link, the
fingers along a batch dimension).
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from allegro_repose import math3d as m3


class HandKinematics:
    """Pose and velocity of the end bodies of equal-depth hinge chains hanging off a fixed root."""

    def __init__(self, model: mujoco.MjModel, joint_names: list[str], bodies: list[str],
                 mount_pos, mount_quat, device):
        index = {n: i for i, n in enumerate(joint_names)}
        chains = []
        for name in bodies:
            chain, b = [], model.body(name).id
            while model.body_parentid[b] != 0:  # up to (excluding) the hand's root body
                jnt = [j for j in range(model.njnt) if model.jnt_bodyid[j] == b]
                assert len(jnt) == 1 and model.jnt_type[jnt[0]] == mujoco.mjtJoint.mjJNT_HINGE, "one hinge per link"
                chain.append((b, jnt[0]))
                b = model.body_parentid[b]
            chains.append(chain[::-1])
        assert len({len(c) for c in chains}) == 1, "chains must have the same depth"
        t = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=torch.float32, device=device)
        # [L, F, ...]: link l of finger f
        self.pos = t([[model.body_pos[b] for b, _ in links] for links in zip(*chains)])
        self.quat = t([[model.body_quat[b] for b, _ in links] for links in zip(*chains)])
        self.axis = t([[model.jnt_axis[j] for _, j in links] for links in zip(*chains)])
        self.jpos = t([[model.jnt_pos[j] for _, j in links] for links in zip(*chains)])
        self.qidx = torch.tensor([[index[model.joint(j).name] for _, j in links] for links in zip(*chains)],
                                 device=device)
        self.com = t([model.body_ipos[model.body(n).id] for n in bodies])  # [F, 3]
        self.mount_pos, self.mount_quat = t(mount_pos), t(mount_quat)
        self.nf = len(bodies)

    def __call__(self, q: torch.Tensor, qd: torch.Tensor, at_com: bool = True):
        """q, qd [..., NJ] -> frame pos [..., F, 3], quat [..., F, 4] (wxyz), centre-of-mass linear velocity
        [..., F, 3] and angular velocity [..., F, 3], all in the world frame (Isaac Lab's ``body_pos_w``,
        ``body_quat_w`` and ``body_vel_w``). ``at_com=False`` gives the linear velocity of the link
        frame origin instead."""
        lead, F = q.shape[:-1], self.nf
        e = lambda x: x.expand(*lead, *x.shape)
        p = e(self.mount_pos[None].expand(F, 3))
        r = e(self.mount_quat[None].expand(F, 4))
        axes, origins, rates = [], [], []
        for link in range(self.pos.shape[0]):
            p = p + m3.quat_rotate(r, e(self.pos[link]))
            r = m3.quat_mul(r, e(self.quat[link]))
            angle = q[..., self.qidx[link]]                       # [..., F]
            axis = e(self.axis[link])
            o = p + m3.quat_rotate(r, e(self.jpos[link]))           # point on the joint axis (world)
            axes.append(m3.quat_rotate(r, axis))
            origins.append(o)
            rates.append(qd[..., self.qidx[link]][..., None])
            r = m3.quat_mul(r, m3.quat_from_axis_angle(axis * angle[..., None]))
            p = o - m3.quat_rotate(r, e(self.jpos[link]))           # rotation about the axis through jpos
        c = p + m3.quat_rotate(r, e(self.com)) if at_com else p
        w = sum(a * qdi for a, qdi in zip(axes, rates))
        v = sum(torch.cross(a * qdi, c - o, dim=-1) for a, o, qdi in zip(axes, origins, rates))
        return p, r, v, w
