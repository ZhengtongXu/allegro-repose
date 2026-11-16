"""Pure-torch definition of the Allegro cube-repose MDP signals (Isaac Lab recipe).

The mjlab environment (``allegro_repose.task.env``) and the evaluation call these functions on a
batched physical state, so every MDP signal has a single definition that can be unit-tested
without the simulator (``tests/test_core.py``).

Physical state layout (``STATE_DIM = 45``), env frame:

    q    [0:16]   hand joint positions, MJCF order (index, middle, ring, thumb; joints 0-3 each)
    qd   [16:32]  hand joint velocities
    p    [32:35]  cube position
    quat [35:39]  cube orientation (wxyz)
    v    [39:42]  cube linear velocity (world)
    w    [42:45]  cube angular velocity (world)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from allegro_repose import math3d as m3

NJ = 16
NTIPS = 4
STATE_DIM = 45
SL_Q = slice(0, 16)
SL_QD = slice(16, 32)
SL_P = slice(32, 35)
SL_QUAT = slice(35, 39)
SL_V = slice(39, 42)
SL_W = slice(42, 45)

@dataclass
class TaskSpec:
    """Task constants on the training device, plus the fingertip kinematics."""

    joint_lo: torch.Tensor
    joint_hi: torch.Tensor
    in_hand_pos: torch.Tensor
    vel_obs_scale: float
    success_tol: float
    dist_reward_scale: float
    rot_reward_scale: float
    rot_eps: float
    action_penalty_scale: float
    reach_goal_bonus: float
    fall_penalty: float
    fall_dist: float
    kinematics: object  # HandKinematics
    orient_band: float = 0.0          # orientation reward is flat within this angle of the goal (0: Isaac Lab)
    action_rate_scale: float = 0.0    # weight (<= 0) of sum(relu(|a - a_prev| - band)^2)
    action_rate_band: float = 0.05
    obs_layout: str = "isaaclab"      # "isaaclab" (124-D, symmetric) or "rot6d" (asymmetric actor / critic)

    @staticmethod
    def from_cfg(task_cfg: dict, joint_lo, joint_hi, kinematics, device) -> TaskSpec:
        r, t = task_cfg["reward"], lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
        init = task_cfg["cube"]["init_pos"]
        in_hand = [init[0], init[1], init[2] + task_cfg["termination"]["in_hand_offset_z"]]
        return TaskSpec(
            joint_lo=t(joint_lo), joint_hi=t(joint_hi), in_hand_pos=t(in_hand),
            vel_obs_scale=float(task_cfg["observation"]["vel_obs_scale"]),
            success_tol=float(task_cfg["goal"]["success_tolerance"]),
            dist_reward_scale=float(r["dist_reward_scale"]), rot_reward_scale=float(r["rot_reward_scale"]),
            rot_eps=float(r["rot_eps"]), action_penalty_scale=float(r["action_penalty_scale"]),
            reach_goal_bonus=float(r["reach_goal_bonus"]), fall_penalty=float(r["fall_penalty"]),
            fall_dist=float(task_cfg["termination"]["fall_dist"]), kinematics=kinematics,
            orient_band=float(r.get("orientation_band", 0.0)),
            action_rate_scale=float(r.get("action_rate_scale", 0.0)),
            action_rate_band=float(r.get("action_rate_band", 0.05)),
            obs_layout=str(task_cfg["observation"].get("layout", "isaaclab")),
        )


def unscale(q: torch.Tensor, spec: TaskSpec) -> torch.Tensor:
    return (2.0 * q - spec.joint_hi - spec.joint_lo) / (spec.joint_hi - spec.joint_lo)


def fingertips(state: torch.Tensor, spec: TaskSpec, at_com: bool = True):
    return spec.kinematics(state[..., SL_Q], state[..., SL_QD], at_com=at_com)


def observation(state: torch.Tensor, goal_quat: torch.Tensor, last_action: torch.Tensor,
                spec: TaskSpec) -> torch.Tensor:
    """Isaac Lab ``compute_full_observations`` (124-D); also the critic's input (symmetric)."""
    quat = state[..., SL_QUAT]
    tip_p, tip_q, tip_v, tip_w = fingertips(state, spec)
    lead = state.shape[:-1]
    s = spec.vel_obs_scale
    return torch.cat((
        unscale(state[..., SL_Q], spec),
        s * state[..., SL_QD],
        state[..., SL_P],
        quat,
        state[..., SL_V],
        s * state[..., SL_W],
        spec.in_hand_pos.expand(*lead, 3),
        goal_quat,
        m3.quat_mul(quat, m3.quat_conj(goal_quat)),
        tip_p.reshape(*lead, NTIPS * 3),
        tip_q.reshape(*lead, NTIPS * 4),
        torch.cat((tip_v, tip_w), -1).reshape(*lead, NTIPS * 6),
        last_action,
    ), dim=-1)


def actor_obs_rot6d(state: torch.Tensor, goal_quat: torch.Tensor, last_action: torch.Tensor,
                    spec: TaskSpec) -> torch.Tensor:
    """Asymmetric layout, actor part (89-D): only what a real hand with a cube tracker measures.
    Orientations are rotation-6D (first two rotation-matrix columns): continuous, unlike
    quaternions with their q / -q double cover. Joint positions scaled to [-1, 1]; fingertip
    positions and orientations; cube orientation; goal orientation; cube-goal difference;
    in-hand target position; last action. No velocities."""
    quat = state[..., SL_QUAT]
    tip_p, tip_q, _, _ = fingertips(state, spec)
    lead = state.shape[:-1]
    return torch.cat((
        unscale(state[..., SL_Q], spec),
        tip_p.reshape(*lead, NTIPS * 3),
        m3.rot6d_from_quat(tip_q).reshape(*lead, NTIPS * 6),
        m3.rot6d_from_quat(quat),
        m3.rot6d_from_quat(goal_quat),
        m3.rot6d_from_quat(m3.quat_mul(quat, m3.quat_conj(goal_quat))),
        spec.in_hand_pos.expand(*lead, 3),
        last_action,
    ), dim=-1)


def critic_privileged_rot6d(state: torch.Tensor, spec: TaskSpec) -> torch.Tensor:
    """Asymmetric layout: what the critic (138-D) sees in addition to the actor's 89-D observation:
    0.2 x joint velocities, fingertip linear (frame origin) and angular velocities, cube
    position, linear velocity and 0.2 x angular velocity."""
    _, _, tip_v, tip_w = fingertips(state, spec, at_com=False)
    lead = state.shape[:-1]
    return torch.cat((
        spec.vel_obs_scale * state[..., SL_QD],
        torch.cat((tip_v, tip_w), -1).reshape(*lead, NTIPS * 6),
        state[..., SL_P],
        state[..., SL_V],
        spec.vel_obs_scale * state[..., SL_W],
    ), dim=-1)


def rotation_distance(quat: torch.Tensor, goal_quat: torch.Tensor) -> torch.Tensor:
    """Isaac Lab: 2 asin(min(|vec(q g^-1)|, 1))."""
    diff = m3.quat_mul(quat, m3.quat_conj(goal_quat))
    # (max 1 - 1e-6 instead of Isaac Lab's 1: differs only within 1e-6 of 180 deg)
    return 2.0 * torch.asin(torch.clamp(diff[..., 1:4].norm(dim=-1), max=1.0 - 1e-6))


def goal_dist(state: torch.Tensor, spec: TaskSpec) -> torch.Tensor:
    return (state[..., SL_P] - spec.in_hand_pos).norm(dim=-1)


def success_mask(state: torch.Tensor, goal_quat: torch.Tensor, spec: TaskSpec) -> torch.Tensor:
    return rotation_distance(state[..., SL_QUAT], goal_quat) <= spec.success_tol


def reward_terms(state: torch.Tensor, goal_quat: torch.Tensor, action: torch.Tensor, spec: TaskSpec,
                 prev_action: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Isaac Lab ``compute_rewards`` on the state after the step, scored against goal g.

    Hold design (``reward.orientation_band``, ``reward.action_rate_scale`` and ``goal.dwell_steps`` in
    the env): the bonus is paid on every step the cube is within tolerance, so with a dwell it pays
    for holding the goal; the orientation term can be flat near the goal; the action-rate term
    (needs ``prev_action``) penalizes per-joint target changes beyond a small free band.
    """
    rot = rotation_distance(state[..., SL_QUAT], goal_quat)
    dist = goal_dist(state, spec)
    hit = (rot <= spec.success_tol).float()
    fell = (dist >= spec.fall_dist).float()
    if prev_action is not None and spec.action_rate_scale != 0.0:
        excess = ((action - prev_action).abs() - spec.action_rate_band).clamp_min(0.0)
        action_rate = spec.action_rate_scale * (excess * excess).sum(-1)
    else:
        action_rate = torch.zeros_like(dist)
    return {
        "dist": spec.dist_reward_scale * dist,
        # orient_band > 0: flat inside the band, so there is no incentive to keep adjusting near the goal
        "rot": spec.rot_reward_scale / ((rot.abs() - spec.orient_band).clamp_min(0.0) + spec.rot_eps),
        "action_rate": action_rate,
        "action_penalty": spec.action_penalty_scale * (action * action).sum(-1),
        "reach_goal": spec.reach_goal_bonus * hit,
        "fall": spec.fall_penalty * fell,
    }


def termination(state: torch.Tensor, spec: TaskSpec) -> torch.Tensor:
    return goal_dist(state, spec) >= spec.fall_dist


def random_rotation(n: int, device) -> torch.Tensor:
    """Isaac Lab ``randomize_rotation``: Rx(pi u0) * Ry(pi u1), u ~ U(-1, 1) (goals and cube resets)."""
    u = torch.rand(n, 2, device=device) * 2 - 1
    x = torch.zeros(n, 3, device=device)
    y = torch.zeros(n, 3, device=device)
    x[:, 0] = math.pi * u[:, 0]
    y[:, 1] = math.pi * u[:, 1]
    return m3.quat_mul(m3.quat_from_axis_angle(x), m3.quat_from_axis_angle(y))
