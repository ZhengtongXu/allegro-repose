"""Task math: Isaac Lab reward formula, rotation distance, hold-design terms, config composition."""

import torch

from allegro_repose import math3d as m3
from allegro_repose.task import core


def _spec():
    return core.TaskSpec(joint_lo=-torch.ones(16), joint_hi=torch.ones(16), in_hand_pos=torch.tensor([0.0, -0.17, 0.52]),
                         vel_obs_scale=0.2, success_tol=0.2, dist_reward_scale=-10.0, rot_reward_scale=1.0, rot_eps=0.1,
                         action_penalty_scale=-0.0002, reach_goal_bonus=250.0, fall_penalty=0.0, fall_dist=0.24,
                         kinematics=None)


def _state(p, quat):
    s = torch.zeros(1, core.STATE_DIM)
    s[0, core.SL_P] = torch.tensor(p)
    s[0, core.SL_QUAT] = torch.tensor(quat)
    return s


def test_rotation_distance_matches_angle():
    for ang in (0.0, 0.1, 1.0, 3.0):
        q = m3.quat_from_axis_angle(torch.tensor([[0.0, ang, 0.0]]))
        d = core.rotation_distance(q, torch.tensor([[1.0, 0, 0, 0]]))
        assert abs(d.item() - ang) < 2e-3


def test_reward_terms_follow_isaaclab():
    spec = _spec()
    goal = torch.tensor([[1.0, 0, 0, 0]])
    rot = 0.15
    quat = m3.quat_from_axis_angle(torch.tensor([[rot, 0.0, 0.0]]))[0].tolist()
    s = _state([0.0, -0.17, 0.55], quat)  # 3 cm from in_hand_pos, inside the success tolerance
    a = torch.full((1, 16), 0.5)
    t = core.reward_terms(s, goal, a, spec)
    assert abs(t["dist"].item() - (-10 * 0.03)) < 1e-5
    assert abs(t["rot"].item() - 1 / (rot + 0.1)) < 1e-3
    assert abs(t["action_penalty"].item() - (-0.0002 * 16 * 0.25)) < 1e-7
    assert t["reach_goal"].item() == 250.0 and t["fall"].item() == 0.0
    assert not core.termination(s, spec).item()
    assert core.termination(_state([0.0, -0.17, 0.25], quat), spec).item()


def test_hold_reward_terms():
    import dataclasses

    spec = dataclasses.replace(_spec(), orient_band=0.31, action_rate_scale=-0.05, action_rate_band=0.05)
    goal = torch.tensor([[1.0, 0, 0, 0]])
    quat = m3.quat_from_axis_angle(torch.tensor([[0.25, 0.0, 0.0]]))[0].tolist()
    s = _state([0.0, -0.17, 0.52], quat)
    a, a_prev = torch.zeros(1, 16), torch.zeros(1, 16)
    a[0, 0] = 0.15  # 0.1 beyond the free band
    t = core.reward_terms(s, goal, a, spec, prev_action=a_prev)
    assert abs(t["rot"].item() - 1 / 0.1) < 1e-4            # inside the band: flat at the peak
    assert abs(t["action_rate"].item() - (-0.05 * 0.01)) < 1e-6


def test_config_inheritance():
    from allegro_repose.config import load_experiment

    base = load_experiment("allegro_repose", "ppo")["task"]
    hold = load_experiment("allegro_repose_hold", "ppo")["task"]
    assert hold["goal"]["dwell_steps"] == 30 and base["goal"]["dwell_steps"] == 1
    assert hold["observation"]["layout"] == "rot6d" and hold["observation"]["vel_obs_scale"] == 0.2
    assert hold["cube"] == base["cube"] and hold["hand"] == base["hand"]
