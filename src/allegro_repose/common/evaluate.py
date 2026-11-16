"""Evaluation protocol for trained policies.

Every policy is scored the same way: deterministic actions, ``num_envs`` independent episodes of
full length (10 s), fixed evaluation seed. An episode ends early only if the cube falls; goals
reached before the fall still count. ``successes_per_episode`` is the number of goals completed in an
episode (Isaac Lab's ``consecutive_successes`` without its moving average); with ``goal.dwell_steps``
> 1 a goal is completed only after the cube was held within tolerance for that many steps.
``on_goal_frac`` is the fraction of steps with the cube within tolerance of its goal.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from allegro_repose.task import core


@torch.no_grad()
def evaluate_policy(env, policy: Callable[[torch.Tensor], torch.Tensor], seed: int = 12345) -> dict:
    """``policy`` maps actor observations [N, D] to actions [N, 16] (clipped to [-1, 1] here).
    The global RNG state is saved and restored."""
    from allegro_repose.common.run import rng_state, set_rng_state

    saved = rng_state()
    try:
        return _evaluate(env, policy, seed)
    finally:
        set_rng_state(saved)


def _evaluate(env, policy, seed):
    env.seed(seed)
    obs, _ = env.reset()
    n, dev = env.num_envs, env.device
    active = torch.ones(n, dtype=torch.bool, device=dev)
    successes = torch.zeros(n, device=dev)
    ret = torch.zeros(n, device=dev)
    err_sum = torch.zeros(n, device=dev)
    steps = torch.zeros(n, device=dev)
    fell = torch.zeros(n, dtype=torch.bool, device=dev)
    on_goal = torch.zeros(n, device=dev)
    for _ in range(env.max_episode_length):
        a = policy(obs["actor"]).clamp(-1.0, 1.0)
        goal_before = env.goal.clone()
        obs, r, term, trunc, _ = env.step(a)
        done = term | trunc
        live = active & ~done  # the state after the step is valid (no auto-reset happened)
        state = env.state()
        successes += ((env.goal != goal_before).any(-1) & live).float()  # completed -> replaced
        on_goal += (core.success_mask(state, goal_before, env.spec) & live).float()
        err_sum += core.rotation_distance(state[:, core.SL_QUAT], goal_before) * live
        steps += live.float()
        ret += r * active
        fell |= term & active
        active &= ~done
        if not active.any():
            break
    return {
        "eval/successes_per_episode": float(successes.mean()),
        "eval/successes_median": float(successes.median()),
        "eval/return": float(ret.mean()),
        "eval/rot_dist": float((err_sum / steps.clamp_min(1)).mean()),
        "eval/on_goal_frac": float((on_goal / steps.clamp_min(1)).mean()),
        "eval/fall_rate": float(fell.float().mean()),
        "eval/episode_length_s": float(steps.mean() * env.step_dt),
    }
