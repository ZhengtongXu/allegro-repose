"""Evaluate a trained policy with the shared protocol, and optionally render a video.

    python scripts/evaluate.py runs/ppo_s0 --num-envs 4096 [--ckpt runs/ppo_s0/checkpoints/ckpt_9999.pt]
    python scripts/evaluate.py runs/hold_dw90 --num-envs 0 --video hold_dw90.mp4 --seconds 30 --speed 0.5

The video shows one env; the translucent cube to the side is the goal orientation (Isaac Lab's
goal marker), and the overlay counts the goals reached.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from allegro_repose.common.evaluate import evaluate_policy
from allegro_repose.common.run import RunDir
from allegro_repose.config import apply_override
from allegro_repose.task.env import ReposeEnv


def load_policy(state: dict, env, device):
    """Deterministic policy obs -> absolute joint target in [-1, 1]."""
    from allegro_repose.ppo.train import build_ppo, deterministic_policy

    _, alg = build_ppo(state["config"], env, device)
    alg.load(state["ppo"], None, strict=True)
    alg.eval_mode()
    return deterministic_policy(alg)


def render(cfg, state, out: str, seconds: float, seed: int, device: str, speed: float = 1.0, hold_s: float = 0.0) -> dict:
    """``speed`` < 1 plays the video slower than real time; ``hold_s`` > 0 keeps each reached goal for that long
    (the policy holds the cube there) before the next goal is drawn. Training draws the next goal at once."""
    import mediapy
    import numpy as np
    from PIL import Image, ImageDraw

    from allegro_repose.assets import GOAL_MARKER_BODY

    env = ReposeEnv(cfg, num_envs=1, device=device, seed=seed, render_mode="rgb_array")
    policy = load_policy(state, env, device)
    mocap = env.sim.mj_model.body(f"cube/{GOAL_MARKER_BODY}").mocapid[0]
    obs, _ = env.reset()
    cmd = env.command_manager.get_term("goal")
    hold_steps, held, hold_left = round(hold_s / env.step_dt), None, 0
    frames, goals, falls = [], 0, 0
    with torch.no_grad():
        for _ in range(round(seconds / env.step_dt)):
            goal = env.goal.clone()
            obs, _, term, trunc, _ = env.step(policy(obs["actor"]).clamp(-1, 1))
            done = bool((term | trunc).any())
            if done:
                held, hold_left = None, 0
            elif held is not None:
                # holding: undo the env's resampling until the hold time is over
                hold_left -= 1
                if hold_left > 0:
                    cmd.goal[0] = held
                else:
                    held = None
            elif (env.goal[0] != goal[0]).any():  # the goal was completed and replaced
                goals += 1
                if hold_steps > 0:
                    held, hold_left = goal[0].clone(), hold_steps
                    cmd.goal[0] = held
            falls += int(term.sum())
            env.sim.data.mocap_quat[0, mocap] = env.goal[0]
            env.sim.forward()
            img = Image.fromarray(np.asarray(env.render()))
            label = f"t {len(frames) * env.step_dt:5.1f} s   goals {goals}   drops {falls}"
            if held is not None:
                label += "   goal reached: holding"
            if speed != 1.0:
                label += f"   ({speed:g}x speed)"
            ImageDraw.Draw(img).text((12, 10), label, fill=(255, 255, 255))
            frames.append(np.asarray(img))
    mediapy.write_video(out, frames, fps=max(1, round(speed / env.step_dt)))
    return {"video": out, "goals": goals, "drops": falls}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--ckpt", type=Path, default=None, help="default: newest checkpoint of the run")
    ap.add_argument("--num-envs", type=int, default=4096, help="0: skip the evaluation")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--video", default=None)
    ap.add_argument("--seconds", type=float, default=30.0, help="simulated seconds")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed of the video (0.5: half speed)")
    ap.add_argument("--hold", type=float, default=0.0, help="seconds to keep each reached goal before the next")
    ap.add_argument("--json", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="evaluate under a changed environment, e.g. task.goal.dwell_steps=30")
    args = ap.parse_args()
    device = "cuda:0"
    if args.ckpt is None:
        state, path = RunDir(args.run_dir.parent, args.run_dir.name, create=False).load_latest(map_location=device)
        if state is None:
            raise SystemExit(f"no checkpoint in {args.run_dir}/checkpoints")
    else:
        path, state = args.ckpt, torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = copy.deepcopy(state["config"])
    for kv in args.set:
        apply_override(cfg, kv)
    state["config"] = cfg
    out = {"checkpoint": str(path), "env_steps": state["env_steps"], "wall_time_h": state["wall_time"] / 3600,
           **({"overrides": args.set} if args.set else {})}
    if args.num_envs:
        env = ReposeEnv(cfg, num_envs=args.num_envs, device=device, seed=args.seed)
        out.update(evaluate_policy(env, load_policy(state, env, device), seed=args.seed))
    if args.video:
        out.update(render(cfg, state, args.video, args.seconds, args.seed, device, args.speed, args.hold))
    print(json.dumps(out, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
