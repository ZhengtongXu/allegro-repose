"""Train PPO on fixed-wrist Allegro cube repose (resumable; one call = one Slurm chunk).

    python scripts/train.py --run-name ppo_s0 --seed 0
    python scripts/train.py --run-name hold_dw90 --task allegro_repose_hold task.goal.dwell_steps=90
    python scripts/train.py --run-name hold_dw90 --max-chunk-min 225     # stop after 225 min (exit 3)

Re-running the same command resumes from the newest loadable checkpoint in runs/<run-name>/ with
the config stored there (CLI overrides are ignored on resume). Exit codes: 0 = finished
(runs/<run-name>/DONE written), 3 = stopped early with a checkpoint (time budget or Slurm signal),
anything else = crash.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

from allegro_repose.common.run import RunDir, WandbLogger, runs_root, seed_everything
from allegro_repose.config import load_experiment


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", default="allegro_repose")
    ap.add_argument("--num-envs", type=int, default=None)
    ap.add_argument("--eval-envs", type=int, default=1024)
    ap.add_argument("--max-chunk-min", type=float, default=None, help="checkpoint and exit (code 3) after this many minutes")
    ap.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "allegro-repose"))
    ap.add_argument("--wandb-group", default=None)
    ap.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "online"))
    ap.add_argument("overrides", nargs="*", help="dotted overrides, e.g. algo.max_iterations=20000")
    args = ap.parse_args()

    run = RunDir(runs_root(), args.run_name)
    if run.done:
        print(f"[train] {run.path} is DONE; nothing to do")
        return 0
    cfg = run.read_config()
    if cfg is None:
        cfg = load_experiment(args.task, "ppo", args.overrides)
        if args.num_envs:
            cfg["algo"]["num_envs"] = args.num_envs
        cfg["run"] = {"name": args.run_name, "seed": args.seed, "eval_envs": args.eval_envs}
        run.write_config(cfg)
    else:
        print(f"[train] resuming {run.path} with its stored config")
    run.keep_last, run.keep_every = cfg["algo"].get("keep_last", 3), cfg["algo"].get("keep_every", 0)

    seed_everything(cfg["run"]["seed"])
    device = "cuda:0"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from allegro_repose.task.env import ReposeEnv

    env = ReposeEnv(cfg, num_envs=cfg["algo"]["num_envs"], device=device, seed=cfg["run"]["seed"])
    eval_n = cfg["run"]["eval_envs"]
    eval_env = ReposeEnv(cfg, num_envs=eval_n, device=device, seed=10_000) if eval_n else None
    logger = WandbLogger(run, cfg, project=args.wandb_project, group=args.wandb_group or "ppo",
                         tags=["ppo"], mode=args.wandb_mode)
    max_wall_s = args.max_chunk_min * 60 if args.max_chunk_min else None
    from allegro_repose.ppo.train import train_ppo

    try:
        code = train_ppo(cfg, env, eval_env, run, logger, device, max_wall_s)
    finally:
        logger.finish()
    return code


if __name__ == "__main__":
    code = main()
    from allegro_repose.common import run as run_mod

    if run_mod.RESTART_IN_PLACE and code == run_mod.EXIT_PREEMPTED:
        print("[train] SIGUSR2: restarting in place on the current code", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os.execv(sys.executable, [sys.executable, *sys.argv])
    sys.exit(code)
