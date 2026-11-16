# Tutorial: train, resume, evaluate and render an Allegro repose policy

This walks through the whole workflow, from a fresh checkout to a rendered video of a trained
policy, on a workstation and on a Slurm cluster. Every command is run from the repository root.

1. [Setup](#1-setup)
2. [Train and resume on one GPU](#2-train-and-resume-on-one-gpu)
3. [Evaluate and render](#3-evaluate-and-render)
4. [Train on Slurm (priority and standby)](#4-train-on-slurm)
5. [How auto-resume works](#5-how-auto-resume-works)
6. [Monitor runs](#6-monitor-runs)
7. [Recipes](#7-recipes)

## 1. Setup

Requirements: Linux, an NVIDIA GPU with a CUDA 12.8 driver, [uv](https://docs.astral.sh/uv/).
A full-size run (8192 envs) uses about 12 GB of GPU memory; any 16 GB card or larger works.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv is missing (installs to ~/.local/bin)
git clone <this repository> allegro-repose && cd allegro-repose
uv sync --extra dev        # Python 3.12, torch 2.10 (cu128), mjlab 1.5.3, rsl-rl-lib 5.4.0 into .venv/
uv run pytest -q           # asset checksums, hand setup, fingertip kinematics vs MuJoCo, task math
uv run ruff check src scripts tests
uv run wandb login         # once; the key is stored in ~/.netrc
```

On a cluster, run these on a login node inside the checkout (the tests need no GPU); the jobs use
the checkout's `.venv/` directly. Keep the checkout on a scratch file system: runs and checkpoints
are written to `runs/` inside it.

Without W&B, set `WANDB_MODE=disabled` (or `offline`) in the environment of every command below.
Rendering uses MuJoCo's EGL backend: `export MUJOCO_GL=egl` on headless machines.

## 2. Train and resume on one GPU

Two task configs exist (`configs/task/`):

| config | what counts as a goal | use it for |
|---|---|---|
| `allegro_repose` | the cube touches the goal (Isaac Lab's task, unchanged) | reproducing Isaac Lab |
| `allegro_repose_hold` | the cube stays on the goal for `goal.dwell_steps` (30 = 1 s) | policies that hold a pose |

```bash
# Isaac Lab recipe (about 3.5 h on an A100 for 10 000 iterations = 1.31 B env steps)
uv run python scripts/train.py --run-name ppo_s0 --seed 0

# Recommended: hold design with a 3 s dwell (about 4.5 h on an A100;
# add algo.max_iterations=20000 for the best policy, about 9 h)
uv run python scripts/train.py --run-name hold_dw90 --seed 0 \
    --task allegro_repose_hold task.goal.dwell_steps=90
```

Arguments after the options are dotted overrides of the composed config (`task.*` from the task
file, `algo.*` from `configs/algo/ppo.yaml`), e.g. `algo.max_iterations=20000` or
`task.reward.fall_penalty=-100.0`. `--num-envs` and `--eval-envs` (default 1024) set the batch sizes.

**Where things go.** A run lives in `runs/<run-name>/`:

```
runs/<run-name>/
  config.yaml              resolved config, written once at the start
  checkpoints/ckpt_<it>.pt the newest algo.keep_last (3), saved every algo.save_interval (250) iterations
  wandb_id.txt             W&B run id, reused by every resumed chunk (one continuous W&B run)
  DONE                     written when training reaches algo.max_iterations
  history.log, MODE        Slurm chain bookkeeping (section 4)
```

Set `AREPOSE_RUNS=/some/dir` to put runs somewhere else. `algo.keep_every=1000` additionally keeps
every 1000th checkpoint.

**Resume.** Run the same command again. `train.py` finds `runs/<run-name>/config.yaml`, ignores the
overrides on the command line (the stored config wins), loads the newest checkpoint that loads
(a truncated one is renamed `*.bad` and the next one is tried), and continues the same W&B run.
Iteration counter, env-step counter, learning rate, RNG states and the training clock all continue.

**Stopping early.** `--max-chunk-min 225` makes `train.py` checkpoint and exit after 225 minutes,
and SIGUSR1 or SIGTERM (`kill -USR1 <pid>`) does the same at once. Exit codes: `0` finished
(`DONE` written), `3` stopped with a checkpoint (run it again to continue), anything else crashed.

## 3. Evaluate and render

```bash
# 4096 deterministic 10 s episodes, newest checkpoint; prints and saves the metrics
uv run python scripts/evaluate.py runs/hold_dw90 --num-envs 4096 --json hold_dw90.json

# a specific checkpoint
uv run python scripts/evaluate.py runs/hold_dw90 --ckpt runs/hold_dw90/checkpoints/ckpt_9999.pt

# score under another rule, e.g. a 1 s hold for a policy trained with 3 s
uv run python scripts/evaluate.py runs/hold_dw90 --set task.goal.dwell_steps=30

# 30 s video at half speed (60 s of video), no evaluation
MUJOCO_GL=egl uv run python scripts/evaluate.py runs/hold_dw90 --num-envs 0 \
    --video hold_dw90.mp4 --seconds 30 --speed 0.5
```

Metrics (`src/allegro_repose/common/evaluate.py`): `successes_per_episode` (goals completed; with a
dwell, goals held for the dwell), `on_goal_frac` (fraction of steps within 0.2 rad of the goal),
`fall_rate`, `rot_dist` (mean rotation error), `return`.

In the video the translucent cube to the right of the hand shows the current goal orientation and
the overlay counts goals and drops. For a policy trained on the Isaac Lab task, which draws a new
goal the moment the cube touches one, `--hold 2` keeps every reached goal on screen for 2 s (the
policy was not trained to hold, so it may keep turning the cube).

## 4. Train on Slurm

`cluster/` runs each training run as a chain of Slurm jobs ("chunks") that resume one another, so
a run can be longer than the wall-clock limit of any single job.

**One-time setup.**

```bash
cp cluster/site.env.example cluster/site.env   # git-ignored; fill in account, partitions, QOS
```

`site.env` describes two submission modes. The example file is filled in for Purdue Gilbreth with
account `shey`:

| mode | Gilbreth | partitions | wall limit | each chunk ends |
|---|---|---|---|---|
| `priority` | the group quota (4 A100-40GB, 8 A30), `-q normal` | one: `a100-40gb` (or `a30`) | 3 days per chunk | 5 min before the limit (SIGUSR1), or when the run is done |
| `standby` | idle GPUs anywhere, `-q standby` | `a30,a100-40gb,a100-80gb,h100` | 4 h | after `STANDBY_CHUNK_MIN` (225) minutes, or SIGUSR1 |

Use `priority` while the group quota has a free GPU: the job starts at once and a 10 000-iteration
run finishes in one chunk (about 4.5 h). Use `standby` when the quota is full; the run then takes a
few 4 h chunks, each resuming the last. To see the quota:
`scontrol show assoc_mgr accounts=shey flags=assoc | grep -o "gres/hp_a[0-9a-z-]*=[^,]*" | sort -u`
(`4(2)` = 2 of 4 in use). Each chunk uses 8 CPU cores in priority mode and 6 in standby (fits busy nodes).

Every `site.env` setting can be overridden from the environment of `submit.sh` (for example
`PRIORITY_PARTITION=a30`); the override stays in effect for the later chunks of that chain, since
Slurm passes the submitting environment on to the job.

**Declare the run** in `cluster/jobs.txt`: name, seed, then `train.py` arguments and overrides.
Every run in `docs/RESULTS.md` is already declared there.

```
# run_name   seed  [train.py arguments and dotted overrides...]
hold_dw90    0     --task allegro_repose_hold task.goal.dwell_steps=90
```

**Launch.**

```bash
bash cluster/submit.sh hold_dw90 --mode priority     # or --mode standby
```

That is all: the job writes `cluster/logs/hold_dw90_<jobid>.out` and continues by itself until
`runs/hold_dw90/DONE` exists. The mode is stored in `runs/hold_dw90/MODE` and reused by every later
chunk. Submitting a run that is already queued or running does nothing, so `submit.sh` is safe to
call repeatedly.

**Switch a running chain to the other mode** (e.g. from a busy priority queue to standby): cancel
the pending successor and submit with the new mode after the running chunk.

```bash
squeue -u $USER -n arp_hold_dw90                       # running chunk R, pending successor P
scancel <P>
bash cluster/submit.sh hold_dw90 --mode standby --after <R>
```

## 5. How auto-resume works

```
submit.sh ──► chunk 1 ──queues──► chunk 2 (afterany:1) ──queues──► chunk 3 (afterany:2) ...
              trains, checkpoints,  waits for chunk 1;
              exits 3               resumes from the newest checkpoint
```

Each chunk (`cluster/job.sbatch`):

1. **queues its successor first** (`--dependency=afterany:<own id>`), so even a chunk that is killed
   without warning (node failure, preemption) leaves a successor behind;
2. **trains** with `scripts/train.py`, which resumes from the newest loadable checkpoint in
   `runs/<run>/checkpoints/`. Slurm sends SIGUSR1 5 minutes before the wall limit
   (`--signal=B:USR1@300`); if `<MODE>_CHUNK_MIN` is set (`STANDBY_CHUNK_MIN=225` by default),
   `train.py` also stops on its own after that many minutes, well inside the limit. Either way it
   writes a checkpoint and exits with code 3;
3. **reads the exit code**: `0` (finished) cancels the successor; `3` lets it run; anything else is
   a crash, and the successor retries. Three crashes in a row, each within 10 minutes of the chunk's
   start, write `runs/<run>/FAILED` and stop the chain (a deterministic bug should not burn the
   queue). After fixing the cause: `rm runs/<run>/FAILED` and submit again.

Because only the newest 3 checkpoints are kept and each is written atomically, a chunk killed
during a save loses at most `save_interval` iterations. The W&B run id is stored in the run
directory, so all chunks log to one W&B run with continuous step counters.

Applying a code fix to a running chunk without giving up its GPU: `kill -USR2 <train.py pid>` saves
a checkpoint and restarts `train.py` in place on the code now on disk.

## 6. Monitor runs

```bash
bash cluster/status.sh            # one line per run: mode, queue state, last iteration, last eval
squeue -u $USER -o "%.10i %.9P %.28j %.2t %.10M %R"
tail -f cluster/logs/hold_dw90_<jobid>.out
cat runs/hold_dw90/history.log    # submitted / start / end (exit code) of every chunk
```

A training line reads `[ppo] it 8890 steps 1.165e+09 ret 41264.5 succ 0.81 rot 0.608 (77855 sps)`:
iteration, env steps, mean episode return and goals per episode of the training batch, mean rotation
error, simulation throughput. Every `algo.eval_interval` iterations an evaluation prints the
`eval/*` metrics of section 3 on `--eval-envs` episodes; W&B shows the same curves (project
`WANDB_PROJECT`, group `ppo`).

What to look for:

* `history.log` shows `exit=3` at every chunk boundary and `exit=0` at the end. Other exit codes
  are crashes; the chunk's `.out` file has the traceback.
* `FAILED` in `status.sh`: three quick crashes; see above.
* A run that stays pending: `squeue` shows the reason (`Priority`, `Resources`: the queue is busy;
  `Dependency`: waiting for the previous chunk). Switching mode (section 4) can help.

## 7. Recipes

```bash
# Reproduce the best policy (declared in jobs.txt), then evaluate and render it on a GPU node
bash cluster/submit.sh ppo_hold_dw90_long --mode priority  # or --mode standby (about 3 chunks)
bash cluster/status.sh                                   # until DONE (about 9 h on an A100)
uv run python scripts/evaluate.py runs/ppo_hold_dw90_long --num-envs 4096 --json dw90_long.json
MUJOCO_GL=egl uv run python scripts/evaluate.py runs/ppo_hold_dw90_long --num-envs 0 \
    --video dw90_long.mp4 --seconds 30 --speed 0.5

# Train longer than planned: raise max_iterations in the stored config, remove DONE, resubmit
sed -i 's/max_iterations: 10000/max_iterations: 20000/' runs/hold_dw90/config.yaml
rm runs/hold_dw90/DONE && bash cluster/submit.sh hold_dw90

# Check the chunking itself in a few minutes: a small run that spans several 4-minute chunks
STANDBY_CHUNK_MIN=4 bash cluster/submit.sh smoke --mode standby
cat runs/smoke/history.log                               # exit=3 ... exit=0
rm -r runs/smoke                                         # afterwards
```

`smoke` is defined in `cluster/jobs.txt` (1024 envs, 600 iterations, a checkpoint every 50). When
the standby queue is busy, `PRIORITY_CHUNK_MIN=4 bash cluster/submit.sh smoke --mode priority` runs
the same short-chunk chain on the group quota. On Gilbreth, such a check gave: chunk 1 stopped at its
4-minute budget (iteration 334, exit 3), chunk 2 started a minute later, resumed from
`ckpt_334.pt` at iteration 335 and finished the run (exit 0), cancelling its own successor.
