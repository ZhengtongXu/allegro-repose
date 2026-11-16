# allegro-repose

Fixed-wrist in-hand cube reorientation ("repose") with the **Allegro hand** in
[mjlab](https://github.com/mujocolab/mjlab) (MuJoCo Warp), trained with **PPO** (rsl_rl).

The environment is Isaac Lab's `Isaac-Repose-Cube-Allegro-Direct-v0` (Isaac Lab v2.3.2) adapted to
mjlab: the same hand setup, cube, observation, reward, resets and terminations, on the MuJoCo
Menagerie Allegro model, with Isaac Lab's PPO recipe. A second task config, the **hold design**,
asks the policy to stay on each goal orientation for a while instead of only touching it.

| policy (4096 episodes of 10 s) | goals held 1 s / episode | time on goal | drops |
|---|---|---|---|
| Isaac Lab task and reward | 0.12 (11.3 touched) | 31 % | 9 % |
| hold design, 3 s dwell | 2.61 | 60 % (own 3 s rule) | 16 % (own rule) |
| hold design, 3 s dwell, 20 000 iterations (best) | 2.83 | 62 % (own 3 s rule) | 15 % (own rule) |

Videos and all runs: `docs/RESULTS.md`, `docs/videos/`.

## Quick start

```bash
uv sync --extra dev && uv run pytest -q && uv run wandb login

# train (about 4.5 h on one A100); re-running the same command resumes
uv run python scripts/train.py --run-name hold_dw90 --task allegro_repose_hold task.goal.dwell_steps=90

# evaluate and render
uv run python scripts/evaluate.py runs/hold_dw90 --num-envs 4096
MUJOCO_GL=egl uv run python scripts/evaluate.py runs/hold_dw90 --num-envs 0 --video hold_dw90.mp4 --speed 0.5

# or on Slurm, as a self-resuming chain of jobs (long-wall priority QOS or 4 h standby)
cp cluster/site.env.example cluster/site.env      # Gilbreth/shey values; edit for another cluster
bash cluster/submit.sh ppo_hold_dw90 --mode standby
bash cluster/status.sh
```

**[docs/TUTORIAL.md](docs/TUTORIAL.md)** is the full walkthrough: setup, training and resuming,
evaluation and rendering, Slurm launch in priority and standby mode, how the auto-resume chain
works, and monitoring.

## Documentation

| file | contents |
|---|---|
| `docs/TUTORIAL.md` | step-by-step usage, cluster workflow |
| `docs/ENVIRONMENT.md` | every environment value and every difference from Isaac Lab; the hold design |
| `docs/METHOD.md` | PPO configuration |
| `docs/RESULTS.md` | results, evaluations (`docs/evals/*.json`) and videos (`docs/videos/`) |

## Layout

```
assets/menagerie_wonik_allegro/  Allegro Hand V3 MJCF + meshes, byte-identical to MuJoCo Menagerie (BSD-2)
configs/task/                    allegro_repose.yaml (Isaac Lab values), allegro_repose_hold.yaml (hold design)
configs/algo/ppo.yaml            Isaac Lab's rsl_rl runner config for this task
src/allegro_repose/assets.py     Isaac Lab Allegro setup applied in memory to the Menagerie model
src/allegro_repose/task/         mjlab env, pure-torch MDP (core.py), fingertip kinematics
src/allegro_repose/ppo/          PPO training loop (rsl_rl PPO)
src/allegro_repose/common/       checkpoint/resume, W&B logging, evaluation protocol
scripts/                         train.py, evaluate.py (metrics and videos)
cluster/                         Slurm chains with automatic resume (submit.sh, job.sbatch, status.sh, common.sh,
                                 jobs.txt, site.env.example)
tests/                           assets, kinematics vs MuJoCo, task math
```

## License

Apache-2.0 (see `LICENSE`). Third-party material and attributions are listed in `NOTICE`.
