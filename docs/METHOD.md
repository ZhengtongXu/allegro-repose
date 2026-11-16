# Method

PPO trains on the environment described in `docs/ENVIRONMENT.md` and is scored by the evaluation
in `src/allegro_repose/common/evaluate.py`.

## PPO

rsl_rl 5.4 PPO with Isaac Lab's runner configuration for this task (`configs/algo/ppo.yaml`):

| setting | value |
|---|---|
| envs x steps per iteration | 8192 x 16 (131 072 env steps) |
| iterations | 10 000 (1.31 B env steps) |
| actor, critic | MLP 1024-512-256-128, ELU, observation normalization |
| policy | Gaussian, initial std 1.0 |
| optimization | 5 epochs x 4 mini-batches, Adam, learning rate 5e-4 with the adaptive KL schedule (target 0.016), grad norm 1.0 |
| returns | gamma 0.99, lambda 0.95 |
| losses | clip 0.2, entropy 0.005, value loss 1.0 (clipped) |
| actions | absolute joint targets: a in [-1, 1] maps linearly onto each joint's limits |

With the Isaac Lab task the actor and critic both see the 124-D Isaac Lab observation. The hold
design uses an asymmetric actor-critic: an 89-D actor observation with rotation-6D orientations and
no velocities, and a 138-D critic observation with privileged velocities
(`docs/ENVIRONMENT.md`, "Hold variant").

Training runs at about 80 000-110 000 env steps per second on one A100-40GB (3.3-4.7 h per
10 000 iterations), with the fingertip kinematics of the observation compiled by `torch.compile`.

## Evaluation

Deterministic actions (the policy mean), 4096 independent 10 s episodes with a fixed seed. An
episode ends early only when the cube falls; goals reached before the fall count. With a dwell, a
goal counts when it has been held for the dwell. `scripts/evaluate.py --set` scores a policy under a
different rule (e.g. a 1 s dwell for every policy, to compare them on equal terms).
