"""PPO: rsl_rl's PPO algorithm in a resumable loop (checkpoint / resume, in-loop evaluation, W&B)."""

from __future__ import annotations

import copy
import time

import torch
from mjlab.rl import RslRlVecEnvWrapper
from tensordict import TensorDict

from allegro_repose.common.evaluate import evaluate_policy
from allegro_repose.common.logging_utils import EpisodeStats
from allegro_repose.common.run import EXIT_PREEMPTED, Clock, PreemptionFlag, RunDir, rng_state, set_rng_state


def rsl_cfg(a: dict) -> dict:
    return {
        "num_steps_per_env": a["num_steps_per_env"],
        "obs_groups": {"actor": ["actor"], "critic": ["critic"]},
        "multi_gpu": None,
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": list(a["actor_hidden"]),
            "activation": a["activation"],
            "obs_normalization": a["obs_normalization"],
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": a["init_std"], "std_type": "scalar"},
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": list(a["critic_hidden"]),
            "activation": a["activation"],
            "obs_normalization": a["obs_normalization"],
        },
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": a["num_learning_epochs"],
            "num_mini_batches": a["num_mini_batches"],
            "clip_param": a["clip_param"],
            "gamma": a["gamma"],
            "lam": a["lam"],
            "value_loss_coef": a["value_loss_coef"],
            "entropy_coef": a["entropy_coef"],
            "learning_rate": a["learning_rate"],
            "max_grad_norm": a["max_grad_norm"],
            "schedule": a["schedule"],
            "desired_kl": a["desired_kl"],
            "use_clipped_value_loss": True,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
    }


def build_ppo(cfg: dict, env, device: str):
    venv = RslRlVecEnvWrapper(env, clip_actions=cfg["task"]["action"].get("clip", 1.0))
    obs = venv.get_observations().to(device)
    from rsl_rl.algorithms import PPO

    return venv, PPO.construct_algorithm(obs, venv, copy.deepcopy(rsl_cfg(cfg["algo"])), device)


def deterministic_policy(alg):
    pol = alg.get_policy()
    return lambda o: pol(TensorDict({"actor": o}, batch_size=[o.shape[0]]))


def train_ppo(cfg: dict, env, eval_env, run: RunDir, logger, device: str, max_wall_s: float | None) -> int:
    a = cfg["algo"]
    venv, alg = build_ppo(cfg, env, device)
    start_it, env_steps, clock_offset = 0, 0, 0.0
    state, path = run.load_latest(map_location=device)
    if state is not None:
        alg.load(state["ppo"], None, strict=True)
        alg.learning_rate = state["learning_rate"]
        for g in alg.optimizer.param_groups:
            g["lr"] = alg.learning_rate
        start_it, env_steps, clock_offset = state["iteration"] + 1, state["env_steps"], state["wall_time"]
        set_rng_state(state["rng"])
        env.common_step_counter = state.get("common_step_counter", 0)
        print(f"[ppo] resumed from {path} at iteration {start_it}", flush=True)
    clock = Clock(clock_offset)
    flag = PreemptionFlag()
    stats = EpisodeStats(env.num_envs, device)
    chunk_t0 = time.time()

    def checkpoint(it):
        p = run.save({"ppo": alg.save(), "learning_rate": alg.learning_rate, "iteration": it,
                      "env_steps": env_steps, "wall_time": clock.elapsed, "rng": rng_state(),
                      "common_step_counter": env.common_step_counter, "config": cfg}, it)
        logger.log_checkpoint(p, it)

    def evaluate() -> dict:  # evaluation time is excluded from the training clock
        clock.pause()
        alg.eval_mode()
        ev = evaluate_policy(eval_env, deterministic_policy(alg))
        alg.train_mode()
        clock.resume()
        return ev

    obs = venv.get_observations().to(device)
    alg.train_mode()
    it = start_it - 1
    for it in range(start_it, a["max_iterations"]):
        t0 = time.time()
        with torch.inference_mode():
            for _ in range(a["num_steps_per_env"]):
                actions = alg.act(obs)
                obs, rew, dones, extras = venv.step(actions.to(env.device))
                obs, rew, dones = obs.to(device), rew.to(device), dones.to(device)
                alg.process_env_step(obs, rew, dones, extras)
                stats.step(rew, dones.bool(), extras)
            alg.compute_returns(obs)
        t1 = time.time()
        losses = alg.update()
        t2 = time.time()
        env_steps += env.num_envs * a["num_steps_per_env"]
        metrics = {
            "train/env_steps": env_steps, "train/iteration": it, "train/wall_time_s": clock.elapsed,
            "perf/collect_s": t1 - t0, "perf/learn_s": t2 - t1,
            "perf/env_steps_per_s": env.num_envs * a["num_steps_per_env"] / (t2 - t0),
            "ppo/learning_rate": alg.learning_rate, "ppo/action_std": alg.get_policy().output_std.mean(),
            **{f"loss/{k}": v for k, v in losses.items()},
            **stats.summary(),
        }
        if eval_env is not None and (it % a["eval_interval"] == 0 or it == a["max_iterations"] - 1):
            ev = evaluate()
            metrics.update(ev)
            print(f"[ppo] eval {ev}", flush=True)
        logger.log(metrics)
        if it % 10 == 0:
            print(f"[ppo] it {it} steps {env_steps:.3e} ret {metrics.get('episode/return', float('nan')):.1f} "
                  f"succ {metrics.get('repose/successes', float('nan')):.2f} "
                  f"rot {metrics.get('repose/rot_dist', float('nan')):.3f} "
                  f"({metrics['perf/env_steps_per_s']:.0f} sps)", flush=True)
        if it % a["save_interval"] == 0 and it > start_it:
            checkpoint(it)
        if flag.requested or (max_wall_s is not None and time.time() - chunk_t0 > max_wall_s):
            checkpoint(it)
            print(f"[ppo] stopping at iteration {it} ({flag.reason or 'chunk time budget'})", flush=True)
            return EXIT_PREEMPTED

    if eval_env is not None:
        final = evaluate()
        logger.log({"train/env_steps": env_steps, "train/wall_time_s": clock.elapsed,
                    **{k.replace("eval/", "final_eval/"): v for k, v in final.items()}})
    checkpoint(it)
    run.mark_done(f"iterations={it + 1} env_steps={env_steps} wall_s={clock.elapsed:.0f}")
    return 0
