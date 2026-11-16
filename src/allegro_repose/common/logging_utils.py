"""Training-batch episode statistics, logged under episode/* and repose/* in W&B."""

from __future__ import annotations

from collections import deque

import torch


class EpisodeStats:
    """Tracks completed-episode return/length and the env's per-episode log dict."""

    def __init__(self, num_envs: int, device, window: int = 200):
        self.ret = torch.zeros(num_envs, device=device)
        self.len = torch.zeros(num_envs, device=device)
        self.returns: deque[float] = deque(maxlen=window)
        self.lengths: deque[float] = deque(maxlen=window)
        self.env_log: dict[str, list[torch.Tensor]] = {}

    def step(self, reward: torch.Tensor, done: torch.Tensor, extras: dict) -> None:
        self.ret += reward
        self.len += 1
        ids = done.nonzero().flatten()
        if len(ids):
            self.returns.extend(self.ret[ids].tolist())
            self.lengths.extend(self.len[ids].tolist())
            self.ret[ids] = 0
            self.len[ids] = 0
        for k, v in extras.get("log", {}).items():  # kept on the GPU until summary(): no per-step sync
            self.env_log.setdefault(k, []).append(torch.as_tensor(v, dtype=torch.float32).clone())

    def summary(self) -> dict[str, float]:
        out = {}
        if self.returns:
            out["episode/return"] = sum(self.returns) / len(self.returns)
            out["episode/length"] = sum(self.lengths) / len(self.lengths)
        for k, vals in self.env_log.items():
            name = (
                k.replace("Episode_Reward/", "episode/reward_")
                .replace("Metrics/goal/", "repose/")
                .replace("Episode_Termination/", "episode/termination_")
            )
            out[name] = torch.stack([v.to(vals[0].device) for v in vals]).mean().item()
        self.env_log = {}
        return out
