"""Run directory, checkpointing, resume, preemption signals and W&B logging.

A run lives in ``<runs_root>/<run_name>/``::

    config.yaml          resolved experiment config (written once)
    wandb_id.txt         W&B run id, reused by every resumed chunk
    checkpoints/ckpt_<iter>.pt
    DONE                 written when training reaches max_iterations

Checkpoints are written atomically (tmp file + rename) and contain the learner state,
optimizer(s), iteration, cumulative env steps and wall-clock, and all RNG states, so a
resumed chunk continues the same curves. When a checkpoint fails to load (e.g. a job
killed mid-write before the rename existed), it is renamed ``*.bad`` and the next newest
is tried.

Slurm sends SIGUSR1 a few minutes before the wall-clock limit (``--signal=B:USR1@300``);
``PreemptionFlag`` turns that (and SIGTERM) into a clean "checkpoint and exit" request.
SIGUSR2 does the same and asks ``scripts/train.py`` to exec itself afterwards: the process keeps its
PID, Slurm job and GPU, and resumes on the code now on disk (applies a fix without giving up a slot).
"""

from __future__ import annotations

import os
import random
import re
import signal
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import yaml

EXIT_PREEMPTED = 3  # trainer exit code: checkpointed early, chain should continue


RESTART_IN_PLACE = False  # set by SIGUSR2; scripts/train.py execs itself after the checkpoint


class PreemptionFlag:
    def __init__(self):
        self.requested = False
        self.reason = ""
        for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGUSR2):
            signal.signal(sig, self._handler)

    def _handler(self, signum, _frame):
        global RESTART_IN_PLACE
        self.requested = True
        self.reason = signal.Signals(signum).name
        if signum == signal.SIGUSR2:
            RESTART_IN_PLACE = True
        print(f"[run] received {self.reason}: will checkpoint and "
              f"{'restart in place' if RESTART_IN_PLACE else 'exit'}", flush=True)


def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu() if hasattr(state["torch"], "cpu") else state["torch"])
    if state.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RunDir:
    """``keep_last`` newest checkpoints are kept, plus every iteration divisible by ``keep_every`` (> 0)."""

    def __init__(self, root: str | Path, run_name: str, keep_last: int = 3, keep_every: int = 0,
                 create: bool = True):
        self.path = Path(root) / run_name
        self.ckpt_dir = self.path / "checkpoints"
        if create:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.keep_last = keep_last
        self.keep_every = keep_every

    # ---------------- config ----------------
    def write_config(self, cfg: dict) -> None:
        p = self.path / "config.yaml"
        if not p.exists():
            with open(p, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)

    def read_config(self) -> dict | None:
        p = self.path / "config.yaml"
        if p.exists():
            with open(p) as f:
                return yaml.safe_load(f)
        return None

    @property
    def done(self) -> bool:
        return (self.path / "DONE").exists()

    def mark_done(self, info: str = "") -> None:
        (self.path / "DONE").write_text(info + "\n")

    # ---------------- checkpoints ----------------
    def checkpoints(self) -> list[Path]:
        def it(p: Path) -> int:
            m = re.search(r"ckpt_(\d+)\.pt$", p.name)
            return int(m.group(1)) if m else -1

        return sorted((p for p in self.ckpt_dir.glob("ckpt_*.pt")), key=it, reverse=True)

    def save(self, state: dict, iteration: int) -> Path:
        path = self.ckpt_dir / f"ckpt_{iteration}.pt"
        tmp = path.with_suffix(".pt.tmp")
        torch.save(state, tmp)
        os.replace(tmp, path)
        self._prune()
        return path

    def _prune(self) -> None:
        ckpts = self.checkpoints()
        for p in ckpts[self.keep_last :]:
            it = int(re.search(r"ckpt_(\d+)\.pt$", p.name).group(1))
            if self.keep_every and it % self.keep_every == 0:
                continue
            p.unlink(missing_ok=True)

    def load_latest(self, map_location="cpu") -> tuple[dict | None, Path | None]:
        for p in self.checkpoints():
            try:
                return torch.load(p, map_location=map_location, weights_only=False), p
            except Exception as e:  # noqa: BLE001  (truncated / corrupt file)
                print(f"[run] checkpoint {p} unloadable ({e!r}); renaming to .bad", flush=True)
                p.rename(p.with_suffix(".pt.bad"))
        return None, None


class Clock:
    """Cumulative training wall-clock across chunks (evaluation time excluded)."""

    def __init__(self, offset: float = 0.0):
        self.offset = offset
        self._t0 = time.time()
        self._paused = 0.0

    def pause(self):
        self._pause_t = time.time()

    def resume(self):
        self._paused += time.time() - self._pause_t

    @property
    def elapsed(self) -> float:
        return self.offset + time.time() - self._t0 - self._paused


class WandbLogger:
    """One W&B run per training run, resumed across Slurm chunks."""

    def __init__(self, run_dir: RunDir, cfg: dict, project: str, group: str | None,
                 tags: list[str], mode: str = "online"):
        import wandb

        self.wandb = wandb
        id_file = run_dir.path / "wandb_id.txt"
        if id_file.exists():
            run_id = id_file.read_text().strip()
        else:
            run_id = uuid.uuid4().hex[:12]
            id_file.write_text(run_id)
        self.run = wandb.init(
            project=project,
            entity=os.environ.get("WANDB_ENTITY") or None,
            name=run_dir.run_name,
            id=run_id,
            resume="allow",
            group=group,
            tags=tags,
            config=cfg,
            dir=str(run_dir.path),
            mode=mode,
        )
        wandb.define_metric("train/env_steps")
        wandb.define_metric("*", step_metric="train/env_steps")

    def log(self, metrics: dict) -> None:
        self.wandb.log({k: float(v.item() if isinstance(v, torch.Tensor) else v) for k, v in metrics.items()})

    def log_checkpoint(self, path: Path, iteration: int) -> None:
        self.wandb.log({"checkpoint/iteration": iteration})
        self.run.summary["checkpoint/latest_path"] = str(path)
        self.run.summary["checkpoint/latest_iteration"] = iteration

    def finish(self) -> None:
        self.wandb.finish()


def runs_root() -> Path:
    from allegro_repose.config import REPO_ROOT

    return Path(os.environ.get("AREPOSE_RUNS", REPO_ROOT / "runs"))
