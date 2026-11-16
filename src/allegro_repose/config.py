"""YAML config loading with dotted command-line overrides.

An experiment is the composition of two files::

    task     configs/task/<name>.yaml      environment: hand setup, cube, rewards, resets
    algo     configs/algo/<name>.yaml      learning algorithm (ppo)

Overrides use ``section.key.sub=value`` (value parsed as YAML), e.g. ``algo.max_iterations=20000``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _load_yaml(section: str, name: str) -> dict:
    """Load a config; a top-level ``_base: <name>`` loads that config first and overrides it."""
    path = Path(name)
    if path.suffix not in (".yaml", ".yml"):
        path = CONFIG_DIR / section / f"{name}.yaml"
    with open(resolve_path(path)) as f:
        data = yaml.safe_load(f) or {}
    if "_base" in data:
        data = _merge(_load_yaml(section, data.pop("_base")), data)
    return data


def apply_override(cfg: dict, dotted: str) -> None:
    key, sep, raw = dotted.partition("=")
    if not sep:
        raise ValueError(f"override must look like a.b.c=value, got {dotted!r}")
    parts = key.strip().split(".")
    node = cfg
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = yaml.safe_load(raw)


def load_experiment(task: str = "allegro_repose", algo: str = "ppo",
                    overrides: list[str] | None = None) -> dict[str, Any]:
    cfg = {"task": _load_yaml("task", task), "algo": _load_yaml("algo", algo)}
    for o in overrides or []:
        apply_override(cfg, o)
    return cfg
