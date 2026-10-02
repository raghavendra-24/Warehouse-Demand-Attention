"""Start and finish of every experiment script, and weight files (ARCHITECTURE §3).

`start` makes a run repeatable and clears the stage's old outputs, so a failed
rerun can never leave stale files that look current. `finish` writes:
- metrics.json: configs and results only, so a rerun on the same machine
  reproduces it exactly and `git diff` can show it;
- run.json: runtime, versions, hardware, code revision and FINAL_TEST, which
  legitimately change between runs;
- table.md: the summary table, also printed.
"""

import json
import math
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from wda import config

RESULTS = Path(__file__).resolve().parents[1] / "results"

_started = {}


def start(stage: str) -> Path:
    """Fix threads and determinism, empty results/<stage>/, start the timer."""
    if not re.fullmatch(r"[a-z][a-z0-9_]*", stage):
        raise ValueError(f"invalid stage name {stage!r}: lower-case letters, digits and underscores only")
    torch.set_num_threads(config.NUM_THREADS)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)  # parameters and batch order use their own seeded generators
    out = RESULTS / stage
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    _started[stage] = time.perf_counter()
    return out


def environment() -> dict:
    """Versions and hardware recorded in every metrics file (§22, A-25)."""
    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cpu": cpu,
        "threads": config.NUM_THREADS,
        "final_test": config.FINAL_TEST,
        "git": _git_revision(),
    }


def _git_revision() -> str:
    """Commit hash, with '+dirty' if there are uncommitted changes; 'unknown' outside Git."""
    root = RESULTS.parent
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                               capture_output=True, text=True, check=True).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def finish(stage: str, metrics: dict, table: str, configs: dict) -> None:
    """Write metrics.json, run.json and table.md for the stage, and print the table.

    `configs` maps a name (one per arm in multi-arm stages) to the resolved
    configuration it ran with, so arms can be compared directly (FAC-28).
    Non-finite numbers are written as null (JSON has no NaN).
    """
    out = RESULTS / stage
    runtime = round(time.perf_counter() - _started[stage], 1)
    record = {"stage": stage, "configs": configs, "metrics": metrics}
    _write_json(out / "metrics.json", record)
    _write_json(out / "run.json", {"stage": stage, "runtime_seconds": runtime, "environment": environment()})
    (out / "table.md").write_text(table.rstrip() + "\n")
    print(table)
    print(f"[{stage}] done in {runtime} s -> {out}")


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(_clean(obj), indent=2, allow_nan=False) + "\n")


def save_weights(path: Path, state: dict, extra: dict) -> None:
    """Save model weights plus plain values needed to reuse them (μ, s, the model config).

    `extra` is converted to plain Python values first, because the safe loader
    (weights_only=True) accepts nothing else.
    """
    torch.save({"state_dict": state, "extra": _clean(extra)}, path)


def load_weights(path: Path) -> dict:
    """Load a weight file without unpickling arbitrary objects."""
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run the stage that writes it first")
    return torch.load(path, weights_only=True)


def _clean(value):
    """Plain JSON-compatible Python values: dataclasses, NumPy and tensors converted; NaN/inf → None."""
    if hasattr(value, "__dataclass_fields__"):
        value = {k: getattr(value, k) for k in value.__dataclass_fields__}
    if isinstance(value, (np.ndarray, torch.Tensor)):
        value = value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"cannot write {type(value).__name__} to JSON")
