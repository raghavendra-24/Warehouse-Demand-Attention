"""Start and finish of every experiment script, and weight files (ARCHITECTURE §3).

`start` makes a run repeatable and clears the stage's old outputs, so a failed
rerun can never leave stale files that look current. `finish` writes the
stage's metrics and summary table. Nothing else is shared between scripts.
"""

import json
import platform
import shutil
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
    }


def finish(stage: str, metrics: dict, table: str, configs: dict) -> None:
    """Write metrics.json and table.md for the stage, and print the table.

    `configs` maps a name (one per arm in multi-arm stages) to the resolved
    configuration it ran with, so arms can be compared directly (FAC-28).
    """
    out = RESULTS / stage
    record = {
        "stage": stage,
        "runtime_seconds": round(time.perf_counter() - _started[stage], 1),
        "environment": environment(),
        "configs": configs,
        "metrics": metrics,
    }
    (out / "metrics.json").write_text(json.dumps(record, indent=2, default=_jsonable) + "\n")
    (out / "table.md").write_text(table.rstrip() + "\n")
    print(table)
    print(f"[{stage}] done in {record['runtime_seconds']} s -> {out}")


def save_weights(path: Path, state: dict, extra: dict) -> None:
    """Save model weights plus plain values needed to reuse them (for example μ, s)."""
    torch.save({"state_dict": state, "extra": extra}, path)


def load_weights(path: Path) -> dict:
    """Load a weight file without unpickling arbitrary objects."""
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run the stage that writes it first")
    return torch.load(path, weights_only=True)


def _jsonable(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return value.tolist()
    if hasattr(value, "__dataclass_fields__"):
        return {k: getattr(value, k) for k in value.__dataclass_fields__}
    raise TypeError(f"cannot write {type(value).__name__} to JSON")
