"""Run every stage in order (§3, §22; FAC-50, FAC-51).

One command regenerates the dataset and every result:
    python -m experiments.run_all
It stops at the first failing stage and names it, then prints each stage's
runtime (also recorded in results/<stage>/run.json).
"""

import importlib
import json
import sys
import time

from wda import run

STAGES = ["generate_data", "attention_trace", "gradcheck", "stability", "toy", "ablation",
          "warehouse", "shift", "failure"]
RESULT_FOLDER = {"generate_data": "data", "attention_trace": "trace"}


def main():
    started = time.perf_counter()
    for stage in STAGES:
        print(f"\n=== {stage} ===", flush=True)
        try:
            importlib.import_module(f"experiments.{stage}").main()
        except Exception as e:                      # report which stage failed, then stop
            print(f"\nrun_all stopped: stage '{stage}' failed: {type(e).__name__}: {e}", file=sys.stderr)
            raise
    print("\n=== runtimes (s) ===")
    for stage in STAGES:
        info = json.loads((run.RESULTS / RESULT_FOLDER.get(stage, stage) / "run.json").read_text())
        print(f"{stage:16s} {info['runtime_seconds']:8.1f}")
    print(f"{'total':16s} {time.perf_counter() - started:8.1f}")


if __name__ == "__main__":
    main()
