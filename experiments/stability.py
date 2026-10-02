"""Numerical stability of softmax in float32 (§11, V2, FAC-21).

Naive softmax exp(s)/Σexp(s) against the max-subtracted version used in the
attention core. Saturation at large logits is analysed from the ablation logs,
not here (a scope decision: no separate sweep).

Run: python -m experiments.stability
"""

import torch

from wda import run
from wda.attention import stable_softmax

# 88.5 sits in the band where each exp is finite but their sum overflows (V2's third claim).
MAX_LOGITS = (80.0, 85.0, 88.0, 88.5, 89.0, 90.0, 100.0, 1000.0)


def naive_softmax(s: torch.Tensor) -> torch.Tensor:
    return torch.exp(s) / torch.exp(s).sum(dim=-1, keepdim=True)


def main():
    run.start("stability")
    rows = []
    for m in MAX_LOGITS:
        s = torch.tensor([m, m - 1.0, 0.0], dtype=torch.float32)
        naive, stable = naive_softmax(s), stable_softmax(s)
        finite = bool(torch.isfinite(naive).all())
        rows.append({
            "logits": s.tolist(),
            "naive": naive.tolist(),
            "stable": stable.tolist(),
            "naive_finite": finite,
            "stable_sum": stable.sum().item(),
            "max_diff_where_finite": (naive - stable).abs().max().item() if finite else None,
            "naive_correct": finite and (naive - stable).abs().max().item() <= 1e-6,
        })
    lines = [
        "# Softmax in float32: naive exp(s)/Σexp(s) vs max-subtracted",
        "",
        "| Logits | Naive | Stable | Naive finite | Naive correct (≤ 1e-6) | Stable sums to | Max difference |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        fmt = lambda v: "[" + ", ".join(f"{x:.4g}" for x in v) + "]"
        diff = f"{r['max_diff_where_finite']:.1e}" if r["naive_finite"] else "—"
        lines.append(f"| {fmt(r['logits'])} | {fmt(r['naive'])} | {fmt(r['stable'])} | "
                     f"{'yes' if r['naive_finite'] else 'no'} | {'yes' if r['naive_correct'] else 'no'} | "
                     f"{r['stable_sum']:.7f} | {diff} |")
    lines += ["", "float32 exp overflows above ln(3.4e38) ≈ 88.72; the stable form keeps every exponent ≤ 0.",
              "Finite is not the same as correct: when each exp is finite but their sum overflows, naive softmax",
              "returns finite zeros."]
    run.finish("stability", {"rows": rows}, "\n".join(lines), {"max_logits": MAX_LOGITS, "dtype": "float32"})


if __name__ == "__main__":
    main()
