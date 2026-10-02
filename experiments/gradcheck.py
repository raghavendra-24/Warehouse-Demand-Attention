"""Demo part 2 (§30): gradient verification, autograd against float64 central differences (§9, A-17, V1).

Run: python -m experiments.gradcheck
"""

import torch

from wda import config, run
from wda.attention import attention
from wda.gradcheck import gradcheck, summarise


def main():
    run.start("gradcheck")
    c = config.GRADCHECK
    g = torch.Generator().manual_seed(c["seed"])
    f64 = dict(generator=g, dtype=torch.float64)
    tensors = {
        "X": torch.randn(c["n"], c["d_model"], **f64),
        "W_Q": torch.randn(c["d_model"], c["d_k"], **f64),
        "W_K": torch.randn(c["d_model"], c["d_k"], **f64),
        "W_V": torch.randn(c["d_model"], c["d_v"], **f64),
    }
    R = torch.randn(c["n"], c["d_v"], **f64)

    def loss(X, W_Q, W_K, W_V):
        return (attention(X, W_Q, W_K, W_V).Y * R).sum()   # L = Σ R ⊙ Y

    rows = gradcheck(loss, tensors, h=c["h"], rtol=c["rtol"], atol=c["atol"])
    summary = summarise(rows)
    unexplained = sum(not r["agrees"] for r in rows)

    lines = [
        f"# Gradient check: L = Σ R ⊙ Y, float64, central differences, h = {c['h']}",
        f"Agreement: |autograd − numerical| ≤ {c['atol']} + {c['rtol']}·|numerical| (A-17)",
        "",
        "| Tensor | Entries | Max abs. difference | Max rel. difference | Disagreeing |",
        "|---|---|---|---|---|",
    ]
    lines += [f"| {k} | {s['entries']} | {s['max_abs_diff']:.2e} | {s['max_rel_diff']:.2e} | {s['disagree']} |"
              for k, s in summary.items()]
    lines += ["", "| Tensor | Index | Autograd | Numerical | Abs. diff | Rel. diff | Agrees |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {r['tensor']} | {r['index']} | {r['autograd']:+.10f} | {r['numerical']:+.10f} | "
              f"{r['abs_diff']:.2e} | {r['rel_diff']:.2e} | {'yes' if r['agrees'] else 'NO'} |" for r in rows]
    lines += ["", f"Unexplained entries: {unexplained}"]
    run.finish("gradcheck", {"summary": summary, "unexplained": unexplained, "rows": rows},
               "\n".join(lines), {"gradcheck": c})


if __name__ == "__main__":
    main()
