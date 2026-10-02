"""Demo part 1 (§30): the seven attention operations of §7 on the fixed tiny example.

Run: python -m experiments.attention_trace
"""

import torch

from wda import config, run
from wda.attention import attention

STEPS = [
    ("Q", "Q = X W_Q", "query projection"),
    ("K", "K = X W_K", "key projection"),
    ("V", "V = X W_V", "value projection"),
    ("S", "S = Q Kᵀ", "similarity scores"),
    ("S_scaled", "S′ = S / √d_k", "scaling"),
    ("A", "A = softmax(S′), row by row", "attention weights"),
    ("Y", "Y = A V", "output"),
]


def matrix(t: torch.Tensor) -> str:
    return "\n".join("    [" + ", ".join(f"{v:9.6f}" for v in row) + "]" for row in t.tolist())


def main():
    run.start("trace")
    t = {k: torch.tensor(v, dtype=torch.float64) for k, v in config.TINY_EXAMPLE.items()}
    out = attention(t["X"], t["W_Q"], t["W_K"], t["W_V"])
    n, d_model = t["X"].shape
    d_k, d_v = t["W_Q"].shape[1], t["W_V"].shape[1]
    lines = [f"# Attention trace: tiny example (n = {n}, d_model = {d_model}, d_k = {d_k}, d_v = {d_v})", ""]
    for name in ("X", "W_Q", "W_K", "W_V"):
        lines += [f"{name} =", "```", matrix(t[name]), "```", ""]
    for i, (field, formula, meaning) in enumerate(STEPS, start=1):
        value = getattr(out, field)
        lines += [f"{i}. {meaning}: {formula}   shape {tuple(value.shape)}", "```", matrix(value), "```", ""]
    lines.append(f"Rows of A sum to: {[round(s, 12) for s in out.A.sum(dim=-1).tolist()]}")
    run.finish(
        "trace",
        {field: getattr(out, field) for field, _, _ in STEPS},
        "\n".join(lines),
        {"tiny_example": config.TINY_EXAMPLE},
    )


if __name__ == "__main__":
    main()
