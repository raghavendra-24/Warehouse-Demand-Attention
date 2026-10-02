"""Gradient verification: autograd against central finite differences (§9, A-17).

For a scalar function L of some tensors, every entry θ of every tensor gets

    numerical gradient  (L(θ + h) − L(θ − h)) / 2h,

computed in float64, next to the gradient autograd reports. Central
differences have truncation error ∝ h² and round-off error ∝ ε/h; in float64
(ε ≈ 1.1e-16) both are far below the tolerance at h = 1e-6. An entry agrees if
|g_auto − g_num| ≤ atol + rtol·|g_num|, which is `torch.isclose` (A-17).
"""

import itertools

import torch


def gradcheck(fn, tensors: dict, h: float = 1e-6, rtol: float = 1e-6, atol: float = 1e-8) -> list:
    """One row per entry: tensor, index, autograd, numerical, abs/rel difference, agrees."""
    leaves = {k: t.detach().clone().to(torch.float64).requires_grad_(True) for k, t in tensors.items()}
    loss = fn(**leaves)
    # allow_unused: a tensor cut off from the graph (for example by a stray detach)
    # gets gradient zero, which the comparison then reports as a disagreement.
    grads = torch.autograd.grad(loss, list(leaves.values()), allow_unused=True)
    auto = {k: torch.zeros_like(t) if g is None else g for (k, t), g in zip(leaves.items(), grads)}
    rows = []
    with torch.no_grad():
        for name, t in leaves.items():
            for idx in itertools.product(*(range(s) for s in t.shape)):
                original = t[idx].item()
                t[idx] = original + h
                f_plus = fn(**leaves).item()
                t[idx] = original - h
                f_minus = fn(**leaves).item()
                t[idx] = original
                g_num = (f_plus - f_minus) / (2 * h)
                g_auto = auto[name][idx].item()
                diff = abs(g_auto - g_num)
                scale = max(abs(g_auto), abs(g_num))
                rows.append({
                    "tensor": name,
                    "index": idx,
                    "autograd": g_auto,
                    "numerical": g_num,
                    "abs_diff": diff,
                    "rel_diff": diff / scale if scale > 0 else 0.0,
                    "agrees": bool(torch.isclose(torch.tensor(g_auto, dtype=torch.float64),
                                                    torch.tensor(g_num, dtype=torch.float64), rtol=rtol, atol=atol)),
                })
    return rows


def summarise(rows: list) -> dict:
    """Per tensor: number of entries, largest absolute and relative difference, entries that disagree."""
    out = {}
    for name in dict.fromkeys(r["tensor"] for r in rows):
        mine = [r for r in rows if r["tensor"] == name]
        out[name] = {
            "entries": len(mine),
            "max_abs_diff": max(r["abs_diff"] for r in mine),
            "max_rel_diff": max(r["rel_diff"] for r in mine),
            "disagree": sum(not r["agrees"] for r in mine),
        }
    return out
