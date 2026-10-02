"""Gradient verification (§9, §23, A-17, FAC-15, FAC-56)."""

import pytest
import torch

from wda.attention import AttentionWeights, attention
from wda.config import GRADCHECK
from wda.gradcheck import gradcheck, summarise

N, D_MODEL, D_K, D_V = GRADCHECK["n"], GRADCHECK["d_model"], GRADCHECK["d_k"], GRADCHECK["d_v"]


def inputs():
    g = torch.Generator().manual_seed(GRADCHECK["seed"])
    f64 = dict(generator=g, dtype=torch.float64)
    tensors = {
        "X": torch.randn(N, D_MODEL, **f64),
        "W_Q": torch.randn(D_MODEL, D_K, **f64),
        "W_K": torch.randn(D_MODEL, D_K, **f64),
        "W_V": torch.randn(D_MODEL, D_V, **f64),
    }
    R = torch.randn(N, D_V, **f64)   # fixed random weights: L = Σ R ⊙ Y (sum(A) would have zero gradient)
    return tensors, R


def check(loss_fn):
    tensors, _ = inputs()
    return gradcheck(loss_fn, tensors, h=GRADCHECK["h"], rtol=GRADCHECK["rtol"], atol=GRADCHECK["atol"])


def test_autograd_matches_float64_central_differences_for_every_entry():
    _, R = inputs()
    rows = check(lambda X, W_Q, W_K, W_V: (attention(X, W_Q, W_K, W_V).Y * R).sum())
    assert len(rows) == N * D_MODEL + 2 * D_MODEL * D_K + D_MODEL * D_V   # 45 entries
    assert all(r["agrees"] for r in rows)
    assert all(s["disagree"] == 0 for s in summarise(rows).values())


@pytest.mark.parametrize("scaled", [True, False])
def test_gradients_agree_for_both_ablation_arms_on_batched_input(scaled):
    g = torch.Generator().manual_seed(3)
    f64 = dict(generator=g, dtype=torch.float64)
    tensors = {"X": torch.randn(2, N, D_MODEL, **f64), "W_Q": torch.randn(D_MODEL, D_K, **f64),
               "W_K": torch.randn(D_MODEL, D_K, **f64), "W_V": torch.randn(D_MODEL, D_V, **f64)}
    R = torch.randn(2, N, D_V, **f64)
    rows = gradcheck(lambda X, W_Q, W_K, W_V: (attention(X, W_Q, W_K, W_V, scaled=scaled).Y * R).sum(),
                     tensors, h=GRADCHECK["h"], rtol=GRADCHECK["rtol"], atol=GRADCHECK["atol"])
    assert all(r["agrees"] for r in rows)


def test_check_catches_a_broken_autograd_graph():
    """Negative control: detaching W_Q makes autograd report zero, while the function still depends on it."""
    _, R = inputs()
    rows = check(lambda X, W_Q, W_K, W_V: (attention(X, W_Q.detach(), W_K, W_V).Y * R).sum())
    assert summarise(rows)["W_Q"]["disagree"] > 0
    assert summarise(rows)["W_K"]["disagree"] == 0


def test_one_step_trains_all_three_projections():
    """FAC-15: finite, non-zero gradients for W_Q, W_K, W_V, and all three change after one step."""
    g = torch.Generator().manual_seed(0)
    weights = AttentionWeights(D_MODEL, D_K, D_V, 1.0, g)
    X = torch.randn(4, N, D_MODEL, generator=g)
    R = torch.randn(4, N, D_V, generator=g)
    before = {k: v.detach().clone() for k, v in weights.named_parameters()}
    assert {name for name, _ in weights.named_parameters()} == {"W_Q", "W_K", "W_V"}
    loss = (weights(X).Y * R).sum()
    loss.backward()
    for name, p in weights.named_parameters():
        assert torch.isfinite(p.grad).all() and p.grad.norm() > 0, name
    torch.optim.Adam(weights.parameters(), lr=1e-2).step()
    for name, p in weights.named_parameters():
        assert not torch.equal(p.detach(), before[name]), name
