"""Attention core: each test is named after the bug it would catch (§23, ARCHITECTURE §11)."""

import math

import pytest
import torch

from wda import config
from wda.attention import AttentionWeights, attention, stable_softmax
from wda.config import GRADCHECK, TINY_EXAMPLE
from wda.models import WarehouseModel
from wda.toy_data import make_recall

N, D_MODEL, D_K, D_V = GRADCHECK["n"], GRADCHECK["d_model"], GRADCHECK["d_k"], GRADCHECK["d_v"]


def random_inputs(batch=None, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    shape = (N, D_MODEL) if batch is None else (batch, N, D_MODEL)
    X = torch.randn(*shape, generator=g, dtype=dtype)
    W_Q, W_K = (torch.randn(D_MODEL, D_K, generator=g, dtype=dtype) for _ in range(2))
    W_V = torch.randn(D_MODEL, D_V, generator=g, dtype=dtype)
    return X, W_Q, W_K, W_V


@pytest.mark.parametrize("batch", [None, 3])
def test_shapes_catch_swapped_dimensions_or_missing_transpose(batch):
    out = attention(*random_inputs(batch))
    lead = () if batch is None else (batch,)
    assert out.Q.shape == out.K.shape == (*lead, N, D_K)
    assert out.V.shape == (*lead, N, D_V)
    assert out.S.shape == out.S_scaled.shape == out.A.shape == (*lead, N, N)
    assert out.Y.shape == (*lead, N, D_V)


def test_softmax_rows_sum_to_one_catches_wrong_axis():
    A = attention(*random_inputs(batch=4)).A
    torch.testing.assert_close(A.sum(dim=-1), torch.ones(4, N, dtype=A.dtype))
    assert (A >= 0).all()
    # The check is not vacuous: with random scores the columns do not sum to 1.
    assert not torch.allclose(A.sum(dim=-2), torch.ones(4, N, dtype=A.dtype))


def test_stable_softmax_does_not_overflow_on_large_logits():
    s = torch.tensor([1000.0, 999.0, 0.0])            # float32: exp(1000) = inf
    assert not torch.isfinite(torch.exp(s)).all()      # the naive form would overflow
    a = stable_softmax(s)
    assert torch.isfinite(a).all()
    expected = torch.tensor([1.0, math.exp(-1.0), 0.0]) / (1.0 + math.exp(-1.0))
    torch.testing.assert_close(a, expected)


def test_stable_softmax_matches_naive_on_moderate_logits():
    """Only moderate logits: near float32 overflow, naive softmax can be finite and still wrong (V2, results/stability)."""
    s = torch.randn(6, 7, generator=torch.Generator().manual_seed(1), dtype=torch.float64) * 5
    naive = torch.exp(s) / torch.exp(s).sum(dim=-1, keepdim=True)
    torch.testing.assert_close(stable_softmax(s), naive)


def test_stable_softmax_is_correct_where_naive_is_finite_but_wrong():
    """V2's band: at [88.5, 87.5, 0] in float32 each exp is finite but their sum overflows."""
    s = torch.tensor([88.5, 87.5, 0.0])
    naive = torch.exp(s) / torch.exp(s).sum()
    assert torch.isfinite(naive).all() and naive.sum() == 0          # finite, and wrong
    expected = torch.tensor([1.0, math.exp(-1.0), math.exp(-88.5)]) / (1.0 + math.exp(-1.0) + math.exp(-88.5))
    torch.testing.assert_close(stable_softmax(s), expected)


def tiny():
    return {k: torch.tensor(v, dtype=torch.float64) for k, v in TINY_EXAMPLE.items()}


def test_forward_matches_hand_computation_catches_kqt_or_missing_scale():
    """Every value below was worked out by hand from TINY_EXAMPLE (see docs/DERIVATION.md)."""
    t = tiny()
    out = attention(t["X"], t["W_Q"], t["W_K"], t["W_V"])
    f64 = dict(dtype=torch.float64)
    torch.testing.assert_close(out.Q, torch.tensor([[2.0, 1, 1, 0], [1, 3, 0, 2]], **f64))
    torch.testing.assert_close(out.K, torch.tensor([[1.0, 1, 1, 0], [0, 1, 0, 1]], **f64))
    torch.testing.assert_close(out.V, torch.tensor([[1.0], [4.0]], **f64))
    torch.testing.assert_close(out.S, torch.tensor([[4.0, 1], [4, 5]], **f64))   # asymmetric
    torch.testing.assert_close(out.S_scaled, torch.tensor([[2.0, 0.5], [2.0, 2.5]], **f64))
    sigmoid = lambda z: 1.0 / (1.0 + math.exp(-z))     # softmax of two logits z apart
    a, c = sigmoid(1.5), sigmoid(0.5)                  # row 1: 2 vs 0.5; row 2: 2 vs 2.5
    torch.testing.assert_close(out.A, torch.tensor([[a, 1 - a], [1 - c, c]], **f64))   # asymmetric
    expected_Y = torch.tensor([[a * 1 + (1 - a) * 4], [(1 - c) * 1 + c * 4]], **f64)
    torch.testing.assert_close(out.Y, expected_Y)
    # Removing 1/√d_k must break the match (this is what makes the test meaningful).
    unscaled = attention(t["X"], t["W_Q"], t["W_K"], t["W_V"], scaled=False)
    assert not torch.allclose(unscaled.Y, expected_Y)


def test_matches_elementwise_reference_catches_kqt_on_any_input():
    """s_ij = Σ_d q_i[d] k_j[d], computed entry by entry, independent of the matrix code."""
    X, W_Q, W_K, W_V = random_inputs()
    out = attention(X, W_Q, W_K, W_V)
    S = torch.tensor([[sum(out.Q[i, d] * out.K[j, d] for d in range(D_K)) for j in range(N)]
                      for i in range(N)], dtype=torch.float64)
    torch.testing.assert_close(out.S, S)
    S_scaled = S / math.sqrt(D_K)
    A = torch.tensor([[math.exp(S_scaled[i, j] - S_scaled[i].max()) for j in range(N)] for i in range(N)],
                     dtype=torch.float64)
    A = A / A.sum(dim=1, keepdim=True)
    Y = torch.tensor([[sum(A[i, j] * out.V[j, c] for j in range(N)) for c in range(D_V)] for i in range(N)],
                     dtype=torch.float64)
    torch.testing.assert_close(out.A, A)
    torch.testing.assert_close(out.Y, Y)


def test_permutation_equivariance_catches_position_leaking_into_the_core():
    X, W_Q, W_K, W_V = random_inputs()
    perm = torch.tensor([3, 0, 4, 1, 2])
    torch.testing.assert_close(attention(X[perm], W_Q, W_K, W_V).Y, attention(X, W_Q, W_K, W_V).Y[perm])


def test_unscaled_equals_scaled_with_rescaled_query_weights():
    """softmax(QKᵀ) = softmax((√d_k Q)Kᵀ/√d_k): the switch changes only the divisor (A-19)."""
    X, W_Q, W_K, W_V = random_inputs(batch=2)
    unscaled = attention(X, W_Q, W_K, W_V, scaled=False)
    rescaled = attention(X, W_Q * math.sqrt(D_K), W_K, W_V, scaled=True)
    torch.testing.assert_close(unscaled.Y, rescaled.Y)
    torch.testing.assert_close(unscaled.A, rescaled.A)


def test_uniform_switch_gives_the_mean_of_the_values():
    X, W_Q, W_K, W_V = random_inputs(batch=2)
    out = attention(X, W_Q, W_K, W_V, uniform=True)
    expected = out.V.mean(dim=-2, keepdim=True).expand_as(out.Y)
    torch.testing.assert_close(out.Y, expected)


def test_batched_values_equal_per_sequence_values():
    X, W_Q, W_K, W_V = random_inputs(batch=3)
    batched = attention(X, W_Q, W_K, W_V).Y
    for b in range(3):
        torch.testing.assert_close(batched[b], attention(X[b], W_Q, W_K, W_V).Y)


def test_toy_initialisation_gives_unit_variance_queries():
    """A-19 with the configured scale: ‖x‖² = 2 for toy tokens, so q components have variance ≈ 1."""
    m = config.TOY_MODEL
    w = AttentionWeights(m.d_model, 64, m.d_v, m.qkv_init_std, torch.Generator().manual_seed(0))
    tokens, _ = make_recall(config.TOY, 500, seed=1)
    q = (tokens.reshape(-1, m.d_model) @ w.W_Q).detach()
    assert 0.8 < q.var().item() < 1.2


def test_warehouse_initialisation_chain_gives_unit_variance_queries():
    """A-19: E‖f‖² = 3 and W_in ~ N(0, 1/3) give unit-variance X; W ~ N(0, 1/16) then gives unit-variance q."""
    torch.manual_seed(0)
    model = WarehouseModel(config.WAREHOUSE_MODEL, seed=0)
    angles = torch.rand(4000, 2) * 2 * math.pi
    f = torch.stack([torch.randn(4000), angles[:, 0].sin(), angles[:, 0].cos(),
                     angles[:, 1].sin(), angles[:, 1].cos()], dim=1)
    with torch.no_grad():
        X = model.inp(f)
        q = X @ model.attn.W_Q
    assert 0.8 < X.var().item() < 1.25 and 0.75 < q.var().item() < 1.3
