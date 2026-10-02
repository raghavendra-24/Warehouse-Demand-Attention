"""Training loop and metrics (A-11, A-12, A-14, FAC-27, FAC-28, FAC-35, FAC-38)."""

import math
from dataclasses import replace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from wda import config
from wda.metrics import comparison_table, error_table, errors, row_entropy, score, steps_to
from wda.models import ToyModel, WarehouseModel
from wda.toy_data import make_recall
from wda.train import diagnostics, fit
from wda.warehouse_data import generate_series, make_windows

SMALL = replace(config.TOY_TRAIN, steps=20, eval_every=10, batch_size=32)


def toy_problem(n=256):
    tokens, target = make_recall(config.TOY, n, seed=5)
    return (tokens, target)


def run_fit(cfg=SMALL, model_cfg=config.TOY_MODEL, seed=0, loss_fn=F.cross_entropy):
    data = toy_problem()
    model = ToyModel(model_cfg, seed=seed)
    metric = lambda logits, y: (logits.argmax(-1) == y).float().mean().item()
    return fit(model, data, data, loss_fn, metric, cfg, seed=seed, higher_is_better=True,
               diag=(data[0][:64], data[1][:64]))


def test_fit_is_deterministic_for_a_seed():
    a, b = run_fit(), run_fit()
    assert [r["val_loss"] for r in a["history"]] == [r["val_loss"] for r in b["history"]]
    for k in a["best_state"]:
        assert torch.equal(a["best_state"][k], b["best_state"][k])


def test_first_evaluation_is_before_any_update_and_logs_diagnostics():
    h = run_fit()["history"]
    assert [r["step"] for r in h] == [0, 10, 20]
    assert h[0]["train_loss"] is None
    for key in ("entropy_all", "entropy_readout", "max_weight_readout", "logit_sd",
                "grad_W_Q", "grad_W_K", "grad_W_V", "grad_q_readout_p50"):
        assert key in h[0] and math.isfinite(h[0][key])


def test_non_finite_loss_stops_the_run_with_the_step():
    with pytest.raises(FloatingPointError, match="non-finite loss at step 0"):
        run_fit(loss_fn=lambda pred, y: F.cross_entropy(pred, y) * float("nan"))


def test_paired_arms_differ_only_by_the_switch():
    scaled = run_fit()
    unscaled = run_fit(model_cfg=replace(config.TOY_MODEL, scaled=False))
    # Same data, initial weights and batch order, so step-0 weights are identical but scores differ.
    assert scaled["history"][0]["logit_sd"] != unscaled["history"][0]["logit_sd"]


def test_training_learns_the_toy_task_above_chance():
    data = make_recall(config.TOY, 4000, seed=11)
    model = ToyModel(config.TOY_MODEL, seed=0)
    metric = lambda logits, y: (logits.argmax(-1) == y).float().mean().item()
    out = fit(model, data, data, F.cross_entropy, metric, replace(config.TOY_TRAIN, steps=300, eval_every=100),
              seed=0, higher_is_better=True, diag=(data[0][:64], data[1][:64]))
    assert out["history"][-1]["val_loss"] < out["history"][0]["val_loss"]
    assert out["best_metric"] > 2 / config.TOY.n_values


def test_uniform_control_has_zero_query_gradient():
    data = toy_problem()
    model = ToyModel(replace(config.TOY_MODEL, uniform=True), seed=0)
    d = diagnostics(model, data[0][:32], data[1][:32], F.cross_entropy)
    assert d["grad_W_Q"] == 0.0 and d["grad_q_readout_p90"] == 0.0 and d["grad_W_V"] > 0


def test_row_entropy_is_zero_for_one_hot_and_ln_n_for_uniform():
    one_hot = torch.tensor([[0.0, 1.0, 0.0]])
    assert row_entropy(one_hot).item() == 0.0
    assert math.isclose(row_entropy(torch.full((1, 4), 0.25)).item(), math.log(4), rel_tol=1e-6)


def test_errors_and_groups():
    e = errors(np.array([1.0, 3.0]), np.array([2.0, 2.0]))
    assert e == {"mae": 1.0, "rmse": 1.0, "bias": 0.0, "count": 2}
    t = error_table({"m": np.array([0.0, 10.0, 4.0])}, np.array([1.0, 7.0, 4.0]), np.array(["inside", "other", "other"]))
    assert t["m"]["all"]["count"] == 3 and t["m"]["inside"]["mae"] == 1.0 and t["m"]["other"]["mae"] == 1.5
    assert t["m"]["after"] is None


def test_steps_to_threshold():
    h = [{"step": 0, "acc": 0.1}, {"step": 50, "acc": 0.96}, {"step": 100, "acc": 0.99}]
    assert steps_to(h, "acc", 0.95) == 50 and steps_to(h, "acc", 0.999) is None


def test_score_inverse_standardises_and_includes_every_baseline():
    w = make_windows(generate_series(config.WAREHOUSE))
    w = {k: v[:100] for k, v in w.items()}
    model = WarehouseModel(config.WAREHOUSE_MODEL, seed=0)
    with torch.no_grad():                    # a model that always predicts ẑ = 0 must forecast μ
        model.head.weight.zero_()
        model.head.bias.zero_()
    preds, table = score({"pred_s0": model}, w, mean=100.0, sd=50.0, profile=np.arange(168.0))
    assert set(preds) == {"pred_s0", "B1", "B2", "B3", "B4"}
    np.testing.assert_array_equal(preds["B4"], w["target_day"] * 24 + w["target_hour"])
    np.testing.assert_allclose(preds["pred_s0"], 100.0, rtol=1e-6)
    np.testing.assert_array_equal(preds["B1"], w["raw_window"][:, -1])
    assert table["B3"]["all"]["count"] == 100


def test_comparison_table_difference_is_model_minus_reference():
    table = {"B1": {"all": {"mae": 20.0, "rmse": 25.0}}, "m0": {"all": {"mae": 18.0, "rmse": 24.0}},
             "m1": {"all": {"mae": 18.0, "rmse": 24.0}}}
    text = comparison_table(table, "B1", ["m0", "m1"], [], "t")
    assert "-2.00 (-10.0%)" in text
