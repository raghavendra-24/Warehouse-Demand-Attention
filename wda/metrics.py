"""Metrics, attention diagnostics and the shared warehouse scoring (A-11, A-14, A-15, FAC-35).

`score` is the only scoring code: warehouse, shift and failure all use it, so
every model, control and baseline is scored the same way, in orders per hour.
"""

import math

import numpy as np
import torch

from wda.baselines import BASELINES, hour_of_week_mean
from wda.warehouse_data import model_inputs

GROUPS = ("all", "inside", "after", "other")


def row_entropy(A: torch.Tensor) -> torch.Tensor:
    """Entropy of each attention row in nats; entr(0) = 0, so one-hot rows give exactly 0."""
    return torch.special.entr(A).sum(dim=-1)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == targets).float().mean().item()


def steps_to(history: list, key: str, threshold: float):
    """First evaluated step at which history[key] ≥ threshold, or None if never reached."""
    return next((r["step"] for r in history if r[key] >= threshold), None)


def errors(pred: np.ndarray, target: np.ndarray) -> dict:
    e = pred - target
    return {"mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
            "bias": float(e.mean()), "count": int(len(e))}


def error_table(predictions: dict, target: np.ndarray, group: np.ndarray) -> dict:
    """MAE, RMSE, mean signed error and count for every prediction column, overall and per A-11 group."""
    out = {}
    for name, pred in predictions.items():
        out[name] = {}
        for g in GROUPS:
            mask = np.ones(len(target), bool) if g == "all" else group == g
            out[name][g] = errors(pred[mask], target[mask]) if mask.any() else None
    return out


@torch.no_grad()
def predict(model, X: torch.Tensor, mean: float, sd: float) -> np.ndarray:
    """Forecasts in orders per hour: ŷ = μ + s·ẑ (A-13)."""
    z, _ = model(X)
    return (mean + sd * z).double().numpy()


def score(models: dict, windows: dict, mean: float, sd: float, profile: np.ndarray) -> tuple:
    """Per-window predictions for every model and baseline (B1–B4), and their error table.

    μ, s and the B4 hour-of-week profile all come from the training targets (A-13).
    """
    X = model_inputs(windows, mean, sd)
    predictions = {name: predict(m, X, mean, sd) for name, m in models.items()}
    predictions.update({b: f(windows["raw_window"]) for b, f in BASELINES.items()})
    predictions["B4"] = hour_of_week_mean(windows, profile)
    return predictions, error_table(predictions, windows["target"], windows["group"])


def comparison_table(table: dict, reference: str, model_cols: list, control_cols: list, title: str) -> str:
    """Rows MAE and RMSE; Reference / Attention model (mean ± SD, every seed) / Uniform control / Difference.

    Difference = model − reference, in orders per hour and %; negative means the model is better (A-15).
    """
    lines = [f"### {title}", "",
             f"| Metric | Reference ({reference}) | Attention model | Uniform control | Difference (model − reference) |",
             "|---|---|---|---|---|"]
    for metric in ("mae", "rmse"):
        ref = table[reference]["all"][metric]
        model = np.array([table[c]["all"][metric] for c in model_cols])
        ctrl = np.array([table[c]["all"][metric] for c in control_cols]) if control_cols else None
        diff = model.mean() - ref
        seeds = ", ".join(f"{v:.2f}" for v in model)
        ctrl_txt = f"{ctrl.mean():.2f} ± {ctrl.std():.2f}" if ctrl is not None else "—"
        lines.append(f"| {metric.upper()} | {ref:.2f} | {model.mean():.2f} ± {model.std():.2f} ({seeds}) | "
                     f"{ctrl_txt} | {diff:+.2f} ({100 * diff / ref:+.1f}%) |")
    return "\n".join(lines)


def group_table(table: dict, columns: list) -> str:
    """MAE per A-11 group, with counts, for the given columns."""
    lines = ["| Group | Count | " + " | ".join(columns) + " |", "|---|---|" + "---|" * len(columns)]
    for g in GROUPS:
        first = table[columns[0]][g]
        if first is None:
            continue
        cells = " | ".join(f"{table[c][g]['mae']:.2f}" for c in columns)
        lines.append(f"| {g} | {first['count']} | {cells} |")
    return "\n".join(lines)


def entropy_fraction(A: torch.Tensor) -> torch.Tensor:
    """Row entropy as a fraction of its maximum ln n (H2)."""
    return row_entropy(A) / math.log(A.shape[-1])


@torch.no_grad()
def attention_rows(models: dict, pred_cols, w: dict, mean: float, sd: float) -> dict:
    """H8 / FAC-33: the readout row of A (lags t−23 … t) averaged over windows, per seed and per group."""
    X = model_inputs(w, mean, sd)
    out = {}
    for col in pred_cols:
        _, att = models[col](X)
        rows = att.A[:, -1, :].numpy()
        out[col] = {"all": rows.mean(axis=0).tolist(),
                    **{g: rows[w["group"] == g].mean(axis=0).tolist() for g in ("inside", "after", "other")
                       if (w["group"] == g).any()},
                    "weight_on_t_minus_23_and_t": float(rows[:, 0].mean() + rows[:, -1].mean())}
    return out
