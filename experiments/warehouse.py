"""Demo part 4 (§30): the warehouse model against the baselines (§15–§18; A-12, A-15, A-27; H5–H8).

Trains the attention model and its uniform-attention control, three seeds each,
keeping the best-validation-MAE checkpoint. The reference baseline is chosen on
validation from B1–B4. While config.FINAL_TEST is False, only validation is
scored. When it is True, the test split is also scored, once, and the
validation outputs are kept alongside.

Run: python -m experiments.warehouse
"""

from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from wda import config, run
from wda.baselines import hour_of_week_profile
from wda.metrics import attention_rows, comparison_table, group_table, score
from wda.models import WarehouseModel
from wda.train import fit
from wda.warehouse_data import generate_series, make_windows, model_inputs, select, split_masks, standardise, training_stats

ARMS = {"pred": config.WAREHOUSE_MODEL, "ctrl": replace(config.WAREHOUSE_MODEL, uniform=True)}
BASELINE_NAMES = ("B1", "B2", "B3", "B4")


def z_mae(pred, target):
    """Checkpoint metric: MAE on the standardised scale (proportional to MAE in orders/h)."""
    return (pred - target).abs().mean().item()


def main():
    out = run.start("warehouse")
    windows = make_windows(generate_series(config.WAREHOUSE))
    masks = split_masks(windows)
    mean, sd = training_stats(windows, masks)
    profile = hour_of_week_profile(windows, masks["train"])
    X = model_inputs(windows, mean, sd)
    z = torch.from_numpy(standardise(windows["target"], mean, sd)).float()
    tr, va = torch.from_numpy(masks["train"]), torch.from_numpy(masks["val"])
    diag = (X[va][:config.DIAG_BATCH_SIZE], z[va][:config.DIAG_BATCH_SIZE])

    models, histories = {}, {}
    for arm, model_cfg in ARMS.items():
        for seed in config.WAREHOUSE_TRAIN.seeds:
            name = f"{arm}_s{seed}"
            model = WarehouseModel(model_cfg, seed=seed)
            r = fit(model, (X[tr], z[tr]), (X[va], z[va]), F.mse_loss, z_mae, config.WAREHOUSE_TRAIN, seed,
                    higher_is_better=False, diag=diag)
            model.load_state_dict(r["best_state"])
            model.eval()
            models[name] = model
            histories[name] = {"best_step": r["best_step"], "best_val_z_mae": r["best_metric"], "history": r["history"]}
            run.save_weights(out / f"weights_{name}.pt", model.state_dict(),
                             {"mean": mean, "sd": sd, "profile": profile, "model": model_cfg, "seed": seed, "arm": arm})
            print(f"{name}: best step {r['best_step']}, validation MAE (z) {r['best_metric']:.4f}")

    splits = ["val", "test"] if config.FINAL_TEST else ["val"]
    metrics, lines = {"mean": mean, "sd": sd, "training": histories}, []
    val_windows = select(windows, masks["val"])
    _, val_table = score(models, val_windows, mean, sd, profile)
    reference = min(BASELINE_NAMES, key=lambda b: val_table[b]["all"]["mae"])
    metrics["reference_baseline"] = reference
    pred_cols = [f"pred_s{s}" for s in config.WAREHOUSE_TRAIN.seeds]
    ctrl_cols = [f"ctrl_s{s}" for s in config.WAREHOUSE_TRAIN.seeds]

    for split in splits:
        w = select(windows, masks[split])
        predictions, table = score(models, w, mean, sd, profile)
        metrics[split] = {"errors": table, "hypotheses": hypothesis_measures(predictions, table, w, reference,
                                                                            pred_cols, ctrl_cols),
                          "attention": attention_rows(models, pred_cols, w, mean, sd)}
        write_predictions(out / f"predictions_{split}.csv", w, predictions)
        lines += [comparison_table(table, reference, pred_cols, ctrl_cols, f"{split}: all hours (orders/h)"), "",
                  f"#### {split}: MAE by A-11 group", "",
                  group_table(table, list(BASELINE_NAMES) + pred_cols + ctrl_cols), ""]
        plot_week(out / f"forecast_week_{split}.png", w, predictions, split)
        plot_attention(out / f"attention_{split}.png", models["pred_s0"], w, predictions, mean, sd, split,
                       metrics[split]["attention"])

    header = [f"# Warehouse model vs baselines (FINAL_TEST = {config.FINAL_TEST})", "",
              f"Reference baseline, chosen on validation MAE among B1–B4: **{reference}** "
              f"(validation MAE: " + ", ".join(f"{b} {val_table[b]['all']['mae']:.2f}" for b in BASELINE_NAMES) + ").",
              f"Standardisation from training targets: μ = {mean:.3f}, s = {sd:.3f}.", ""]
    run.finish("warehouse", metrics, "\n".join(header + lines),
               {name: {"model": cfg, "train": config.WAREHOUSE_TRAIN, "data": config.WAREHOUSE} for name, cfg in ARMS.items()})


def hypothesis_measures(predictions: dict, table: dict, w: dict, reference: str, pred_cols, ctrl_cols) -> dict:
    """The quantities H5–H7 are scored on, for this split (H8 is in attention_rows)."""
    other = w["group"] == "other"
    mae = lambda col, mask: float(np.abs(predictions[col][mask] - w["target"][mask]).mean())
    day_mae = {b: [mae(b, other & (w["target_day"] == d)) for d in range(7)] for b in ("B1", "B3")}
    model_mae = np.array([table[c]["all"]["mae"] for c in pred_cols])
    ctrl_mae = np.array([table[c]["all"]["mae"] for c in ctrl_cols])
    improvement = {}
    for g in ("after", "other"):
        mask = w["group"] == g
        improvement[g] = [1 - mae(c, mask) / mae("B3", mask) for c in pred_cols]
    return {
        "H5": {"mae_all": {b: table[b]["all"]["mae"] for b in ("B1", "B2", "B3")},
               "B2_over_B3": table["B2"]["all"]["mae"] / table["B3"]["all"]["mae"],
               "B1_vs_B3_normal_hours_relative_gap": abs(table["B1"]["other"]["mae"] / table["B3"]["other"]["mae"] - 1),
               "B3_monday_over_tue_thu_normal": day_mae["B3"][0] / np.mean(day_mae["B3"][1:4]),
               "normal_hour_mae_by_target_day_mon_sun": day_mae},
        "H6": {"reference": reference, "reference_mae": table[reference]["all"]["mae"],
               "model_mae_per_seed": model_mae.tolist(),
               "model_improvement_over_reference_per_seed": (1 - model_mae / table[reference]["all"]["mae"]).tolist(),
               "control_mae_per_seed": ctrl_mae.tolist(),
               "control_over_model_per_seed": (ctrl_mae / model_mae).tolist()},
        "H7": {"improvement_over_B3_after_event": improvement["after"],
               "improvement_over_B3_normal": improvement["other"]},
    }


def write_predictions(path, w: dict, predictions: dict) -> None:
    cols = ["target_index", "target"] + list(predictions) + ["group", "event_sign", "hours_since_onset",
                                                             "target_hour", "target_day"]
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for i in range(len(w["target"])):
            vals = [str(w["target_index"][i]), f"{w['target'][i]:.0f}"]
            vals += [f"{predictions[c][i]:.3f}" for c in predictions]
            vals += [str(w["group"][i]), str(w["event_sign"][i]), str(w["hours_since_onset"][i]),
                     str(w["target_hour"][i]), str(w["target_day"][i])]
            f.write(",".join(vals) + "\n")


def plot_week(path, w: dict, predictions: dict, split: str) -> None:
    week = slice(0, 168)
    t = w["target_index"][week]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(t, w["target"][week], color="black", lw=1.2, label="actual y(t+1)")
    ax.plot(t, predictions["pred_s0"][week], color="C0", lw=1.2, label="attention model (seed 0)")
    for b, colour in zip(BASELINE_NAMES, ("C1", "C2", "C3", "C4")):
        ax.plot(t, predictions[b][week], color=colour, lw=0.8, alpha=0.7, label=b)
    ax.set(xlabel="target hour", ylabel="orders / hour", title=f"First week of the {split} split")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


@torch.no_grad()
def plot_attention(path, model, w: dict, predictions: dict, mean: float, sd: float, split: str, rows: dict) -> None:
    """Average readout row, plus a typical, an event and the worst-error window (FAC-33)."""
    _, att = model(model_inputs(w, mean, sd))
    A = att.A[:, -1, :].numpy()
    err = np.abs(predictions["pred_s0"] - w["target"])
    other = np.flatnonzero(w["group"] == "other")
    inside = np.flatnonzero(w["group"] == "inside")
    picks = {"typical": other[np.argsort(err[other])[len(other) // 2]], "worst error": int(np.argmax(err))}
    if len(inside):
        picks["event"] = inside[0]
    lags = np.arange(-23, 1)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(lags, rows["pred_s0"]["all"], color="black", lw=2, label="average over windows")
    for (label, i), colour in zip(picks.items(), ("C0", "C3", "C1")):
        ax.plot(lags, A[i], color=colour, lw=1, alpha=0.8, label=f"{label} (target {w['target_index'][i]})")
    ax.axhline(1 / 24, color="grey", ls=":", lw=0.8)
    ax.set(xlabel="input position (lag relative to hour t; −23 = same hour yesterday as the target)",
           ylabel="weight in the readout row", title=f"Attention weights, seed 0, {split} split (descriptive, not causal)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
