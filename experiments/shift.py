"""Demo part 6 (§30): generalisation under distribution shift (§19; A-20; PHASE0 §8 and Amendments 1–2; H9–H11).

Loads the trained warehouse models and their controls (no retraining), with the
training μ, s and B4 profile stored in the weight files, and the reference
baseline chosen on validation. Scores them and B1–B4 on four 52-week series
generated from the shift seed: control, higher noise, larger spikes, both.
This stage reads no test-split output, so it can run before FINAL_TEST.

Run: python -m experiments.shift
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wda import config, run
from wda.config import ModelConfig
from wda.metrics import attention_rows, comparison_table, group_table, score
from wda.models import WarehouseModel
from wda.warehouse_data import generate_series, make_windows

SEEDS = config.WAREHOUSE_TRAIN.seeds
PRED = [f"pred_s{s}" for s in SEEDS]
CTRL = [f"ctrl_s{s}" for s in SEEDS]
BASELINES = ("B1", "B2", "B3", "B4")


def load_models():
    models, extra = {}, None
    for name in PRED + CTRL:
        w = run.load_weights(run.RESULTS / "warehouse" / f"weights_{name}.pt")
        model = WarehouseModel(ModelConfig(**w["extra"]["model"]), seed=w["extra"]["seed"])
        model.load_state_dict(w["state_dict"])
        model.eval()
        models[name], extra = model, w["extra"]
    return models, extra["mean"], extra["sd"], np.asarray(extra["profile"])


def main():
    out = run.start("shift")
    reference = json.loads((run.RESULTS / "warehouse" / "metrics.json").read_text())["metrics"]["reference_baseline"]
    models, mean, sd, profile = load_models()
    per_series, tables = {}, {}
    for name, cfg in config.SHIFT_SERIES.items():
        series = generate_series(cfg)
        w = make_windows(series)
        predictions, table = score(models, w, mean, sd, profile)
        tables[name] = table
        onset3 = (w["group"] == "inside") & (w["event_sign"] == 1) & (w["hours_since_onset"] <= 2)
        normal = series["event_label"] == 0
        near_mean = normal & (np.abs(series["level"] - 97.1) < 7.5)
        per_series[name] = {
            "errors": table,
            "spike_onsets": int(len(np.unique(series["event_onset"][(series["event_label"] == 1)]))),
            "first_3h_after_spike_onset": {"count": int(onset3.sum()),
                                           **{c: float(np.abs(predictions[c][onset3] - w["target"][onset3]).mean())
                                              for c in PRED + list(BASELINES)}},
            "measured_noise_sd_near_mean_level": float((series["demand"][near_mean] - series["level"][near_mean]).std()),
            "attention": attention_rows(models, PRED, w, mean, sd),
        }
        write_predictions(out / f"predictions_{name}.csv", w, predictions)

    hyp = hypotheses(tables, per_series, reference)
    verdict = section19_verdict(tables, reference)
    metrics = {"reference_baseline": reference, "series": per_series, "hypotheses": hyp, "section19": verdict}
    plot(out / "mae_by_condition.png", tables, reference)

    lines = [f"# Distribution shift: four 52-week series, shift seed {config.SHIFT_SERIES['control'].seed} "
             f"(reference baseline {reference}, chosen on validation)", "",
             "| Series | Spike onsets | First-3-h spike targets | Measured noise SD near the mean level |",
             "|---|---|---|---|"]
    for name, p in per_series.items():
        lines.append(f"| {name} | {p['spike_onsets']} | {p['first_3h_after_spike_onset']['count']} | "
                     f"{p['measured_noise_sd_near_mean_level']:.1f} |")
    lines.append("")
    for name in config.SHIFT_SERIES:
        lines += [comparison_table(tables[name], reference, PRED, CTRL, f"{name}: all hours (orders/h)"), "",
                  f"#### {name}: MAE by A-11 group", "", group_table(tables[name], list(BASELINES) + PRED), ""]
    lines += ["## Change from the control series (MAE, orders/h)", "",
              "| Forecaster | " + " | ".join(n for n in config.SHIFT_SERIES if n != "control") + " |",
              "|---|---|---|---|"]
    for c in list(BASELINES) + PRED:
        base = tables["control"][c]["all"]["mae"]
        cells = [f"{tables[n][c]['all']['mae'] - base:+.2f} ({100 * (tables[n][c]['all']['mae'] / base - 1):+.0f}%)"
                 for n in config.SHIFT_SERIES if n != "control"]
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    lines += ["", "## §19 verdict (PHASE0 Amendment 1)", "",
              f"Per seed: beats {reference} on the control series; absolute MAE increase control → both vs "
              f"{reference}'s {verdict['reference_increase']:+.2f}.", ""]
    for s in verdict["per_seed"]:
        lines.append(f"- {s['seed']}: control MAE {s['control_mae']:.2f} (beats reference: {s['beats_reference']}), "
                     f"increase {s['increase']:+.2f} → {s['label']}")
    lines += ["", f"**Verdict: {verdict['verdict']}.** Original relative rule (reported for transparency): "
              f"{verdict['original_relative_rule']}."]
    run.finish("shift", metrics, "\n".join(lines),
               {name: cfg for name, cfg in config.SHIFT_SERIES.items()})


def hypotheses(tables, per_series, reference) -> dict:
    """The measures H9–H11 are scored on."""
    mae = lambda n, c, g="all": tables[n][c][g]["mae"]
    inc = lambda n, c: mae(n, c) - mae("control", c)
    return {
        "H9": {"normal_ratio_B3": mae("higher_noise", "B3", "other") / mae("control", "B3", "other"),
               "normal_ratio_B1": mae("higher_noise", "B1", "other") / mae("control", "B1", "other"),
               "model_increase_per_seed": [inc("higher_noise", c) for c in PRED],
               "B3_increase": inc("higher_noise", "B3"),
               "model_minus_B3_under_higher_noise_per_seed": [mae("higher_noise", c) - mae("higher_noise", "B3") for c in PRED]},
        "H10": {"normal_change_pct": {c: 100 * (mae("larger_spikes", c, "other") / mae("control", c, "other") - 1)
                                      for c in list(BASELINES) + PRED},
                "inside_factor": {c: mae("larger_spikes", c, "inside") / mae("control", c, "inside")
                                  for c in ["B1"] + PRED},
                "first_3h_larger_spikes": per_series["larger_spikes"]["first_3h_after_spike_onset"]},
        "H11": {"combined_increase_per_seed": [inc("both", c) for c in PRED],
                "sum_of_single_increases_per_seed": [inc("higher_noise", c) + inc("larger_spikes", c) for c in PRED]},
    }


def section19_verdict(tables, reference) -> dict:
    """Amendment 1: beat the reference on the control series AND rise by no more than it, in absolute MAE."""
    mae = lambda n, c: tables[n][c]["all"]["mae"]
    ref_inc = mae("both", reference) - mae("control", reference)
    ref_rel = mae("both", reference) / mae("control", reference) - 1
    per_seed, rel = [], []
    for c in PRED:
        increase = mae("both", c) - mae("control", c)
        beats = mae("control", c) < mae("control", reference)
        label = "learned useful structure" if beats and increase <= ref_inc else (
            "simply adapted" if increase > ref_inc else "inconclusive")
        per_seed.append({"seed": c, "control_mae": mae("control", c), "beats_reference": beats,
                         "increase": increase, "label": label})
        rel.append(mae("both", c) / mae("control", c) - 1 <= ref_rel)
    labels = {s["label"] for s in per_seed}
    if labels == {"learned useful structure"}:
        overall = "learned useful structure"
    elif all(s["increase"] > ref_inc for s in per_seed):
        overall = "simply adapted to the training distribution"
    else:
        overall = "inconclusive"
    original = ("learned useful structure" if all(rel) else
                "simply adapted to the training distribution" if not any(rel) else "inconclusive")
    return {"reference_increase": ref_inc, "per_seed": per_seed, "verdict": overall, "original_relative_rule": original}


def write_predictions(path, w: dict, predictions: dict) -> None:
    cols = ["target_index", "target"] + list(predictions) + ["group", "event_sign", "hours_since_onset",
                                                             "target_hour", "target_day"]
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for i in range(len(w["target"])):
            vals = [str(w["target_index"][i]), f"{w['target'][i]:.0f}"] + [f"{predictions[c][i]:.3f}" for c in predictions]
            vals += [str(w[k][i]) for k in ("group", "event_sign", "hours_since_onset", "target_hour", "target_day")]
            f.write(",".join(vals) + "\n")


def plot(path, tables, reference) -> None:
    names = list(config.SHIFT_SERIES)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for c, colour in zip(BASELINES, ("C1", "C2", "C3", "C4")):
        ax.plot(names, [tables[n][c]["all"]["mae"] for n in names], marker="o", color=colour, lw=1,
                label=c + (" (reference)" if c == reference else ""))
    model = np.array([[tables[n][c]["all"]["mae"] for c in PRED] for n in names])
    ax.errorbar(names, model.mean(axis=1), yerr=model.std(axis=1), marker="s", color="C0", lw=2, capsize=3,
                label="attention model (mean ± SD, 3 seeds)")
    ax.set(ylabel="MAE, all hours (orders/h)", title="Error under distribution shift")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
