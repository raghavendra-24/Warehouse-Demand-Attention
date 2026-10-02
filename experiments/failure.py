"""Demo part 7 (§30): failure investigation (§20; A-21; PHASE0 F1–F5).

1. Error slices for the pre-registered failure modes F1–F5, on the 52-week
   shift series (no test-split data is needed).
2. The investigated case: during large spikes the attention model's forecast
   saturates far below demand. With g_j = w_out · V_j (token j's value projected
   on the head), the forecast is ẑ = Σ_j A_j g_j + b, a convex combination.
   The targeted experiment scales each spike in the same windows (k = 0 … 16)
   and records the forecast, the attention mass on the spike tokens, the spike
   tokens' own value level, and the all-token ceiling max_j g_j + b.

Run: python -m experiments.failure
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from wda import config, run
from wda.metrics import score
from wda.models import load_warehouse_models
from wda.warehouse_data import generate_series, make_windows, model_inputs, select

SEEDS = config.WAREHOUSE_TRAIN.seeds
PRED = [f"pred_s{s}" for s in SEEDS]
SCALES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0, 16.0)
windows_of = lambda a: np.lib.stride_tricks.sliding_window_view(a, config.WINDOW)[:-1]


def main():
    out = run.start("failure")
    models, mean, sd, profile = load_warehouse_models(run.RESULTS / "warehouse", PRED)
    data = {}
    for name in ("control", "larger_spikes"):
        s = generate_series(config.SHIFT_SERIES[name])
        w = make_windows(s)
        data[name] = (s, w, *score(models, w, mean, sd, profile))
    slices = failure_mode_slices(data, models, mean, sd)
    case = saturation_case(data, models, mean, sd)
    plot_dose_response(out / "dose_response.png", case["dose_response"])
    lines = case_table(case) + [""] + slices_table(slices)
    (out / "failure_modes.md").write_text("\n".join(slices_table(slices)) + "\n")
    run.finish("failure", {"slices": slices, "case": case}, "\n".join(lines),
               {"series": {k: config.SHIFT_SERIES[k] for k in data}, "scales": SCALES})


def mae(pred, y, mask):
    return float(np.abs(pred[mask] - y[mask]).mean()) if mask.any() else None


def failure_mode_slices(data, models, mean, sd) -> dict:
    """MAE of the model (seed mean) and B1–B4 on the slices each failure mode names."""
    s, w, pred, _ = data["control"]
    y, g, h, day = w["target"], w["group"], w["hours_since_onset"], w["target_day"]
    model = np.mean([pred[c] for c in PRED], axis=0)
    cols = {"model": model, **{b: pred[b] for b in ("B1", "B2", "B3", "B4")}}
    spike = (g == "inside") & (w["event_sign"] == 1)
    slices = {
        "F1 spike onset hour (control)": spike & (h == 0),
        "F1 spike hours 1-2 (control)": spike & (h >= 1) & (h <= 2),
        "F2 echo: 24-29 h after a spike starts (control)": (g == "after") & (w["event_sign"] == 1) & (h >= 24) & (h <= 29),
        "F3 Saturday/Monday 00-11 h, normal (control)": (g == "other") & np.isin(day, (0, 5)) & (w["target_hour"] < 12),
        "F3 reference: Tue-Thu 00-11 h, normal (control)": (g == "other") & np.isin(day, (1, 2, 3)) & (w["target_hour"] < 12),
        "F5 day type differs between t+1 and t-23 (Sat, Mon), normal": (g == "other") & np.isin(day, (0, 5)),
        "F5 reference: same day type (Tue-Fri, Sun), normal": (g == "other") & ~np.isin(day, (0, 5)),
    }
    out = {name: {"count": int(mask.sum()), **{c: mae(v, y, mask) for c, v in cols.items()}}
           for name, mask in slices.items()}
    _, wl, predl, _ = data["larger_spikes"]
    model_l = np.mean([predl[c] for c in PRED], axis=0)
    inside_c, inside_l = (g == "inside") & (w["event_sign"] == 1), (wl["group"] == "inside") & (wl["event_sign"] == 1)
    out["F4 in-spike MAE, larger spikes / control"] = {
        "count": int(inside_l.sum()),
        "model": mae(model_l, wl["target"], inside_l) / mae(model, y, inside_c),
        "B1": mae(predl["B1"], wl["target"], inside_l) / mae(pred["B1"], y, inside_c)}
    return out


@torch.no_grad()
def saturation_case(data, models, mean, sd) -> dict:
    s, w, pred, _ = data["larger_spikes"]
    after_onset = (w["group"] == "inside") & (w["event_sign"] == 1) & (w["hours_since_onset"] >= 1)
    y, b1 = w["target"], w["raw_window"][:, -1]
    m0 = models["pred_s0"]
    err = np.abs(pred["pred_s0"] - y)
    worst = int(np.flatnonzero(after_onset)[np.argmax(err[after_onset])])
    case = {
        "scenario": {"series": "larger_spikes (shift seed 202)", "windows": int(after_onset.sum()),
                     "worst_window_target_index": int(w["target_index"][worst]),
                     "worst_window_target": float(y[worst]), "worst_window_forecast_s0": float(pred["pred_s0"][worst]),
                     "worst_window_B1": float(b1[worst])},
        "actual": {"mean_target": float(y[after_onset].mean()), "mean_B1": float(b1[after_onset].mean()),
                   **{c: float(pred[c][after_onset].mean()) for c in PRED},
                   "slope_of_forecast_on_y_t_s0": float(np.polyfit(b1[after_onset], pred["pred_s0"][after_onset], 1)[0]),
                   "slope_of_target_on_y_t": float(np.polyfit(b1[after_onset], y[after_onset], 1)[0])},
        "value_gain_on_demand": {c: value_gain(models[c]) for c in PRED},
        "ceiling": {},
    }
    for name in ("control", "larger_spikes"):
        sn, wn, pn, _ = data[name]
        sel = (wn["group"] == "inside") & (wn["event_sign"] == 1) & (wn["hours_since_onset"] >= 1)
        cap = ceiling(m0, select(wn, sel), mean, sd)
        case["ceiling"][name] = {"mean_forecast_s0": float(pn["pred_s0"][sel].mean()), "mean_ceiling_s0": float(cap.mean()),
                                 "share_within_5pct_of_ceiling": float((pn["pred_s0"][sel] >= 0.95 * cap).mean()),
                                 "mean_target": float(wn["target"][sel].mean())}
    case["dose_response"] = dose_response(data["control"], models, mean, sd)
    return case


def value_gain(model) -> float:
    """∂g/∂z: how much a token's projected value changes per standardised unit of demand (exact, linear)."""
    direction = model.inp.weight[:, 0]                       # d X / d z
    return float(model.head.weight[0] @ (model.attn.W_V.T @ direction))


def ceiling(model, w: dict, mean: float, sd: float) -> np.ndarray:
    """Largest forecast the convex combination allows in each window: μ + s · (max_j g_j + b)."""
    _, att = model(model_inputs(w, mean, sd))
    g = att.V @ model.head.weight[0]
    return (mean + sd * (g.max(dim=-1).values + model.head.bias)).numpy()


def dose_response(control, models, mean, sd) -> dict:
    """Scale each spike's excess over its no-event level by k in the same control windows."""
    s, w, _, _ = control
    sel = (w["group"] == "inside") & (w["event_sign"] == 1) & (w["hours_since_onset"] >= 1)
    ws = select(w, sel)
    base = windows_of(np.maximum(1.0, config.WAREHOUSE.base + s["daily"] + s["weekly"]))[sel]
    spike = windows_of(s["event_label"])[sel] == 1
    base_target = np.maximum(1.0, config.WAREHOUSE.base + s["daily"] + s["weekly"])[ws["target_index"]]
    rows = []
    for k in SCALES:
        raw = np.where(spike, base + k * (ws["raw_window"] - base), ws["raw_window"])
        wk = dict(ws, raw_window=raw)
        X = model_inputs(wk, mean, sd)
        row = {"k": k, "expected_target": float((base_target + k * (ws["target"] - base_target)).mean()),
               "B1": float(raw[:, -1].mean())}
        spike_t = torch.from_numpy(spike)
        for c in PRED:
            with torch.no_grad():
                z, att = models[c](X)
                g = att.V @ models[c].head.weight[0]
            row[c] = float((mean + sd * z).mean())
            row[f"spike_attention_mass_{c}"] = float((att.A[:, -1, :] * spike_t).sum(1).mean())
            row[f"spike_value_level_{c}"] = float(mean + sd * (((g * spike_t).sum(1) / spike_t.sum(1)).mean()
                                                              + models[c].head.bias.item()))
        row["ceiling_s0"] = float(ceiling(models["pred_s0"], wk, mean, sd).mean())
        rows.append(row)
    return {"windows": int(sel.sum()), "rows": rows}


def case_table(case) -> list:
    sc, ac, vg, ce = case["scenario"], case["actual"], case["value_gain_on_demand"], case["ceiling"]
    lines = ["# Failure case: the forecast saturates during large spikes", "",
             f"**Scenario.** {sc['series']}, the {sc['windows']} targets from the second hour of a spike onward. "
             f"Worst window: target index {sc['worst_window_target_index']} (demand {sc['worst_window_target']:.0f}, "
             f"seed-0 forecast {sc['worst_window_forecast_s0']:.0f}, B1 {sc['worst_window_B1']:.0f}).", "",
             f"**Actual.** Mean demand {ac['mean_target']:.0f}; B1 {ac['mean_B1']:.0f}; model "
             + ", ".join(f"{ac[c]:.0f}" for c in PRED) + f". The forecast follows y(t) with slope "
             f"{ac['slope_of_forecast_on_y_t_s0']:.2f} (demand: {ac['slope_of_target_on_y_t']:.2f}).", "",
             "**Evidence.**", "",
             "| Measure | Value |", "|---|---|"]
    lines += [f"| Value gain on demand ∂g/∂z, {c} | {v:.3f} |" for c, v in vg.items()]
    for name, c in ce.items():
        lines.append(f"| {name}: mean forecast / mean ceiling / mean demand (seed 0) | "
                     f"{c['mean_forecast_s0']:.0f} / {c['mean_ceiling_s0']:.0f} / {c['mean_target']:.0f} |")
        lines.append(f"| {name}: forecasts within 5% of the ceiling | {c['share_within_5pct_of_ceiling']:.0%} |")
    lines += ["", f"**Dose-response** (control series, {case['dose_response']['windows']} spike windows; "
              "each spike's excess over its no-event level scaled by k):", "",
              "| k | Expected demand | B1 | Model (seeds 0, 1, 2) | Seed 0: attention mass on spike tokens | "
              "Seed 0: spike tokens' value level | Seed 1: mass / value level | All-token ceiling (seed 0) |",
              "|---|---|---|---|---|---|---|---|"]
    for r in case["dose_response"]["rows"]:
        lines.append(f"| {r['k']:g} | {r['expected_target']:.0f} | {r['B1']:.0f} | "
                     + ", ".join(f"{r[c]:.0f}" for c in PRED)
                     + f" | {r['spike_attention_mass_pred_s0']:.3f} | {r['spike_value_level_pred_s0']:.0f} | "
                     f"{r['spike_attention_mass_pred_s1']:.3f} / {r['spike_value_level_pred_s1']:.0f} | {r['ceiling_s0']:.0f} |")
    lines += ["", "**Explanation.** The learned value path carries almost no demand magnitude: a token's projected "
              "value g changes by only 0.02–0.06 per standardised unit of demand, so values are set mostly by the "
              "calendar features. Because the readout is a convex combination with no residual path (A-06), the "
              "forecast can rise only by moving attention between tokens, a bounded route that saturates. Seed 0 moves "
              "attention onto the spike tokens (mass → 1 as k grows) and its forecast converges to their value level; "
              "seed 1 moves attention elsewhere. Both plateau far below demand, while B1 tracks it.",
              "",
              "**Refuted explanations (kept for the record).** (1) Large spikes push attention away from the spike "
              "tokens: refuted, the mass on them rises for seed 0. (2) The all-token convex-combination ceiling binds: "
              "refuted, forecasts stay far below it (no forecast within 5%).",
              "",
              "**Potential improvement.** Give demand magnitude a direct path to the output (a residual connection, or "
              "y(t) as an input to the head), so attention chooses where to look and the value carries how much; or "
              "train with heavier-tailed spikes. Not implemented (§20: a perfect fix is not required)."]
    return lines


def slices_table(slices) -> list:
    cols = ["model", "B1", "B2", "B3", "B4"]
    lines = ["## Failure-mode slices (MAE, orders/h; model = mean of 3 seeds)", "",
             "| Slice | Count | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for name, row in slices.items():
        if name.startswith("F4"):
            continue
        lines.append(f"| {name} | {row['count']} | " + " | ".join("—" if row[c] is None else f"{row[c]:.1f}" for c in cols) + " |")
    f4 = slices["F4 in-spike MAE, larger spikes / control"]
    lines += ["", f"F4: in-spike MAE grows ×{f4['model']:.2f} for the model and ×{f4['B1']:.2f} for B1 "
              f"when spikes are doubled ({f4['count']} in-spike targets)."]
    return lines


def plot_dose_response(path, dr) -> None:
    ks = [r["k"] for r in dr["rows"]]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(ks, [r["expected_target"] for r in dr["rows"]], color="black", marker="o", label="expected demand")
    ax.plot(ks, [r["B1"] for r in dr["rows"]], color="C1", marker="o", label="B1 (last observation)")
    for c, colour in zip(PRED, ("C0", "C9", "C2")):
        ax.plot(ks, [r[c] for r in dr["rows"]], color=colour, marker="s", label=f"attention model {c}")
    ax.plot(ks, [r["spike_value_level_pred_s0"] for r in dr["rows"]], color="C0", ls="--",
            label="seed 0: spike tokens' value level")
    ax.axvline(1.0, color="grey", ls=":", lw=0.8)
    ax.set(xlabel="spike scale k (1 = training-size spikes)", ylabel="orders / hour (mean over windows)",
           title="Forecast during spikes as the spike grows")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
