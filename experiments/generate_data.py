"""Generate the warehouse series and check it against the PHASE0 design targets (§4, §6, FAC-10).

Rerunning this script is the reset/regeneration mechanism (§4): the data is a
pure function of wda/config.py, so every other script regenerates it in memory.

Run: python -m experiments.generate_data
"""

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wda import config, run
from wda.warehouse_data import generate_series


def autocorrelation(x: np.ndarray, lag: int) -> float:
    x = x - x.mean()
    return float((x[:-lag] * x[lag:]).sum() / (x * x).sum())


def main():
    out = run.start("data")
    cfg = config.WAREHOUSE
    s = generate_series(cfg)
    y, label, onset = s["demand"].astype(float), s["event_label"], s["event_onset"]
    base_level = np.maximum(cfg.floor, cfg.base + s["daily"] + s["weekly"])

    mean_level = cfg.base + sum(cfg.weekday_offsets) / 7
    noise_sd = math.sqrt(mean_level + mean_level ** 2 / cfg.dispersion)
    amplitude = float((s["daily"]).max())
    onsets = np.unique(onset[onset >= 0])
    spike_onsets = [o for o in onsets if label[o] == 1]
    drop_onsets = [o for o in onsets if label[o] == -1]
    free_hours = int((label == 0).sum())
    spike_m = s["level"][label == 1] / base_level[label == 1]
    drop_m = s["level"][label == -1] / base_level[label == -1]
    normal = label == 0
    near_mean = normal & (np.abs(s["level"] - mean_level) < 7.5)

    stats = {
        "mean_demand": float(y.mean()),
        "autocorrelation": {lag: autocorrelation(y, lag) for lag in (1, 24, 168)},
        "noise_sd_at_mean_level_formula": noise_sd,
        "noise_sd_at_mean_level_measured": float((y[near_mean] - s["level"][near_mean]).std()),
        "daily_amplitude": amplitude,
        "amplitude_over_noise_sd": amplitude / noise_sd,
        "spike_onsets": len(spike_onsets),
        "drop_onsets": len(drop_onsets),
        "spike_onset_rate_per_free_hour": len(spike_onsets) / free_hours,
        "drop_onset_rate_per_free_hour": len(drop_onsets) / free_hours,
        "share_of_hours_in_events": float((label != 0).mean()),
        "spike_multiplier_min_median_max": [float(spike_m.min()), float(np.median(spike_m)), float(spike_m.max())],
        "drop_multiplier_min_median_max": [float(drop_m.min()), float(np.median(drop_m)), float(drop_m.max())],
    }
    targets = {
        "daily amplitude ≥ 5 × noise SD at the mean level": stats["amplitude_over_noise_sd"] >= 5,
        "events affect ≤ ~5% of hours": stats["share_of_hours_in_events"] <= 0.05,
        "spikes 2–4 × the expected level": 2 <= spike_m.min() and spike_m.max() <= 4,
    }
    stats["design_targets_met"] = targets

    ac = stats["autocorrelation"]
    lines = [
        f"# Warehouse series (seed {cfg.seed}, {cfg.n_hours:,} hours)",
        "",
        "| Statistic | Value | PHASE0 expectation |",
        "|---|---|---|",
        f"| Mean demand | {stats['mean_demand']:.1f} orders/h | 100.2 |",
        f"| Autocorrelation, lags 1 / 24 / 168 | {ac[1]:.2f} / {ac[24]:.2f} / {ac[168]:.2f} | 0.84 / 0.63 / 0.66 |",
        f"| Noise SD at the mean level (formula / measured) | {noise_sd:.1f} / {stats['noise_sd_at_mean_level_measured']:.1f} orders/h | 13.8 |",
        f"| Daily amplitude ÷ noise SD | {stats['amplitude_over_noise_sd']:.2f} | ≥ 5 (5.42) |",
        f"| Spike / drop onsets | {len(spike_onsets)} / {len(drop_onsets)} | ≈ 34 / 17 |",
        f"| Onset rate per event-free hour, spike / drop | {stats['spike_onset_rate_per_free_hour']:.4f} / {stats['drop_onset_rate_per_free_hour']:.4f} | 0.004 / 0.002 |",
        f"| Share of hours inside an event | {stats['share_of_hours_in_events']:.1%} | 2.3% (≤ 5%) |",
        f"| Spike multiplier min / median / max | {' / '.join(f'{v:.2f}' for v in stats['spike_multiplier_min_median_max'])} | within 2–4 |",
        f"| Drop multiplier min / median / max | {' / '.join(f'{v:.2f}' for v in stats['drop_multiplier_min_median_max'])} | within 0.3–0.6 |",
        "",
        "Design targets: " + "; ".join(f"{k}: {'met' if v else 'NOT MET'}" for k, v in targets.items()),
    ]

    week = slice(0, 168)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(s["t"][week], y[week], lw=0.9, label="demand y(t)")
    ax.plot(s["t"][week], s["level"][week], lw=1.4, label="expected level λ(t)")
    ax.set(xlabel="hour (week 1, Monday 00:00 = 0)", ylabel="orders / hour", title="One week of demand")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "week.png", dpi=120)
    plt.close(fig)

    first = spike_onsets[0]
    span = slice(max(0, first - 24), first + 48)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(s["t"][span], y[span], lw=0.9, label="demand y(t)")
    ax.plot(s["t"][span], s["level"][span], lw=1.4, label="expected level λ(t)")
    ax.axvspan(first, first + int((onset == first).sum()), color="orange", alpha=0.2, label="spike")
    ax.set(xlabel="hour", ylabel="orders / hour", title=f"First spike (onset hour {first}) and the following 48 hours")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "event.png", dpi=120)
    plt.close(fig)

    run.finish("data", stats, "\n".join(lines), {"data": cfg})


if __name__ == "__main__":
    main()
