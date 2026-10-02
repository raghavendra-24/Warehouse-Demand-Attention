"""Demo part 5 (§30): scaled vs unscaled attention, and the training-dynamics analysis (§13, §14; A-19; H2–H4).

Grid: {scaled, unscaled} × d_k ∈ {4, 64} × 3 seeds. Paired arms share the data,
the initial weights and the batch order; their configs differ only in `scaled`.
At step 0, before any update, the readout row's ‖∂L/∂q‖ is recorded for every
sequence of a fixed batch (H3). A diverging arm is recorded as a result, not a
crash, because divergence is a legitimate H4 outcome.

Run: python -m experiments.ablation
"""

from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from wda import config, run
from wda.metrics import accuracy, steps_to
from wda.models import ToyModel
from wda.toy_data import splits
from wda.train import diagnostics, fit


def arm_config(d_k: int, scaled: bool):
    return replace(config.TOY_MODEL, d_k=d_k, scaled=scaled)


def main():
    out = run.start("ablation")
    data = splits(config.TOY)
    diag = (data["val"][0][:config.DIAG_BATCH_SIZE], data["val"][1][:config.DIAG_BATCH_SIZE])
    runs, configs = {}, {}
    for d_k in config.ABLATION_DK:
        for scaled in (True, False):
            name = f"dk{d_k}_{'scaled' if scaled else 'unscaled'}"
            configs[name] = {"model": arm_config(d_k, scaled), "train": config.TOY_TRAIN}
            runs[name] = []
            for seed in config.TOY_TRAIN.seeds:
                model = ToyModel(arm_config(d_k, scaled), seed=seed)
                init = diagnostics(model, *diag, F.cross_entropy, full=True)
                record = {"seed": seed, "init": init, "diverged_at": None}
                try:
                    r = fit(model, data["train"], data["val"], F.cross_entropy, accuracy, config.TOY_TRAIN, seed,
                            higher_is_better=True, diag=diag)
                except FloatingPointError as e:
                    record["diverged_at"] = str(e)
                    runs[name].append(record)
                    print(f"{name} seed {seed}: DIVERGED ({e})")
                    continue
                model.load_state_dict(r["best_state"])
                with torch.no_grad():
                    logits, _ = model(data["test"][0])
                record.update({
                    "held_out_accuracy": accuracy(logits, data["test"][1]),
                    "steps_to_95": steps_to(r["history"], "val_metric", config.TOY_TARGET_ACCURACY),
                    "history": r["history"],
                })
                runs[name].append(record)
                print(f"{name} seed {seed}: steps to 95% {record['steps_to_95']}, "
                      f"held-out {record['held_out_accuracy']:.4f}, init entropy {init['entropy_all']:.3f}")

    summary = summarise(runs)
    lines = table(summary)
    plot(runs, out)
    for name in runs:                      # full gradient vectors are summarised, not stored per row
        for r in runs[name]:
            r["init"] = {k: v for k, v in r["init"].items() if not isinstance(v, np.ndarray)}
    run.finish("ablation", {"summary": summary, "runs": runs}, "\n".join(lines), configs)


def summarise(runs: dict) -> dict:
    """H2 (initial entropy), H3 (initial readout query gradients) and H4 (steps to 95%) per d_k."""
    summary = {}
    for d_k in config.ABLATION_DK:
        s, u = runs[f"dk{d_k}_scaled"], runs[f"dk{d_k}_unscaled"]
        h3 = []
        for rs, ru in zip(s, u):                             # paired by seed
            gs, gu = rs["init"]["grad_q_readout"], ru["init"]["grad_q_readout"]
            median_s = np.median(gs)
            h3.append({
                "seed": rs["seed"],
                "unscaled_rows_below_1pct_of_scaled_median": float((gu < 0.01 * median_s).mean()),
                "scaled_rows_below_1pct_of_scaled_median": float((gs < 0.01 * median_s).mean()),
                "p90_p10_scaled": float(np.percentile(gs, 90) / np.percentile(gs, 10)),
                "p90_p10_unscaled": float(np.percentile(gu, 90) / max(np.percentile(gu, 10), 1e-300)),
                "median_ratio_unscaled_over_scaled": float(np.median(gu) / median_s),
            })
        steps_s = [r.get("steps_to_95") for r in s]
        steps_u = [r.get("steps_to_95") for r in u]
        reached = [(a, b) for a, b in zip(steps_s, steps_u) if a is not None and b is not None]
        summary[f"dk{d_k}"] = {
            "entropy_all_scaled": [r["init"]["entropy_all"] for r in s],
            "entropy_all_unscaled": [r["init"]["entropy_all"] for r in u],
            "logit_sd_scaled": [r["init"]["logit_sd"] for r in s],
            "logit_sd_unscaled": [r["init"]["logit_sd"] for r in u],
            "max_weight_scaled": [r["init"]["max_weight_readout"] for r in s],
            "max_weight_unscaled": [r["init"]["max_weight_readout"] for r in u],
            "h3": h3,
            "steps_to_95_scaled": steps_s,
            "steps_to_95_unscaled": steps_u,
            "median_step_ratio_unscaled_over_scaled": float(np.median([b / a for a, b in reached])) if reached else None,
            "unscaled_seeds_not_reaching_95": sum(x is None for x in steps_u),
            "diverged": {"scaled": [r["diverged_at"] for r in s], "unscaled": [r["diverged_at"] for r in u]},
            "held_out_scaled": [r.get("held_out_accuracy") for r in s],
            "held_out_unscaled": [r.get("held_out_accuracy") for r in u],
        }
    return summary


def table(summary: dict) -> list:
    fmt = lambda xs: ", ".join("—" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x)) for x in xs)
    lines = ["# Ablation: scaled vs unscaled attention (toy task, 3 paired seeds)", "",
             "## At initialisation (step 0, before any update)", "",
             "| d_k | Arm | Row entropy ÷ ln n (H2) | Logit SD | Readout max weight |", "|---|---|---|---|---|"]
    for d_k in config.ABLATION_DK:
        s = summary[f"dk{d_k}"]
        for arm in ("scaled", "unscaled"):
            lines.append(f"| {d_k} | {arm} | {fmt(s[f'entropy_all_{arm}'])} | {fmt(s[f'logit_sd_{arm}'])} | "
                         f"{fmt(s[f'max_weight_{arm}'])} |")
    lines += ["", "## Readout-row query gradient at step 0 (H3), per seed", "",
              "| d_k | Unscaled rows < 1% of scaled median | Scaled rows < 1% | p90/p10 scaled | p90/p10 unscaled | "
              "Median unscaled ÷ scaled |", "|---|---|---|---|---|---|"]
    for d_k in config.ABLATION_DK:
        h3 = summary[f"dk{d_k}"]["h3"]
        col = lambda k, f: ", ".join(f.format(x[k]) for x in h3)
        lines.append(f"| {d_k} | {col('unscaled_rows_below_1pct_of_scaled_median', '{:.1%}')} | "
                     f"{col('scaled_rows_below_1pct_of_scaled_median', '{:.1%}')} | {col('p90_p10_scaled', '{:.1f}')} | "
                     f"{col('p90_p10_unscaled', '{:.3g}')} | {col('median_ratio_unscaled_over_scaled', '{:.2f}')} |")
    lines += ["", "## Training (H4)", "",
              "| d_k | Steps to 95%: scaled | Steps to 95%: unscaled | Median ratio unscaled ÷ scaled | "
              "Unscaled seeds not reaching 95% | Held-out: scaled | Held-out: unscaled |", "|---|---|---|---|---|---|---|"]
    for d_k in config.ABLATION_DK:
        s = summary[f"dk{d_k}"]
        ratio = s["median_step_ratio_unscaled_over_scaled"]
        lines.append(f"| {d_k} | {fmt(s['steps_to_95_scaled'])} | {fmt(s['steps_to_95_unscaled'])} | "
                     f"{'—' if ratio is None else f'{ratio:.2f}'} | {s['unscaled_seeds_not_reaching_95']} | "
                     f"{fmt(s['held_out_scaled'])} | {fmt(s['held_out_unscaled'])} |")
    diverged = [(k, arm, d) for k, s in summary.items() for arm, ds in s["diverged"].items() for d in ds if d]
    lines += ["", f"Diverged runs: {len(diverged)}" + "".join(f"\n- {k} {arm}: {d}" for k, arm, d in diverged)]
    return lines


def plot(runs: dict, out):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    colours = {"dk4_scaled": "C0", "dk4_unscaled": "C1", "dk64_scaled": "C2", "dk64_unscaled": "C3"}
    for name, rs in runs.items():
        for r in rs:
            if "history" not in r:
                continue
            h = r["history"]
            steps = [x["step"] for x in h]
            label = name if r["seed"] == 0 else None
            axes[0].plot(steps, [x["val_loss"] for x in h], color=colours[name], lw=1, label=label)
            axes[1].plot(steps, [x["entropy_readout"] for x in h], color=colours[name], lw=1, label=label)
            axes[2].semilogy(steps, [max(x["grad_W_Q"], 1e-12) for x in h], color=colours[name], lw=1, label=label)
    axes[0].set(xlabel="step", ylabel="validation cross-entropy", title="Loss")
    axes[1].set(xlabel="step", ylabel="readout entropy ÷ ln n", title="Attention entropy (readout row)")
    axes[2].set(xlabel="step", ylabel="‖∂L/∂W_Q‖", title="Query-weight gradient norm")
    for ax in axes:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
