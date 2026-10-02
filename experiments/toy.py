"""Demo part 3 (§30): the toy learning task, associative recall (§12, A-18, H1).

Scaled attention and the uniform-attention control, three seeds each, with the
same step budget. Weights kept: best validation accuracy. Held-out accuracy is
measured on the separate test sequences.

Run: python -m experiments.toy
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
from wda.train import fit

ARMS = {"attention": config.TOY_MODEL, "uniform_control": replace(config.TOY_MODEL, uniform=True)}


def main():
    out = run.start("toy")
    data = splits(config.TOY)
    diag = (data["val"][0][:config.DIAG_BATCH_SIZE], data["val"][1][:config.DIAG_BATCH_SIZE])
    results = {}
    for arm, model_cfg in ARMS.items():
        results[arm] = []
        for seed in config.TOY_TRAIN.seeds:
            model = ToyModel(model_cfg, seed=seed)
            r = fit(model, data["train"], data["val"], F.cross_entropy, accuracy, config.TOY_TRAIN, seed,
                    higher_is_better=True, diag=diag)
            model.load_state_dict(r["best_state"])
            with torch.no_grad():
                logits, _ = model(data["test"][0])
            results[arm].append({
                "seed": seed,
                "held_out_accuracy": accuracy(logits, data["test"][1]),
                "best_val_accuracy": r["best_metric"],
                "best_step": r["best_step"],
                "steps_to_95": steps_to(r["history"], "val_metric", config.TOY_TARGET_ACCURACY),
                "history": r["history"],
            })
            print(f"{arm} seed {seed}: held-out accuracy {results[arm][-1]['held_out_accuracy']:.4f}")

    chance, guess_present = 1 / config.TOY.n_values, 1 / config.TOY.n_pairs
    lines = [
        "# Toy task: associative recall (8 pairs, 16 keys, 16 values)",
        "",
        f"Chance = 1/{config.TOY.n_values} = {chance:.4f}; guessing among the {config.TOY.n_pairs} values present "
        f"= {guess_present:.4f}. Held-out = {config.TOY.n_test:,} unseen sequences.",
        "",
        "| Arm | Held-out accuracy, mean ± SD | Per seed | Steps to 95% validation accuracy | Best step |",
        "|---|---|---|---|---|",
    ]
    for arm, rs in results.items():
        acc = np.array([r["held_out_accuracy"] for r in rs])
        steps = [r["steps_to_95"] for r in rs]
        lines.append(f"| {arm} | {acc.mean():.4f} ± {acc.std():.4f} | {', '.join(f'{a:.4f}' for a in acc)} | "
                     f"{', '.join('not reached' if s is None else str(s) for s in steps)} | "
                     f"{', '.join(str(r['best_step']) for r in rs)} |")

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 3.6))
    for arm, colour in (("attention", "C0"), ("uniform_control", "C3")):
        for r in results[arm]:
            h = r["history"]
            steps = [x["step"] for x in h]
            label = arm if r["seed"] == 0 else None
            ax_loss.plot(steps[1:], [x["train_loss"] for x in h[1:]], color=colour, alpha=0.4, lw=0.8)
            ax_loss.plot(steps, [x["val_loss"] for x in h], color=colour, lw=1.3, label=label)
            ax_acc.plot(steps, [x["val_metric"] for x in h], color=colour, lw=1.3, label=label)
    ax_loss.set(xlabel="step", ylabel="cross-entropy", title="Loss (faint: training, solid: validation)")
    ax_acc.axhline(config.TOY_TARGET_ACCURACY, color="grey", ls="--", lw=0.8)
    ax_acc.axhline(guess_present, color="grey", ls=":", lw=0.8)
    ax_acc.set(xlabel="step", ylabel="validation accuracy", title="Accuracy (dashed 95%, dotted 1/8)", ylim=(0, 1.02))
    for ax in (ax_loss, ax_acc):
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=120)
    plt.close(fig)

    run.finish("toy", results, "\n".join(lines),
               {arm: {"model": cfg, "train": config.TOY_TRAIN, "data": config.TOY} for arm, cfg in ARMS.items()})


if __name__ == "__main__":
    main()
