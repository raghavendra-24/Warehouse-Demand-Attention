# Demo instructions

Reproducible live-demo instructions (PRD §30). Parts 1 and 2 run live in seconds. The longer runs (parts 3–7) are shown from their committed results, and each one can be rerun with the command given. Every command runs from the repository root, after the setup in the [README](../README.md).

| Part | What it shows | Live command | Runtime | Where to look |
|---|---|---|---|---|
| 1 · Attention | The seven §7 operations on a tiny example worked by hand | `python -m experiments.attention_trace` | < 1 s | console; [results/trace/table.md](../results/trace/table.md); [wda/attention.py](../wda/attention.py) |
| 2 · Gradient verification | Autograd against float64 central differences, every entry | `python -m experiments.gradcheck` | < 1 s | [results/gradcheck/table.md](../results/gradcheck/table.md) |
| 3 · Training | The toy task learning, against the uniform-attention control | (rerun: `python -m experiments.toy`) | ~25–90 s | [results/toy/curves.png](../results/toy/curves.png), [results/toy/table.md](../results/toy/table.md) |
| 4 · Warehouse prediction | The attention model against the baselines B1–B4, on the test split | (rerun: `python -m experiments.warehouse`) | ~3 min | [results/warehouse/table.md](../results/warehouse/table.md), `forecast_week_test.png`, `attention_test.png` |
| 5 · Ablation | Scaled vs unscaled attention: prediction, then observation | (rerun: `python -m experiments.ablation`) | ~4–4.5 min | [results/ablation/table.md](../results/ablation/table.md), `curves.png` |
| 6 · Generalisation | Four 52-week shifted series, and the §19 verdict | `python -m experiments.shift` | ~5 s | [results/shift/table.md](../results/shift/table.md), `mae_by_condition.png` |
| 7 · Failure | Forecasts saturating during large spikes, and why | `python -m experiments.failure` | ~2 s | [results/failure/table.md](../results/failure/table.md), `dose_response.png` |

Parts 6 and 7 are fast enough to run live as well, because they load the saved warehouse weights.

## Talking points, in order

1. **Attention.**
   - Walk through [wda/attention.py](../wda/attention.py) line by line. Each §7 operation is a named statement, and the softmax is hand-written and max-subtracted.
   - Show the trace and point out that $S$ and $A$ are asymmetric. That is why the hand-computed test catches both $KQ^\top$ and $A^\top V$ bugs ([docs/DEBUGGING.md](DEBUGGING.md), episode 1).
2. **Gradients.**
   - Every one of the 45 entries agrees under A-17's rule: $\lvert g_{\text{auto}} - g_{\text{num}} \rvert \le 10^{-8} + 10^{-6} \cdot \lvert g_{\text{num}} \rvert$.
   - Explain $h = 10^{-6}$ in float64: truncation error $\propto h^2$, round-off error $\propto \varepsilon / h$.
   - Explain why the loss is $\sum R \odot Y$ and not $\operatorname{sum}(A)$: each row of $A$ sums to 1, so the gradient of $\operatorname{sum}(A)$ is zero.
3. **Training.**
   - Attention reaches 0.9999 held-out accuracy, while the uniform control stays at 1/8. The control can see which values are present, but not which one belongs to the queried key.
   - Point out the plateau, then the sudden drop in loss.
4. **Warehouse.**
   - Validation picked B4, the hour-of-week mean, as the reference, as Amendment 3 predicted.
   - On test, the attention model's MAE is 2.9% lower on average but not in every seed, so H6 is refuted. Its RMSE is 18% lower, because it handles spikes better.
   - The attention weights are descriptive, not causal.
5. **Ablation.**
   - Start from the Phase 0 predictions. `grep -E '^\| (ID|H2|H3|H4) \|' docs/RESULTS.md | cut -d'|' -f2-5` prints each prediction next to what was measured and the verdict.
   - At $d_k = 64$, unscaled attention starts almost one-hot (entropy 0.15 of $\ln n$).
   - Its query gradients become extremely uneven: 11–15% of rows get almost nothing, while the p90/p10 spread exceeds 1,000.
   - It needs 2× the steps to reach 95%, and one seed never learns. At $d_k = 4$ there is no difference.
6. **Generalisation.**
   - The noise and spike shifts behave as H9–H11 predicted.
   - The §19 verdict is inconclusive: the model's MAE increase matches B4's.
   - Explain why the original relative rule was replaced before any shift result existed (Amendment 1).
7. **Failure.**
   - The dose-response table shows the forecast flattening while demand keeps rising.
   - The value path carries almost no demand magnitude, so the convex-combination readout can only re-weight tokens.
   - Mention the two refuted explanations, and the proposed fix: a direct path for demand magnitude.

## Recording

A terminal recording of all seven parts, in the order above, is linked in the submission form (about 4.5 minutes, no voice-over). It was made from a fresh clone of tag `v1.0-submission`, with the environment installed from `requirements.txt`. The code and results are the same in `v1.1-submission`, which changes only documents.
- every command is typed and run live, including the full toy, warehouse and ablation training runs;
- the long training waits are shortened and marked "time-lapse" in the title bar;
- the committed figures are shown after the stage that produces them;
- the shell prompt is shown as `raghavendra@laptop:~/Warehouse-Demand-Attention`: the real username, hostname and clone path are replaced in the display;
- after the ablation run, the Phase 0 predictions for H2–H4 are shown next to the measured values.

The rerun in the recording reproduced every committed `metrics.json` bit for bit; only the `run.json` files (runtime, git revision) changed.
