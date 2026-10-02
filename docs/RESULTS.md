# Results

This document gives every experiment's result and an outcome for every pre-registered ID in [PHASE0.md](PHASE0.md): H1–H11, V1–V2, F1–F5, the §19 verdict and the A-27 verdict. Amendments 1–3 are applied where they change the frozen text. Every measured number is copied from a committed file under `results/` or from the code, and its source is cited in brackets; expected values are quoted from PHASE0.md. A number marked *computed* is simple arithmetic (a ratio, difference, mean or SD) on values in the cited file.

**Conventions**
- Errors are in orders per hour.
- The model is reported as mean ± population SD over seeds 0, 1 and 2, followed by the per-seed values.
- Difference = model − baseline, so a negative value means the model is better.
- An effect is claimed only with the same sign in all three seeds (A-16).
- *Demonstrated* means a committed result shows it. *Believed* means no experiment here isolated it.

## 1. Summary scorecard

| ID | Prediction (short) | Measured | Verdict | Source |
|---|---|---|---|---|
| H1 | Attention ≥ 95% held-out; uniform control ≤ 1.5/8 | 0.9998, 1.0000, 1.0000; control 0.1196, 0.1250, 0.1302 | **confirmed** | results/toy/table.md |
| H2 | Step-0 entropy ÷ ln n: scaled ≥ 0.75; unscaled 0.5–0.75 (d_k = 4), ≤ 0.25 (d_k = 64) | Scaled 0.821–0.888; unscaled 0.592–0.704 and 0.145–0.155 | **confirmed** | results/ablation/table.md |
| H3 | d_k = 64 unscaled: ≥ 5% of rows < 1% of the scaled median (scaled none); p90/p10 ≥ 100× (scaled ≤ 10×); median ≥ 0.5× scaled | 12.1%, 14.8%, 11.3% (scaled 0.0%); 1.27e+03–4.25e+03 (scaled 2.9–3.7); 1.13, 0.74, 1.58 | **confirmed** | results/ablation/table.md |
| H4 | d_k = 64: unscaled ≥ 1.5× the steps to 95%, or a seed fails; d_k = 4: ratio in [0.67, 1.5] | 2.07, and 1 seed never reached 95%; 0.95 | **confirmed** | results/ablation/table.md |
| V1 | All entries within 1e-8 + 1e-6·\|g_num\|; relative differences around 1e-9 | 45/45 agree; max rel. difference 2.62e-08; unexplained entries 0 | **confirmed** | results/gradcheck/table.md |
| V2 | Naive float32 softmax inf/NaN above ≈ 88.7; stable always finite; naive finite ⇒ agrees within 1e-6 | NaN from 89; stable finite throughout; at 88.5 naive is finite but [0, 0, 0] | **partly refuted** (finite ≠ correct) | results/stability/table.md |
| H5 | Test B1 < B3 < B2; B2/B3 in [1.4, 2.0]; normal-hour B1–B3 gap < 10%; B3 Monday ≥ 1.5× Tue–Thu | 20.59 < 24.91 < 43.41; 1.74; 5.3%; 1.98 | **confirmed** | results/warehouse/metrics.json |
| H6 | Model 5–20% below the reference in every seed; control ≥ 1.5× model | vs B4: 5.6%, 3.5%, −0.5%; control ÷ model 2.80–2.96 | **refuted** (size and sign; control clause holds; see note) | results/warehouse/metrics.json |
| H7 | vs B3: post-event improvement ≥ 2× normal-hour; normal-hour < 10% | Post-event 54.5–56.7%; normal 34.7–36.5%; ratio 1.49–1.63 | **prediction wrong in both numbers**; the refutation clause as written is not met | results/warehouse/metrics.json |
| H8 | Weight on t−23 and t ≥ 50% | 9.1%, 6.4%, 6.2% (uniform 8.3%) | **refuted** | results/warehouse/metrics.json |
| H9 | Higher noise: normal-hour ×1.7–2.1 (B3), ×1.4–2.0 (B1); model's absolute rise < B3's; model stays ahead of B3 | ×1.71, ×1.75; +11.22, +11.58, +11.43 vs +13.31; ahead by 11.95–12.57 | **confirmed** | results/shift/metrics.json |
| H10 | Larger spikes: normal hours change < 10%; model's in-event factor > B1's; B1 better in the first 3 spike hours | ≤ 1.8%; 2.74, 2.79, 2.49 vs 2.00; 229.7 vs 323.8–343.2 | **confirmed** | results/shift/metrics.json |
| H11 | Combined rise within ±25% of the sum of the single rises | 14.76 vs 14.83; 15.10 vs 15.32; 15.10 vs 15.19 | **confirmed** | results/shift/metrics.json |
| §19 | Amendment 1: beat B4 on control, and an absolute rise to "both" ≤ B4's, in every seed | All beat B4; rises +14.76, +15.10, +15.10 vs +14.87 | **inconclusive** (original rule: simply adapted) | results/shift/table.md |
| A-27 | Adds value if Δ_ref < 0 and Δ_ctl < 0 in every seed | Δ_ref −0.86, −0.53, +0.07; Δ_ctl < 0 in all | **inconclusive** | results/warehouse/metrics.json |
| F1 | Every forecaster misses a spike's first hour; the model may lag after it | Onset-hour MAE 205.2–211.5 for all; hours 1–2: model 73.4, B1 59.4 | **confirmed** | results/failure/table.md |
| F2 | B3 (and any t−23 copier) echoes a spike 24 h later | B3 197.4; model 11.7 | **confirmed for B3**; the model does not echo | results/failure/table.md |
| F3 | Model error on Saturday/Monday mornings | Model 10.2 vs 9.8; B3 24.1 vs 13.4 | **weak for the model** | results/failure/table.md |
| F4 | Larger spikes: weights shift; error grows faster than B1's | ×2.86 vs ×2.06; the cause is value-path saturation | **confirmed** (symptom; mechanism differs) | results/failure/table.md |
| F5 | Larger error when t+1 and t−23 differ in day type | 11.9 vs 11.8 | **not observed** | results/failure/table.md |

**H7: prediction wrong in both numbers; the refutation clause as written is not met.** The post-event gain over B3 is only 1.49–1.63 times the normal-hour gain (*computed*), not ≥ 2×, and the normal-hour gain is 34.7–36.5%, not below 10%. The stated "Refuted if" clause (normal-hour improvement ≥ post-event improvement) is not triggered, because the post-event gain is still the larger one in every seed. H7 is therefore neither confirmed nor refuted by its own clause; its two numerical predictions are false.

**H6:** the 5–20% size and the same-sign condition both fail. The "refuted" verdict reads the clause "no better than the best baseline in all three seeds" as "not better in every seed", which seed 2 meets. Read as "no better in any seed", the clause would not be met, because seeds 0 and 1 beat B4.

**What was wrong.** Four predictions failed in full or in part: V2, H6, H7 (both numbers, though not by its own refutation clause) and H8. Amendment 3 had expected H6 to fail once B4 existed. H7 and H8 were wrong about *how* the model works: it neither concentrates on t−23 and t nor gains mainly after events. Two verdicts are inconclusive, and the model shows F2, F3 and F5 weakly or not at all. H1–H5, V1 and H9–H11 held.

## 2. Data

The training series uses seed 101: 8,736 hours, from a Monday.

| Statistic | Value | PHASE0 expectation |
|---|---|---|
| Mean demand | 100.4 orders/h | 100.2 |
| Autocorrelation, lags 1 / 24 / 168 | 0.84 / 0.66 / 0.68 | 0.84 / 0.63 / 0.66 |
| Noise SD at the mean level (formula / measured) | 13.8 / 13.0 orders/h | 13.8 |
| Daily amplitude ÷ noise SD | 5.42 | ≥ 5 (5.42) |
| Spike / drop onsets | 43 / 13 | ≈ 34 / 17 |
| Share of hours inside an event | 2.5% | 2.3% (≤ 5%) |
| Spike multiplier min / median / max | 2.01 / 2.75 / 3.87 | within 2–4 |
| Drop multiplier min / median / max | 0.30 / 0.41 / 0.60 | within 0.3–0.6 |

[results/data/table.md]

All three design targets are met. The autocorrelation gaps (at most 0.03) are below the 0.1 that PHASE0 §3 set as the trigger for a generator-bug investigation. Two gaps were not investigated: the measured noise SD is 6% below the formula, and there are 43 spike onsets against ≈ 34 expected. For a memoryless count with mean 34 the Poisson SD is about 5.8, so that gap is plausible (*computed*).

The splits are whole weeks: 1–36 / 37–44 / 45–52, giving 6,024 / 1,344 / 1,344 windows (*computed* from `wda/config.py`). The training targets give μ = 99.911 and s = 54.124. A-11 group counts (inside an event / within 24 h after one / other):
- test: 33 / 189 / 1,122, against PHASE0's ≈ 31 / 176 / 1,137;
- validation: 34 / 223 / 1,087.

[results/warehouse/table.md]

## 3. Gradient verification (V1) and numerical stability (V2)

### 3.1 Gradient check (FAC-20)
**Setup.**
- What is compared: autograd's gradient of L = Σ R ⊙ Y, for a fixed random R, against float64 central differences with h = 1e-6, for every entry of X, W_Q, W_K and W_V.
- Sizes: n = 5, d_model = 3, d_k = 4, d_v = 2. They all differ, so a transposed projection fails. Seed 7.
- Loss: Σ A cannot be used, because its gradient is zero (rows sum to 1).
- Agreement test: an entry agrees if |g_auto − g_num| ≤ 1e-8 + 1e-6·|g_num| (A-17).

| Tensor | Entries | Max abs. difference | Max rel. difference | Disagreeing |
|---|---|---|---|---|
| X | 15 | 8.05e-10 | 3.69e-09 | 0 |
| W_Q | 12 | 8.98e-10 | 1.47e-08 | 0 |
| W_K | 12 | 7.57e-10 | 2.62e-08 | 0 |
| W_V | 6 | 3.37e-10 | 4.10e-09 | 0 |

The entry with the largest relative difference in each tensor (all 45 rows are in the source):

| Tensor | Index | Autograd | Numerical | Abs. diff | Rel. diff |
|---|---|---|---|---|---|
| X | (3, 2) | +0.1599364608 | +0.1599364602 | 5.91e-10 | 3.69e-09 |
| W_Q | (1, 2) | +0.0396528129 | +0.0396528135 | 5.82e-10 | 1.47e-08 |
| W_K | (0, 2) | +0.0200799948 | +0.0200799953 | 5.27e-10 | 2.62e-08 |
| W_V | (2, 1) | +0.0296859515 | +0.0296859516 | 1.22e-10 | 4.10e-09 |

**Unexplained entries: 0.** No relative difference exceeds 1e-6, so no entry needed an individual explanation. [results/gradcheck/table.md]

**Why it is not exact.** Autograd is exact to float64 rounding, so the differences measure the error of the finite-difference estimate. That error has two parts:
- truncation, h²|L‴|/6, about 1e-13 for third derivatives of order one;
- round-off, ε|L|/h with ε ≈ 1.1e-16, about 1e-10 for |L| of order one.

The measured absolute differences, 3.82e-12 to 8.98e-10, are at the round-off level. The two largest relative differences sit at the smallest and third-smallest gradients in the check (0.0201 and 0.0397), where the same absolute error is a larger fraction. The worst, 2.62e-08, is 38 times below the tolerance's 1e-6 relative term, and its absolute difference is 57 times below that entry's full tolerance, 1e-8 + 1e-6·|g_num| ≈ 3.0e-08 (*computed*), and 39 of the 45 lie between 1e-10 and 5e-9 (*computed*). **V1 is confirmed.**

**The forward pass is checked separately**, because a forward bug such as KQᵀ would pass the gradient check (both gradients use the same forward code). The hand-checkable trace (n = 2, d_model = 3, d_k = 4, d_v = 1) gives S = [[4, 1], [4, 5]] (asymmetric, so a transpose shows), S′ = [[2.0, 0.5], [2.0, 2.5]], A = [[0.817574, 0.182426], [0.377541, 0.622459]] and Y = [[1.547277], [2.867378]]. By hand, A₁₁ = 1/(1 + e^(−1.5)) and Y₁ = 0.817574·1 + 0.182426·4. [results/trace/table.md]

### 3.2 Numerical stability

| Logits | Naive float32 | Stable | Naive finite | Naive correct (≤ 1e-6) |
|---|---|---|---|---|
| [88, 87, 0] | [0.7311, 0.2689, 4.426e-39] | [0.7311, 0.2689, 4.426e-39] | yes | yes |
| [88.5, 87.5, 0] | [0, 0, 0] | [0.7311, 0.2689, 2.685e-39] | yes | **no** (max difference 7.3e-01) |
| [89, 88, 0] | [nan, 0, 0] | [0.7311, 0.2689, 1.628e-39] | no | no |
| [1000, 999, 0] | [nan, nan, 0] | [0.7311, 0.2689, 0] | no | no |

The source table has eight inputs (80 to 1000); the stable output sums to 1.0000000 in all of them. [results/stability/table.md]

**V2 is partly refuted.** The predicted overflow above ln(3.4e38) ≈ 88.72 happens, and the max-subtracted form stays finite. But "wherever the naive version is finite, the two agree" is false.
- At [88.5, 87.5, 0], e^88.5 and e^87.5 are each representable, but their sum overflows to inf, so every weight becomes finite/inf = 0.
- The prediction checked for overflow in a single term and missed overflow in the denominator.
- The stable form avoids both: softmax(s − c) = softmax(s), and with c = max s every term lies in (0, 1] and the denominator in [1, n].

## 4. Toy task: associative recall (H1)

**4.1 Loss curves.** Training and validation loss, and validation accuracy, are plotted in `results/toy/curves.png`. Over 3,000 steps, validation cross-entropy falls:
- attention: 2.810, 2.818, 2.802 → 0.0010, 0.0009, 0.0009;
- uniform control: 2.796, 2.801, 2.788 → 2.264, 2.265, 2.231.

[results/toy/metrics.json]

**4.2 Training behaviour.** Attention reaches 95% validation accuracy at steps 600, 550 and 500 (best steps 700, 650, 600); the control never does. The curve is a short plateau followed by sudden learning. For seed 0:

| Step | 0 | 100 | 200 | 300 | 400 | 500 | 600 |
|---|---|---|---|---|---|---|---|
| Validation accuracy | 0.0710 | 0.1025 | 0.2340 | 0.4645 | 0.7535 | 0.9125 | 0.9795 |
| Readout entropy ÷ ln n | 0.824 | 0.846 | 0.845 | 0.802 | 0.696 | 0.571 | 0.481 |
| Readout max weight | 0.337 | 0.321 | 0.325 | 0.374 | 0.504 | 0.634 | 0.714 |

Seeds 1 and 2 have the same shape: 0.1545 and 0.1655 at step 200, then 0.9980 and 1.0000 at step 600. [results/toy/table.md, metrics.json]

**4.3 Held-out metric against trivial references.** On 5,000 unseen sequences:
- attention: 0.9999 ± 0.0001 (0.9998, 1.0000, 1.0000);
- uniform control: 0.1249 ± 0.0043 (0.1196, 0.1250, 0.1302);
- chance: 1/16 = 0.0625;
- guessing among the 8 values present: 0.1250.

[results/toy/table.md]

**4.4 Configuration.**
- Tokens: 9 × 33, each a one-hot key, a one-hot value and a query flag; no input projection; 8 pairs drawn from 16 keys and 16 values.
- Attention: d_k = d_v = 16, with W_Q, W_K, W_V ~ N(0, 1/2).
- Head: linear, 16 logits; 1,856 parameters.
- Training: Adam (β 0.9, 0.999; ε 1e-8), learning rate 1e-3, batch 256, 3,000 steps, evaluation every 50 steps.
- Seeds and data: training seeds 0, 1, 2; data 50,000 / 2,000 / 5,000 from seeds 301 / 302 / 303. The weights kept are those with the best validation accuracy.
- Runtime: 92.7 s on a 4-thread laptop CPU.

[results/toy/metrics.json, run.json]

**4.5 Observations.**
- **H1 is confirmed** with a wide margin.
- The control behaves as PHASE0 §7 predicted. A uniform average shows *which* values are present but not which one is bound to the query, so accuracy sits at 1/8, not 1/16. Its loss keeps falling toward ln 8 ≈ 2.08, the loss of a uniform guess among 8. Its ‖∂L/∂W_Q‖ and ‖∂L/∂W_K‖ are exactly 0.0 throughout, so the switch removes the score path.
- The control's best *validation* accuracy (0.1405, 0.1325, 0.1340) exceeds its held-out accuracy. It is the maximum of 61 noisy evaluations, which is why the held-out figure is reported.
- During the jump (seed 0), the median readout-row query-gradient norm rises from 6.11e-04 (step 0) to 2.36e-03 (step 400). It then decays to 3.46e-06 at step 3,000, as the weights concentrate: max weight 0.883, entropy 0.250.
- *Believed, not isolated:* the plateau comes from coupling. The score gradient is useful only once the head maps retrieved values to classes, and the head can learn only once attention retrieves the right value.

[results/toy/metrics.json]

## 5. Training dynamics: softmax saturation in unscaled attention (§13)

The chain below follows §13's form, for d_k = 64 and seeds 0, 1 and 2 unless stated. Each link cites quantities logged in the ablation runs at step 0 and every evaluation (`results/ablation/metrics.json`); `results/ablation/curves.png` plots validation loss, readout entropy and ‖∂L/∂W_Q‖ against step.

**Link 1. Large logits.** At step 0 the logit SD is 7.735, 8.241 and 8.237 unscaled, against 0.967, 1.030 and 1.030 scaled.

*Mechanism.* The initialisation gives q and k independent unit-variance components. Then Var(q·k) = Σ_i Var(q_i k_i) = d_k, so the SD is √64 = 8, and dividing by √d_k restores 1. The same argument at d_k = 4 predicts 2, and the measured values are 1.949, 2.206 and 1.615. [results/ablation/table.md]

**Link 2. Concentrated softmax.** At step 0:
- the mean row entropy ÷ ln n is 0.151, 0.145 and 0.155 unscaled, against 0.841, 0.827 and 0.832 scaled;
- the readout row's maximum weight is 0.868, 0.869 and 0.856, against 0.309, 0.356 and 0.327.

With 9 tokens and logits of SD 8, the gap between a row's two largest logits is usually several units, and e^(gap) makes the row nearly one-hot. [results/ablation/table.md]

**Link 3. Some gradients become very small; not all, and not on average.** Let a = softmax(s). The Jacobian ∂a/∂s = diag(a) − aaᵀ tends to 0 as a tends to one-hot. With g_ij = ∂L/∂a_ij,

∂L/∂q_i = Σ_j a_ij (g_ij − Σ_m a_im g_im) k_j,

times 1/√d_k in the scaled arm. A saturated row therefore passes almost no gradient, while an unsaturated row passes √d_k = 8 times more than in the scaled arm. On the 256 readout rows:

| Step 0, d_k = 64 | Unscaled | Scaled |
|---|---|---|
| Rows with ‖∂L/∂q₉‖ < 1% of the scaled median | 12.1%, 14.8%, 11.3% | 0.0%, 0.0%, 0.0% |
| p90 / p10 of ‖∂L/∂q₉‖ | 1.27e+03, 4.25e+03, 3.29e+03 | 3.0, 3.7, 2.9 |
| Median, unscaled ÷ scaled | 1.13, 0.74, 1.58 | — |
| ‖∂L/∂W_Q‖ | 0.0900, 0.0825, 0.0947 | 0.0160, 0.0204, 0.0220 |

[results/ablation/table.md, metrics.json]

The total W_Q gradient is 4–6 times *larger* without scaling (*computed*). Saturation makes the gradient uneven, not small: about one readout row in eight is frozen, and the rest are amplified.

**Link 4. Changed learning.** Steps to 95%:
- scaled: 350, 350, 400;
- unscaled: 750, 700, and never (held-out accuracy 1.000, 1.000, 0.102).

The seed that never learns, unscaled d_k = 64 seed 2:

| Step | 0 | 500 | 1,000 | 1,500 | 2,000 | 3,000 |
|---|---|---|---|---|---|---|
| Validation accuracy | 0.0595 | 0.0610 | 0.0695 | 0.0725 | 0.0795 | 0.0930 |
| Readout entropy ÷ ln n | 0.167 | 0.191 | 0.181 | 0.175 | 0.246 | 0.357 |
| Readout max weight | 0.856 | 0.834 | 0.846 | 0.851 | 0.768 | 0.621 |
| Logit SD | 8.237 | 8.235 | 8.308 | 8.373 | 8.288 | 8.043 |

[results/ablation/metrics.json]

For 1,500 steps it stays saturated and at chance (1/16). That is *below* the uniform control's 1/8: a saturated row retrieves one essentially arbitrary value, while a uniform row at least reveals the set of values present. After step 1,500 the rows begin to de-saturate, but the budget runs out. *Believed, not tested:* more steps would let it learn.

The two seeds that do learn end in extreme saturation. At step 3,000 they have:
- readout entropy 0.008 and 0.001, against about 0.10 scaled;
- logit SD 8.881 and 9.610, against about 2.17 scaled;
- median readout query gradient 1.52e-07 and 1.33e-09, against 1.10e-06–1.20e-06 scaled.

Saturation also switches learning off once a solution is found. That is harmless when the solution is right, and a trap when it is not. [results/ablation/metrics.json]

**Control condition.** At d_k = 4, unscaled rows start only moderately concentrated (entropy 0.59–0.70), and the step ratio is 0.95: only strong saturation hurts. H4's caveat was that Adam's per-parameter rescaling might hide the effect. Adam can undo a uniform change of gradient scale, but not frozen rows, and the effect survived it. Plain SGD was not tried.

## 6. Ablation: scaled vs unscaled attention (§14)

**6.1 Hypothesis (PHASE0, H4, quoted).**
> At **d_k = 64**, unscaled needs **≥ 1.5×** as many steps as scaled to reach 95% validation accuracy (median over seeds), or fails to reach it within the step budget in at least one seed. At **d_k = 4**, the ratio is between 0.67 and 1.5. In every cell where both variants reach 95%, their final accuracies differ by less than 5 percentage points.

H2 and H3, the mechanism, are scored in Section 1.

**6.2 Experimental setup.**
- Task and arms: the Section 4 task; d_k ∈ {4, 64} × {scaled, unscaled}; d_v = 16; W ~ N(0, 1/2) in every arm; 1,064 parameters at d_k = 4 and 5,024 at d_k = 64.
- Pairing: three seeds. The weight draws do not depend on the scaling switch, so paired arms start from identical weights.
- Training: Adam, learning rate 1e-3, batch 256, 3,000 steps; no clipping, schedule or weight decay.
- Diagnostics: step-0 values on the first 256 validation sequences; held-out accuracy on 5,000.
- Runtime: 268.0 s; diverged runs: 0.

[results/ablation/metrics.json, run.json, table.md]

**6.3 Observed behaviour, per arm.**
- *d_k = 4 scaled:* starts diffuse (entropy 0.82–0.89) and learns slowly (900–1,050 steps, median 950).
- *d_k = 4 unscaled:* starts moderately concentrated, with no visible penalty (850–1,100, median 900).
- *d_k = 64 scaled:* starts diffuse and is the fastest arm (350–400).
- *d_k = 64 unscaled:* starts saturated. Two seeds learn at half the speed, and one never learns (Section 5).

**6.4 Quantitative results** (mean ± SD, *computed* from the per-seed values in results/ablation/metrics.json)

| d_k | Arm | Final val. loss (step 3,000) | Held-out accuracy | Steps to 95% | Step-0 entropy ÷ ln n | Step-0 ‖∂L/∂W_Q‖ | Step-0 ‖∂L/∂W_K‖ |
|---|---|---|---|---|---|---|---|
| 4 | scaled | 0.0017 ± 0.0002 | 0.9998 ± 0.0003 | 900, 1050, 950 | 0.857 ± 0.027 | 0.0132 ± 0.0035 | 0.0147 ± 0.0031 |
| 4 | unscaled | 0.0013 ± 0.0001 | 0.9991 ± 0.0007 | 850, 1100, 900 | 0.651 ± 0.046 | 0.0273 ± 0.0075 | 0.0299 ± 0.0050 |
| 64 | scaled | 0.0009 ± 0.0001 | 0.9998 ± 0.0003 | 350, 350, 400 | 0.833 ± 0.006 | 0.0195 ± 0.0025 | 0.0205 ± 0.0031 |
| 64 | unscaled | 0.89 ± 1.25 (0.0015, 0.0015, 2.65) | 0.70 ± 0.42 (1.000, 1.000, 0.102) | 750, 700, — | 0.151 ± 0.004 | 0.0891 ± 0.0050 | 0.0945 ± 0.0040 |

**6.5 Comparison with the hypothesis.** **H4 is supported**, and so are H2 and H3.
- d_k = 64: the median step ratio is 2.07, and one seed failed, so both conditions hold. The code takes the median over the seeds that reached 95% (725 ÷ 350). Counting the failed seed as "over 3,000" gives 2.14 (*computed*), so the verdict does not hang on that choice.
- d_k = 4: the ratio is 0.95.
- Where both arms reached 95%, held-out accuracies differ by at most 0.12 points (*computed*).

One failed seed out of three meets the pre-registered rule, but it does not estimate how often unscaled d_k = 64 fails.

*Not predicted:* scaled d_k = 64 learned faster than scaled d_k = 4 (350–400 against 900–1,050 steps) and faster than d_k = 16 in Section 4 (500–600). *Believed:* the 16 one-hot keys project to nearly orthogonal directions in 64 dimensions, which makes them easier to separate.

**6.6 Mathematical explanation.** For any W_Q,

softmax( (XW_Q)(XW_K)ᵀ ) = softmax( (X·√d_k W_Q)(XW_K)ᵀ / √d_k ).

Unscaled attention with W_Q is therefore exactly scaled attention with √d_k·W_Q. Both arms represent the same functions, and `test_unscaled_equals_scaled_with_rescaled_query_weights` checks this identity. The differences are in optimisation:
1. **Starting point.** The unscaled arm is the scaled arm started from √d_k times the query weights (×2 at d_k = 4, ×8 at d_k = 64). So the logit SD is √d_k, and the entropy falls to 0.651 and 0.151 (means), against PHASE0's simulated 0.64 and 0.15.
2. **Step size.** Write W̃ = √d_k·W_Q. Then ∂L/∂W_Q = √d_k·∂L/∂W̃. Adam's steps are roughly scale-free, so a step δ on W_Q is a step √d_k·δ on W̃. At d_k = 64 this is an 8× larger effective learning rate on the query weights.
3. **Local geometry.** From that start, diag(a) − aaᵀ is near zero in saturated rows (Section 5). A row saturated on the wrong key gets almost no correction.

At d_k = 4 the factor of 2 leaves rows moderately concentrated, so the arms train alike.

## 7. Warehouse model vs baselines (H5–H8, A-27)

**Setup.**
- Inputs: 24 tokens of 5 features (standardised demand, sin/cos hour of day, sin/cos day of week), projected to 16 dimensions.
- Attention: the shared scaled attention, read out at hour t by a linear head; 881 parameters.
- Training: 4,000 steps and 3 seeds; the weights kept are those with the best validation MAE. The uniform control fixes A = 1/24.
- Baselines: B1 y(t), B2 the 24-hour mean, B3 y(t−23), and B4 the training mean at the same hour of the week (Amendment 3).

**Reference, chosen on validation.** Validation MAE: B1 20.74, B2 44.55, B3 26.74, B4 15.57. **B4** is the reference, as Amendment 3 expected. [results/warehouse/table.md]

**Test scored once.**
1. Validation results were produced with `FINAL_TEST` off (commit `7ab95b8`).
2. Commit `b242a79` froze the configuration.
3. Commit `abf1a8e` changes one line of `wda/config.py`, turning `FINAL_TEST` on, and adds the test results. `run.json` records `final_test: true` on `b242a79` plus that uncommitted switch.
4. Later runs of `run_all` with `FINAL_TEST` on, such as the pre-submission check in the README, recompute the same test numbers deterministically from the frozen configuration and seeds. That is reproduction of this one evaluation, which PHASE0 §6 allows, not a second evaluation.

[results/warehouse/run.json; git history]

### 7.1 The §17 table (test, 1,344 hours)

Every forecaster, baselines included, is scored by the same function (`score` in `wda/metrics.py`) on the same 1,344 test targets, in orders/h after ŷ = μ + s ẑ.

| Metric | Baseline (B4) | Attention model | Difference (model − baseline) | Uniform control |
|---|---|---|---|---|
| MAE | 15.42 | 14.97 ± 0.39 (14.55, 14.88, 15.49) | −0.44 (−2.9%) | 43.21 ± 0.08 |
| RMSE | 30.65 | 25.23 ± 0.75 (24.42, 25.03, 26.22) | −5.43 (−17.7%) | 52.82 ± 0.14 |

Validation: MAE 15.57 against 15.37 ± 0.18 (−0.20, −1.3%); RMSE 30.83 against 25.31 ± 0.70 (−5.52, −17.9%). [results/warehouse/table.md]

### 7.2 Test MAE by A-11 group

| Group | Count | B1 | B2 | B3 | B4 | pred_s0 | pred_s1 | pred_s2 | ctrl_s0 | ctrl_s1 | ctrl_s2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 1344 | 20.59 | 43.41 | 24.91 | 15.42 | 14.55 | 14.88 | 15.49 | 43.14 | 43.16 | 43.32 |
| inside | 33 | 67.82 | 134.47 | 142.48 | 151.99 | 77.95 | 91.66 | 108.86 | 139.17 | 139.40 | 143.22 |
| after | 189 | 25.92 | 47.16 | 37.61 | 12.28 | 17.10 | 16.63 | 16.30 | 43.25 | 43.36 | 42.27 |
| other | 1122 | 18.30 | 40.10 | 19.32 | 11.93 | 12.26 | 12.33 | 12.61 | 40.29 | 40.30 | 40.56 |

[results/warehouse/table.md]

### 7.3 Baseline ranking against Phase 0 (H5)

| Test MAE | B1 | B2 | B3 | B4 |
|---|---|---|---|---|
| Expected, normal hours (PHASE0 §7) | 18.4 | 40.5 | 19.1 | — |
| Measured, normal ("other") hours | 18.30 | 40.10 | 19.32 | 11.93 |
| Expected, all hours | 20.3 | 44.1 | 25.6 | ≈ 15.8 (Amendment 3) |
| Measured, all hours | 20.59 | 43.41 | 24.91 | 15.42 |

[results/warehouse/table.md]

Every closed-form expectation is within 0.7 orders/h of the measurement (*computed*). **H5 holds on test:** B2/B3 = 1.74, the normal-hour B1–B3 gap is 5.3%, and B3's Monday ratio is 1.98. On validation the same figures are 1.67, 5.7% and 1.94. Both effects the first H5 draft had left out are visible:
- the Monday step: B3's normal-hour MAE is 32.20 on Monday against 14.59–17.97 Tuesday–Friday;
- the spike echo (Section 9.7).

[results/warehouse/metrics.json]

### 7.4 Where the model gains and loses against B4

The model's MAE minus B4's, split into each group's contribution, Δ_group × count ÷ 1,344 (*computed* from 7.2):

| Seed | inside (33) | after (189) | other (1,122) | Total |
|---|---|---|---|---|
| 0 | −1.82 | +0.68 | +0.28 | −0.86 |
| 1 | −1.48 | +0.61 | +0.34 | −0.53 |
| 2 | −1.06 | +0.57 | +0.57 | +0.07 |

- **The whole advantage comes from the 33 in-event targets.** There the model partly follows the level (77.95–108.86, where B4, which ignores events, scores 151.99), though less well than B1 (67.82). The RMSE advantage has the same source.
- **In the other 1,311 hours, B4 wins in every seed.** After events the model over-forecasts (signed error +5.03, +4.37, +4.24; B4 +0.53), because spike hours still in the window lift its forecast.
- **PHASE0 §6's predicted signed error of about +0.7 is not supported.** It is −2.15, −0.50 and −2.18 overall (pulled down inside events) and −1.42, +1.16, −0.32 in normal hours: mixed signs.

[results/warehouse/metrics.json]

### 7.5 H6, H7 and H8
- **H6, refuted** (on the reading of its clause given in Section 1). The improvement over B4 is 5.6%, 3.5% and −0.5%: 2.9% on average, below the predicted 5–20%, and not the same sign in all seeds. Amendment 3 had stated before any result that this was likely, since B4 averages 36 training values per hour of the week while the window holds one. The control clause holds (control ÷ model 2.80–2.96).
- **H7: prediction wrong in both numbers; the refutation clause as written is not met.** Relative to B3, the post-event improvement is 54.5%, 55.8% and 56.7%, and the normal-hour improvement 36.5%, 36.2% and 34.7%. The post-event gain is 1.49–1.63 times the normal-hour gain, not ≥ 2×, and the normal-hour gain is far above 10%. The "Refuted if" clause (normal-hour improvement ≥ post-event improvement) is not triggered, since the post-event gain stays larger in every seed. The model was expected to behave like a corrected B3, gaining mainly where B3 echoes spikes. Instead it behaves like B4, which beats B3 in normal hours by itself (11.93 against 19.32).
- **H8, refuted.** Readout weight on t−23 and t: 9.1%, 6.4% and 6.2%, against ≥ 50% predicted.

[results/warehouse/metrics.json]

### 7.6 A-27 verdict, the confound and the attention figure

**A-27: inconclusive.** Paired by seed on test, Δ_ref = −0.86, −0.53, +0.07 and Δ_ctl = −28.58, −28.28, −27.83 (*computed*). The model beats the control in every seed but B4 in only two of three. *Believed, outside the verdict:* on the 52-week control series of Section 8 (8,736 targets), all three seeds beat B4 (14.08, 14.02, 14.49 against 14.63). That hints at a small real advantage that the 8-week test cannot resolve. [results/warehouse/metrics.json, results/shift/table.md]

**The calendar-feature confound.** The model sees hour and day features that B1–B3 lack (A-05). B4 uses only the calendar and nearly ties the model. The uniform control does not isolate "attention to history" either: a window spans all 24 hours, so uniform averaging cancels the hour features, and the control loses the target's hour along with the weighting. Beating it (14.97 against 43.21) shows that non-uniform weighting is *necessary to use the calendar*, not that it extracts information from the demand history.

**The attention figure is descriptive, not causal** (§16). In `results/warehouse/attention_test.png` (seed 0, test), the average readout row is nearly flat, from 0.0316 to 0.0616 per position, with 0.049 on t−23 and 0.041 on t. Single plotted windows concentrate on a band of neighbouring hours (peaks of about 0.16–0.18) whose position varies between windows. A weight says where the readout looks. Section 9 shows that the values carry little demand information, so a high weight on an hour does not mean its demand drives the forecast. [results/warehouse/metrics.json]

*Training note:* the best validation steps were 3,350, 4,000 and 2,700 of 4,000. Seed 1 may have been cut short by the budget.

## 8. Generalisation under distribution shift (§19; H9–H11)

**8.1 Original distribution.** PHASE0 §2:
- negative-binomial counts with r = 100 (noise SD 13.8 at the mean level);
- spikes with onset probability 0.004/h, lasting 3–6 h, with m ~ U(2, 4).

Series 1 (control) uses these parameters with shift seed 202; its measured noise SD is 13.0.

**8.2 Modified distribution.**
- Series 2: r = 14, measured noise SD 25.7.
- Series 3: spike multipliers × 2 on the same draws, i.e. U(4, 8).
- Series 4: both changes.

Each series is 52 weeks after a 24-hour warm-up, giving 8,736 targets (Amendment 2). All four share one event timeline: 27 spike onsets and 78 first-three-hour spike targets. The models are evaluated without retraining, using the training μ and s. [results/shift/table.md]

**8.3 Expected impact (PHASE0, quoted).**
- H9: "Normal-hour MAE of B3 rises by **1.7–2.1×**, and of B1 by **1.4–2.0×**. The attention model's MAE rises by less in absolute terms than B3's, and it stays better than B3."
- H10: "Event-hour MAE rises for every model, and rises by a larger factor for the attention model than for B1."
- H11: "within ±25% of the sum of its increases in H9 and H10."
- PHASE0 §8 also gave expected baseline MAEs, shown in brackets below.

**8.4 Observed impact** (all-hours MAE; PHASE0's expected value in brackets)

| Forecaster | 1 Control | 2 Higher noise | 3 Larger spikes | 4 Both | Change, 1 → 4 |
|---|---|---|---|---|---|
| B1 | 19.77 (20.3) | 33.94 (35.1) | 22.10 (23.2) | 37.21 (38.6) | +17.43 (+88%) |
| B2 | 43.88 (44.1) | 47.02 (47.9) | 50.83 (52.3) | 54.18 (56.0) | +10.30 (+23%) |
| B3 | 24.57 (25.6) | 37.88 (39.5) | 33.14 (35.9) | 46.88 (49.7) | +22.31 (+91%) |
| B4 (reference) | 14.63 | 24.98 | 19.00 | 29.50 | +14.87 (+102%) |
| Model | 14.20 ± 0.21 | 25.61 ± 0.25 | 17.90 ± 0.24 | 29.19 ± 0.30 | +14.76, +15.10, +15.10 (+105%, +108%, +104%) |
| Uniform control | 43.32 ± 0.02 | 46.18 ± 0.02 | 48.07 ± 0.16 | 51.05 ± 0.24 | — |

[results/shift/table.md; PHASE0 §8]

H9–H11 are confirmed (Section 1). Two further effects have the same sign in all seeds:
- Under higher noise, the model falls *behind* B4: 25.31, 25.60 and 25.92 against 24.98.
- Under larger spikes, it *beats* B4: 17.69, 17.77 and 18.24 against 19.00. Inside events, though, B1 is far better (138.79 against 239.01–260.28).

[results/shift/table.md]

**8.5 Mathematical interpretation.**
- *Noise.* A forecaster's normal-hour error is the target's noise plus whatever input noise its forecast carries.
  - B4 averages 36 values per hour of the week, so it carries almost none. Its normal-hour MAE rises ×1.90 (11.70 → 22.27), close to the ×1.98 rise in noise SD.
  - B3 copies one noisy value, which adds a second noise variance; its MAE rises ×1.71.
  - The model rises ×1.90, ×1.96 and ×1.94 (+11.05, +11.65, +11.58 against B4's +10.57). Its demand inputs pass some noise through, which costs its edge over B4.
- *Spikes.* B1's in-spike error is set by the hour-to-hour change in demand, which doubles with the spike, so its in-event factor is 2.00. The model's factor is larger (2.49–2.79) because its forecast saturates, as Section 9 shows.
- *Additivity.* Noise acts on every hour and spikes on about 2%. The combined rise matches the sum of the single rises to within 1.5% in every seed.

[results/shift/table.md; ratios and differences *computed*]

**8.6 Verdict: learned useful structure, or simply adapted?**

| Seed | Control MAE (B4 14.63) | Beats B4 | Rise, control → both (B4 +14.87) | Label |
|---|---|---|---|---|
| pred_s0 | 14.08 | yes | +14.76 | learned useful structure |
| pred_s1 | 14.02 | yes | +15.10 | simply adapted |
| pred_s2 | 14.49 | yes | +15.10 | simply adapted |

[results/shift/table.md]

**Under Amendment 1, the verdict is inconclusive.** The model's rise minus B4's is −0.11, +0.23 and +0.23 (*computed*): differences of the size of the seed spread.

**Under the original relative rule, it is "simply adapted" in every seed:** +105%, +108% and +104%, against +102% for B4. Amendment 1 replaced that rule before any shift result existed. It noted that a forecaster close to λ shows a large *relative* rise when only the irreducible noise grows.

**What the evidence supports.** The model degrades almost exactly as much as B4, which knows only the calendar. This is consistent with a model that learned mainly the calendar structure, which the shift leaves unchanged, plus a weak recent-demand signal that helps in spikes and costs a little under noise. Neither "transfers better" nor "collapsed" is demonstrated.

## 9. Failure investigation: the forecast saturates during large spikes (§20)

**9.1 Scenario.**
- Series: larger_spikes (shift seed 202).
- Windows: the 104 targets from each spike's second hour onward. In these windows y(t) already shows the spike, and a spike continues into the next hour with probability 0.78 (PHASE0 §3).
- Worst window: target index 7718, where demand is 1,285, the seed-0 forecast is 398 and B1 forecasts 1,030.
- Reproduction: per-window predictions are in `results/shift/predictions_larger_spikes.csv`, and the weights in `results/warehouse/weights_pred_s{0,1,2}.pt`.

[results/failure/table.md]

**9.2 Expected behaviour.** F4 and H10 predicted an error growing faster than B1's because "the attention weights shift sharply" when extreme inputs enter bilinear scores. Since y(t) shows the ongoing spike, a forecaster that uses recent demand should track the spike from its second hour, as B1 does.

**9.3 Actual behaviour.** Mean demand is 589, and B1 forecasts 602. The model forecasts 351, 328 and 308. The seed-0 forecast follows y(t) with slope 0.21, while demand follows it with slope 0.92. When spikes are doubled, in-spike MAE grows ×2.86 for the model and ×2.06 for B1 (130 targets). [results/failure/table.md]

**9.4 Explanation.** The forecast is ŷ = Σ_j A_24,j·g_j + b, a convex combination of the tokens' projected values g_j, with no residual path (A-06). The value gain on demand, ∂g/∂z, is only 0.058, 0.019 and 0.040 per standardised unit. So the values are set mostly by the calendar features, and the forecast can rise only by moving attention onto tokens with higher values. That route is bounded, and it saturates.

*Believed, not tested:* training settles on calendar-dominated values because a large demand gain would make the forecast a smoothed average of recent demand. That is B2's failure, and it would be worse in the 97% of hours with no event.

**9.5 Evidence.**

*Dose–response experiment.* The 104 spike windows of the *control* series are used, with each spike's excess over its no-event level scaled by k:

| k | Expected demand | B1 | Model (seeds 0, 1, 2) | Seed 0: mass on spike tokens / their value level | Seed 1: mass / value level | Ceiling (seed 0) |
|---|---|---|---|---|---|---|
| 0 | 95 | 97 | 94, 94, 94 | 0.089 / 316 | 0.111 / 255 | 581 |
| 1 | 296 | 302 | 232, 223, 193 | 0.070 / 328 | 0.137 / 259 | 583 |
| 2 | 497 | 507 | 321, 302, 274 | 0.130 / 340 | 0.175 / 263 | 585 |
| 4 | 899 | 917 | 384, 367, 374 | 0.311 / 364 | 0.175 / 270 | 590 |
| 8 | 1704 | 1738 | 411, 410, 452 | 0.785 / 412 | 0.107 / 286 | 605 |
| 16 | 3312 | 3380 | 530, 435, 490 | 0.999 / 508 | 0.045 / 317 | 655 |

The source table also has k = 0.5, 1.5 and 3, and `results/failure/dose_response.png` plots the curves. [results/failure/table.md]

B1 tracks demand at every k, and the model plateaus. Seed 0 moves its attention onto the spike tokens, and its forecast converges to roughly their value level (411 against 412 at k = 8). Seed 1 moves its attention *away* from them at large k (peak 0.184 at k = 3 in the source table, then 0.045 at k = 16), yet it plateaus as well. The weights behave differently in the two seeds, so they are not the common cause. The small value gain is.

*Refuted explanations, kept for the record:*
1. "Large spikes push attention away from the spike tokens." This was the F4 reading and the first hypothesis (DEBUGGING.md, Episode 4). Seed 0's mass on spike tokens *rises* with k from k = 1 on (0.070 → 0.999).
2. "The all-token convex-combination ceiling max_j g_j binds." No forecast is within 5% of the ceiling, on either series: mean forecasts are 232 and 351, against ceilings of 583 and 585.

**9.6 Potential improvement.** Give demand magnitude a direct path to the output: a residual connection, or y(t) fed to the head. Attention would then choose where to look, and the value would carry how much. Training on heavier-tailed spikes is a second option. Neither was implemented: the configuration was frozen, and §20 does not require a fix.

**Link to Phase 0.** F4 and H10 predicted the symptom. The predicted mechanism, unstable weights, is wrong.

### 9.7 Failure-mode slices (52-week control series; model = mean of 3 seeds)

| Slice | Count | Model | B1 | B2 | B3 | B4 |
|---|---|---|---|---|---|---|
| F1 spike onset hour | 26 | 211.5 | 211.4 | 209.8 | 205.2 | 208.0 |
| F1 spike hours 1–2 | 52 | 73.4 | 59.4 | 178.9 | 194.9 | 192.9 |
| F2 echo: 24–29 h after a spike starts | 125 | 11.7 | 18.0 | 40.3 | 197.4 | 10.5 |
| F3 Saturday/Monday 00–11 h, normal | 1162 | 10.2 | 15.4 | 34.3 | 24.1 | 10.7 |
| F3 reference: Tue–Thu 00–11 h, normal | 1602 | 9.8 | 15.2 | 35.7 | 13.4 | 9.7 |
| F5 day type differs between t+1 and t−23, normal | 2261 | 11.9 | 18.3 | 40.5 | 25.4 | 12.4 |
| F5 reference: same day type, normal | 5402 | 11.8 | 18.2 | 40.8 | 16.4 | 11.4 |

[results/failure/table.md]

- **F1, confirmed.** All forecasters miss the onset hour by about the same amount. In hours 1–2 the model lags B1, consistent with the saturation already visible at training-size spikes (k = 1 in 9.5: 232, 223, 193 against 296).
- **F2, confirmed for B3.** B3's echo error is 197.4, about ten times its normal-hour error. The model does not echo (11.7), consistent with its small weight on t−23 (H8).
- **F3, weak for the model.** It is 4% worse than the reference slice, against 80% worse for B3 (*computed*). Per-seed values are not reported, so even the 4% is not established.
- **F4, confirmed in its symptom.** During in-event hours, seed 0's mean weight on t rises from 0.027 (control) to 0.052 (larger spikes), but its rows stay diffuse: the largest mean weight is 0.068. The weights move; they do not shift "sharply". [results/shift/metrics.json]
- **F5, not observed** (11.9 against 11.8). B3's gap in that slice is H5's Monday/Saturday step, not F5.

## 10. Cross-problem integration (§21)

**The assumption (§21).** Attention learns weighted relationships among sequence positions; demand has temporal dependencies. Attention can therefore give different importance to different past hours when it predicts the next one.

**One shared function.** `attention(X, W_Q, W_K, W_V, scaled, uniform)` in `wda/attention.py` is the only attention implementation. Both `ToyModel` and `WarehouseModel` call it through `AttentionWeights` (`wda/models.py`). As a result:
- the gradient check and the attention tests cover the warehouse path too;
- the same `scaled` and `uniform` switches drive the ablation, the toy control and the A-27 control.

**A Problem 2 choice justified by Problem 1.** The warehouse model uses scaled attention and an initialisation that gives q and k unit-variance components (W_in ~ N(0, 1/3), then N(0, 1/16)). Problem 1 showed that without scaling the logit SD grows as √d_k, and that strong saturation freezes about one readout row in eight (Section 5). With scaling, the logits stay of order one, and the warehouse runs stayed in that regime:

| Seed | Step-0 logit SD | Step-0 readout entropy | Best-step logit SD | Best-step readout entropy | Best-step max weight |
|---|---|---|---|---|---|
| 0 | 0.949 | 0.907 | 1.941 | 0.769 | 0.157 |
| 1 | 0.721 | 0.951 | 1.532 | 0.834 | 0.131 |
| 2 | 1.153 | 0.920 | 1.303 | 0.866 | 0.117 |

[results/warehouse/metrics.json]

No unscaled warehouse model was trained. That scaling *helped* here is inferred from Problem 1, not demonstrated.

**What the model learns from the history.**
1. Its average weights are nearly flat (H8).
2. Its values carry little demand information (∂g/∂z 0.019–0.058).
3. It behaves like B4's calendar profile plus a weak current-level signal. Normal-hour MAE is 12.26–12.61, against B4's 11.93. Inside events it is 77.95–108.86, against B4's 151.99 and B1's 67.82.

**Verdict: the behaviour largely did not support the assumption.** The weighting is used: fixing it to uniform raises MAE from 14.97 to 43.21. But the evidence points to weighting that *selects calendar information*, not weighting of informative demand history:
- H8 is refuted;
- the forecast follows y(t) with slope 0.21 during large spikes;
- the model does not echo t−23 (F2).

The "calendar lookup" reading is believed, not tested; removing the calendar features would test it.

## 11. Debugging (§24)

[DEBUGGING.md](DEBUGGING.md) records four episodes as they happened.

**Episode 1 (T-202, commit `3f55f0f`) has all six §24 steps.**
1. Initial implementation: the attention core and tests, all passing.
2. Unexpected result: a planted S = KQᵀ bug still passed every attention test.
3. Hypothesis: the tiny example's S was symmetric, and the random tests checked only invariants KQᵀ also satisfies.
4. Diagnostic: printing S showed it was symmetric.
5. Correction: an asymmetric W_K and an element-by-element reference test.
6. Verification: five planted bugs (KQᵀ, no scaling, wrong softmax axis, division by d_k, AᵀV) each fail at least one test, and on the correct code the new `test_matches_elementwise_reference_catches_kqt_on_any_input` passes along with the rest of the 19 tests of that time.

A later review fix (`3af601f`) made A asymmetric too. The committed trace (S = [[4, 1], [4, 5]]) is that version; the journal's step 5 quotes the intermediate matrix.

**Episodes 2–4.**
- Episode 2 (T-304, `6d5539a`): NumPy booleans broke `metrics.json`; they are now converted, with a test.
- Episode 3 (T-505): a no-op refactor (`330d426`) changed shift results in the last float32 bit. The committed run had been made under machine load, which can change the maths library's summation order; reruns on the new code were identical.
- Episode 4 (T-506): the first explanation of the spike failure was refuted, and the dose–response experiment refined the second (Section 9.5).

## 12. Known issues and limitations

**Unresolved problems.**
- The spike saturation is diagnosed but not fixed.
- The §19 and A-27 verdicts are inconclusive. Their margins, tenths of an order per hour, are of the size of the seed spread, and the 8-week test split has only 33 in-event targets.
- It is unknown whether the failed unscaled d_k = 64 seed would learn with more steps.
- The toy plateau's mechanism is a belief.
- Warehouse seed 1's best step was its last, so it may be under-trained.

**The calendar-feature confound** (Section 7.6). The model sees calendar features that B1–B3 do not. B4, the calendar-only answer, nearly ties the model, and the uniform control cannot separate weighting history from using the calendar. No calendar-feature ablation was run.

**Reproducibility tolerance.** Generated data are bit-exact for a given seed. Model outputs are bit-exact on an idle machine, and can differ in the last float32 bit under machine load: seen in the shift results (Episode 3) and in four ablation gradient norms during the pre-submission check (README). Identical training across machines is not promised (A-24).

**Tests and statistics.**
- No test is skipped or marked as an expected failure, and no gradient discrepancy is open.
- `test_stable_softmax_matches_naive_on_moderate_logits` compares naive and stable softmax on moderate float64 logits only. The finite-but-wrong V2 case, [88.5, 87.5, 0] in float32, is covered by `test_stable_softmax_is_correct_where_naive_is_finite_but_wrong` (tests/test_attention.py).
- Three seeds per configuration; the F-mode slices give seed means only.
- The shift series have 27 spike onsets each, against ≈ 34 expected.
- The shift and failure stages ran before the freeze commit, using the trained weights and no test-split data.

**Not built:**
- a residual path, more heads or more layers (A-03);
- an unscaled warehouse model;
- a calendar ablation;
- longer runs for the failed seed;
- an hour-of-day error breakdown for the model;
- manual backpropagation (optional, §10).

The README's "Deliberately not built" list gives the scope reasons (§27, §28).
