# Phase 0 design

**Version 1 · 2 October 2026.** Written before any implementation code or result exists. Frozen once committed: every later change, including values this document says will be added, is a dated entry under *Amendments*. Sections 4 and 5 hold the pre-registered hypotheses and failure modes. Drafted with AI assistance; see `AI_LOG.md`. § = PRD section, A-xx = `ASSUMPTIONS.md`.

Values marked *expected* come from a closed-form calculation over the generator's equations (Gaussian approximation of the count noise, no random draws, no project code), committed as `prereg/design_expectations.py` with its output `prereg/design_expectations.txt`. The initial-attention figures in H2 and H3 come from `prereg/init_stats.py` / `prereg/init_stats.txt`. They are predictions, not results.

## 1. Mathematical problem formulation

**Warehouse task.** Hours t = 0 … 8,735, hour of day h(t) = t mod 24, day d(t) = ⌊t/24⌋ mod 7 (0 = Monday); y(t) ∈ ℕ orders/hour. For target y(t + 1), the input is 24 tokens (A-04, A-05); token j = 1 … 24 is hour u = t − 24 + j, so j = 1 is t − 23 and j = 24 is t:

f_j = [ z(u), sin(2πh(u)/24), cos(2πh(u)/24), sin(2πd(u)/7), cos(2πd(u)/7) ] ∈ ℝ⁵,  z = (y − μ)/s,

with μ, s from the training targets (A-13), and F = [f_1; …; f_24] ∈ ℝ^(24×5). The model ŷ = f_θ(x) has θ = {W_in ∈ ℝ^(5×16), b_in, W_Q, W_K, W_V ∈ ℝ^(16×16), w_out ∈ ℝ^16, b_out} (881 parameters):

- X = F W_in + 1 b_inᵀ (24 × 16); Q = X W_Q, K = X W_K, V = X W_V (no biases, as in §7)
- S = Q Kᵀ, S′ = S/√d_k (24 × 24); A_ij = exp(S′_ij − max_l S′_il) / Σ_m exp(S′_im − max_l S′_il); Y = A V
- ẑ = Y_24 · w_out + b_out, ŷ = μ + s ẑ (readout at the last position, A-06)
- Loss (A-14): L(θ) = (1/|𝓑|) Σ_(τ∈𝓑) (ẑ_τ − z(τ))² over mini-batches of training targets.

One head, no mask, no residual path (A-03, A-07). Rows of A sum to 1, so ẑ = Σ_j A_24,j g(f_j) + b_out with g affine: weight on j = 24 approximates B1, on j = 1 B3, uniform weight B2, so losing to a baseline is not a capacity limit. Variants (A-26): unscaled S′ = S (A-19); uniform A_ij = 1/n (A-18, A-27).

**Toy task** (A-18). Eight pairs with distinct keys and distinct values, each from 16, in random order, then a query: x_i = [e_(k_i); e_(v_i); 0] and x_9 = [e_(k*); 0; 1] ∈ ℝ^33 (e one-hot, last entry a query flag), with k* = k_m, m uniform. Target v_m (16 classes); X = tokens; logits = Y_9 W_out + b_out; cross-entropy loss. The query contains its own key, so the model must learn to use the flag to stop the query attending to itself.

### Design values: models and training

| | Toy (H1–H4) | Warehouse (H5–H11) |
|---|---|---|
| Tokens × features | 9 × 33, no input projection | 24 × 5 → 16; W_in ~ N(0, 1/3), b_in = 0, since E‖f‖² = 1 + 4·½ = 3 |
| d_k, d_v | 16, 16; ablation d_k ∈ {4, 64} | 16, 16 |
| W_Q, W_K, W_V entries | N(0, 1/2), as ‖x‖² = 2 | N(0, 1/16) |
| Effect of initialisation | Var(q_j) = Var(k_j) = 1, so Var(q·k) = d_k (A-19) | same |
| Head; parameters | linear → 16 logits; 1,856 (1,064 / 5,024 at d_k = 4 / 64) | linear → 1; 881 |
| Data | 50,000 / 2,000 / 5,000 sequences, seeds 301 / 302 / 303 | 6,024 / 1,344 / 1,344 windows, seed 101 |
| Budget | 3,000 steps (≈ 15 epochs) | 4,000 steps (≈ 170 epochs) |
| Weights kept | best validation accuracy | best validation MAE |
| Common | Adam (β 0.9, 0.999; ε 1e-8), learning rate 1e-3, batch 256, evaluation every 50 steps, training seeds 0, 1, 2 shared by paired arms (A-16); no weight decay, schedule or clipping (clipping would hide the gradient effects under study) | |

**Logged at step 0 and every evaluation:** losses and validation metric; row entropy of A over ln n (all rows; readout row); readout maximum weight; SD of S′; on a fixed diagnostic batch, ‖∂L/∂W_Q‖, ‖∂L/∂W_K‖, ‖∂L/∂W_V‖ and readout-row ‖∂L/∂q‖ percentiles (10, 50, 90).

## 2. Synthetic dataset design

52 weeks (8,736 h) from a Monday; no holidays, clock changes (A-08) or trend (A-10).

λ₀(t) = B + D(h) + W(d),  λ(t) = max(λ_min, λ₀(t) + E(t)),  E(t) = (m(t) − 1) λ₀(t),

where m(t) is the active event's multiplier (1 when none): §4's additive sum with Trend = 0 and an event term that scales with the level. D(h) = 60 cos(2π(h − 14)/24) + 15 cos(4π(h − 14)/24) peaks at +75 at 14:00 and sums to zero over a day; the second harmonic flattens the night (≈ −45, 00:00–04:00) and steepens the ramps to ≈ 20 orders/h. λ₀ runs from 30 (Sunday 02:00) to 180 (Monday–Thursday 14:00).

**Counts** (A-09): y | λ ~ NegBin(mean λ, dispersion r), P(y = k) = Γ(k + r)/(k! Γ(r)) · (r/(r + λ))^r (λ/(r + λ))^k, Var = λ + λ²/r. Equivalently y ~ Poisson(g), g ~ Gamma(shape r, mean λ): the hour's rate varies with coefficient of variation 1/√r (10% at r = 100). Counts are independent given λ.

**Events** (A-11) start, in an hour with no event active, with the probabilities below; one constant multiplier per event; no overlaps; never a model input. **Seeds:** each seed is split into an event stream and a noise stream, so changing r leaves the event timeline unchanged; regenerating means rerunning (§4).

| Parameter | Value | Reason |
|---|---|---|
| Base level B | 100 orders/h | scale |
| Daily profile D(h) | as above | amplitude 75 ≥ 5 × noise SD |
| Weekday offsets W(d) | Mon–Thu +5, Fri 0, Sat −15, Sun −25 | weekend dip; the +30 Sunday→Monday step is the hardest transition (F3) |
| Dispersion r | 100 | noise SD 13.8 at the mean level |
| Floor λ_min | 1 | valid mean; never binds (lowest level: a night drop, 0.3 × 30 = 9) |
| Spikes | onset p_s = 0.004/h; U{3 … 6} h; m ~ U(2, 4) | ≈ 34/year; lasting beyond one hour makes an ongoing spike predictable |
| Drops | onset p_d = 0.002/h; U{2 … 4} h; m ~ U(0.3, 0.6) | ≈ 17/year |
| Seeds | data 101, shift 202 | fixed now |

**Design-target check (section 4).**
1. Mean expected level 100 − 20/7 = 97.1, where σ = √(97.1 + 97.1²/100) = 13.8; amplitude 75/13.8 = 5.4 × σ ✓.
2. Waits between events are geometric with mean 1/0.006 = 167 h; events last (0.004·4.5 + 0.002·3)/0.006 = 4.0 h on average; share of hours 4.0/170.7 = 2.3% ✓ (≤ 5%).
3. Spike multiplier m ∈ [2, 4] by construction ✓.

Expected counts: 24 spikes and 12 drops in training; 5.2 and 2.6 in the test split and in each shift series, whose 1,344 targets fall ≈ 31 inside events, ≈ 176 within 24 h after one and ≈ 1,137 elsewhere (A-11 groups). Five spikes per series is thin, but comparisons are paired on the same hours, and a higher rate would make events less occasional.

## 3. Statistical assumptions

### Temporal structure
Hourly: dependence between hours comes only from λ. Daily: D(h), identical every day. Weekly: W(d), a step function. Trend: none; λ₀ repeats every 168 h, so test weeks differ from training only in their random draws.

### Noise
Negative binomial, r = 100. SD 6.2 orders/h at the Sunday-night trough (λ = 30), 13.8 at the mean level, 22.4 at the weekday peak, 77 inside a 4× peak-hour spike. The variance grows with the level (CV 21% at λ = 30, 12% at λ = 180, limit 10%), so absolute errors are largest at the peak and in spikes. A forecaster knowing λ(t + 1) would score an expected normal-hour MAE of 11.0 (not a strict bound: the median minimises MAE, A-09).

### Unusual events
Onsets are memoryless (hazard 0.006/h) and unannounced, so the first event hour cannot be forecast (F1). An ongoing spike continues into the next hour with probability E[L − 1]/E[L] = 3.5/4.5 = 0.78 (drops 2/3), so y(t) is informative during an event. Events do not recur at lag 24, so an event at t − 23 says nothing about t + 1 (F2).

### Correlation
Informative history: y(t), the current level including any ongoing event; y(t − 23), whose expected level differs from λ₀(t + 1) only by W(d) − W(d − 1) (zero Tuesday–Thursday). Lag 168 is outside the window, so weekly structure arrives only through the weekday features (A-05). Noise is independent across hours, so other inputs help only by averaging it.

| Lag | Expected ACF, no events | With events (design) | Why |
|---|---|---|---|
| 1 | 0.87 | 0.84 | smooth profile; events last 3–6 h |
| 12 | −0.71 | −0.52 | opposite phase of the daily cycle |
| 24 | 0.87 | 0.63 | Monday step; events do not recur |
| 168 | 0.91 | 0.66 | only noise and events differ |

Events add a third of the variance (3,287 vs 2,252) but little lag-24 or lag-168 covariance (renewal approximation). The measured lags 1, 24 and 168 are added under *Amendments* at the first generator run (FAC-10); a gap above about 0.1 is investigated as a possible generator bug before any training.

### Distribution shift
Section 8 (A-20): r = 100 → 14 and spike multipliers × 2, separately and together.

## 4. Hypotheses

**Status:** written on 1 October 2026, before any implementation code or experiment result exists. This section is committed before the first result (ASSUMPTIONS A-22) and is not edited afterwards. Changes go in the *Amendments* section with a date. Outcomes are reported in the results documents by hypothesis ID.

**Drafting:** prepared with AI assistance; see the AI-assistance log (A-23).

Unless stated otherwise:
- MAE is the test-set MAE in orders per hour, averaged over three training seeds (A-14, A-16).
- B1 = last observation, B2 = 24-hour moving average, B3 = seasonal naive y(t−23) (A-15).
- "Uniform" means equal weight on all positions.

**Design targets these predictions assume** (fixed in Phase 0):
- The daily amplitude (peak minus mean) is at least 5× the noise standard deviation at the mean level.
- Events affect at most about 5% of hours.
- Spikes raise demand to about 2–4× the expected level.

### Problem 1: attention mechanism

#### H1 · Attention learns the toy task, and attention is what solves it
- **Experiment:** associative recall (A-18).
- **Prediction:**
  - Scaled attention reaches **≥ 95%** accuracy on held-out sequences.
  - The uniform-attention control stays near **1/n_pairs**, at most 1.5/n_pairs. n_pairs is the number of key–value pairs in a sequence.
- **Refuted if:** attention stays below 90%, or the control exceeds 1.5/n_pairs.
- **Why:**
  - The answer is stored in one token, chosen by content. Matching the query to that key is exactly what QKᵀ followed by softmax computes.
  - Uniform weights average all the tokens together. The head can still see which values are present, but not which one belongs to the query, so the control can only guess among them.

#### H2 · At initialisation, unscaled attention is saturated when d_k is large
- **Experiment:** the ablation (A-19), measured before the first update. Q and K components have unit variance at initialisation.
- **Prediction:** mean row entropy of A, as a fraction of its maximum ln n:

  | | d_k = 4 | d_k = 64 |
  |---|---|---|
  | Scaled | ≥ 0.75 | ≥ 0.75 |
  | Unscaled | between 0.5 and 0.75 | ≤ 0.25 |

- **Refuted if:** unscaled entropy at d_k = 64 is above 0.25, or scaled entropy at either d_k is below 0.75.
- **Why:**
  - For independent unit-variance components, q·k has variance d_k. The unscaled scores therefore have a standard deviation of about 2 at d_k = 4 and about 8 at d_k = 64; scaling brings both back to about 1.
  - A simulation run before implementation gave entropy fractions of about 0.85 (scaled), 0.64 (unscaled, d_k = 4) and 0.15 (unscaled, d_k = 64), for n between 9 and 24.

#### H3 · Saturation makes gradients uneven, not uniformly small
- **Experiment:** same as H2. Per-row gradient norm of the loss with respect to the query vectors, at initialisation.
- **Prediction:** at d_k = 64, unscaled versus scaled:
  - **≥ 5%** of unscaled rows have a gradient below 1% of the scaled median; scaled has none.
  - The ratio between the 90th and 10th percentiles of gradient norm is **≥ 100×** unscaled, against **≤ 10×** scaled.
  - The *median* gradient is **not** smaller for unscaled: the unscaled median is at least 0.5× the scaled median.
- **Refuted if:** any of the three fails.
- **Why:**
  - The softmax Jacobian, diag(a) − aaᵀ, approaches zero as a row of A approaches one-hot, so strongly saturated rows barely learn.
  - Removing the 1/√d_k factor multiplies every score gradient by √d_k = 8, which more than compensates in the rows that are not fully saturated.
  - The same simulation gave about 9–16% near-zero rows, percentile ratios above 1000× for unscaled versus about 5× for scaled, and an unscaled median 0.9–2.1× the scaled median (`prereg/init_stats.txt`).
  - This is the training-dynamics phenomenon analysed for §13.

#### H4 · Unscaled attention trains worse at d_k = 64, but about the same at d_k = 4
- **Experiment:** the ablation (A-19). Optimiser and learning rate fixed in advance; three paired seeds per cell.
- **Prediction:**
  - At **d_k = 64**, unscaled needs **≥ 1.5×** as many steps as scaled to reach 95% validation accuracy (median over seeds), or fails to reach it within the step budget in at least one seed.
  - At **d_k = 4**, the ratio is between 0.67 and 1.5.
  - In every cell where both variants reach 95%, their final accuracies differ by less than 5 percentage points.
- **Refuted if:** the d_k = 64 ratio is below 1.5 with no failed seed, or the d_k = 4 ratio falls outside [0.67, 1.5].
- **Why:**
  - Unscaled attention is scaled attention with W_Q multiplied by √d_k, so both can represent the same solution. The difference is in optimisation, which starts from the saturated, uneven state of H2–H3.
  - Caveat stated in advance: Adam rescales updates per parameter, which may shrink the gap. If the gap is small, that is the first explanation to test.

#### V1 · Gradient verification (expected result)
- **Prediction:** in float64 with h = 1e-6, every entry agrees within 1e-8 + 1e-6·|g_num| (A-17). For entries not close to zero, a relative difference around 1e-9 is expected.
- **Why:** central differences have truncation error ∝ h² and round-off error ∝ ε/h. In float64 (ε ≈ 1.1e-16) both are far below the tolerance at h = 1e-6. float32 (ε ≈ 6e-8) would not be, which is why the check runs in float64.

#### V2 · Numerical stability (expected result)
- **Prediction:**
  - Naive float32 softmax returns inf or NaN once the largest logit exceeds about 88.7, where exp overflows.
  - The max-subtracted version stays finite for any finite logits.
  - Wherever the naive version is finite, the two agree to within 1e-6.
- **Why:** softmax(z − c) = softmax(z) for any constant c. Subtracting the maximum keeps every exponent ≤ 0.

### Problem 2: warehouse demand

#### H5 · Expected baseline behaviour
- **Prediction:**
  - On test MAE, **B1 < B3 < B2**, and B2 is **1.4–2.0×** B3 (calculated 1.7).
  - In normal hours, B1 and B3 are within 10% of each other.
  - B3's normal-hour MAE on Mondays is **≥ 1.5×** its Tuesday–Thursday MAE.
- **Refuted if:** the order differs, B2/B3 falls outside [1.4, 2.0], or the Monday ratio is below 1.5.
- **Why:** this is the closed-form expectation from the generator (§7; `prereg/design_expectations.txt`):
  - normal hours: B1 18.4, B2 40.5, B3 19.1;
  - all hours: B1 20.3, B2 44.1, B3 25.6;
  - B2 discards the daily shape;
  - B1 and B3 nearly tie in normal hours, but B3 pays for each spike twice, once inside it and again 24 hours later through y(t−23) (F2);
  - B3 also pays for the Monday step (Sunday −25 → Monday +5).
- **History:** the first draft predicted B3 < B1 and B2 ≥ 2× B3. It was revised on 2 October, after the design calculation and before any code existed, because that draft left out the Monday step and the spike echo.

#### H6 · Attention beats the best baseline by a modest margin
- **Prediction:**
  - The attention model's test MAE is **5–20% lower** than the best baseline (expected to be B1, per H5).
  - The gap has the same sign in all three seeds (A-16).
  - The uniform-attention control (A-27) has an MAE **at least 1.5×** the attention model's.
- **Refuted if:** the attention model is no better than the best baseline in all three seeds, or the control is within 1.5× of it. An improvement above 20% means the prediction was wrong about the size.
- **Why:**
  - The model can copy B3 by putting its weight on t−23, and can improve on it by mixing in recent hours and day-of-week information.
  - With a linear head on a weighted average (A-06), it cannot learn arbitrary nonlinear daily shapes, so the gain is bounded.
  - Losing would point to optimisation or generalisation, not capacity (A-06).
  - Every window covers all 24 hours of the day, so averaging the hour features uniformly cancels them exactly. The control is left with roughly a daily average plus day-of-week information, close to B2, which H5 expects to be about 2.2× worse than B1 and 1.7× worse than B3.

#### H7 · The gain comes mostly from event-affected hours
- **Prediction:** relative to B3:
  - the attention model's MAE improvement is **at least twice as large** for targets within 24 hours after an event (A-11 group 2) as for normal hours;
  - in normal hours the improvement is below 10%.
- **Refuted if:** the normal-hour improvement is ≥ the post-event improvement.
- **Why:** B3 repeats yesterday's spike 24 hours later, an echo error. The model can see from y(t) and the recent hours that no event is in progress, and lower its weight on t−23.

#### H8 · Attention concentrates on the same hour yesterday and on the latest hour
- **Prediction:** averaged over test windows, the readout row puts **≥ 50%** of its weight on positions t−23 and t together, against 8.3% under uniform weights.
- **Refuted if:** their combined weight is below 50%.
- **Least certain part:** which of the two gets more weight. Hours t and t+1 (the hour of t−23) are neighbours on the hour circle, so the time features barely separate them (A-05).
- **Why:**
  - Token t−23 carries both yesterday's demand at the target hour and the target hour's own calendar features.
  - Token t carries the current demand level.
  - Following §16, this is a statement about the weights, not a causal claim.

### Generalisation (A-20)

#### H9 · Higher noise: everything degrades, the ranking holds
- **Change:** noise standard deviation at the mean level about doubled; nothing else changes.
- **Prediction:**
  - Normal-hour MAE of B3 rises by **1.7–2.1×**, and of B1 by **1.4–2.0×**.
  - The attention model's MAE rises by less in absolute terms than B3's, and it stays better than B3.
- **Refuted if:** the attention model's absolute increase exceeds B3's, or it falls behind B3.
- **Why:**
  - Each baseline's error contains the noise from two observations, so it scales with the noise.
  - B1's error also contains the hour-to-hour change in the daily pattern, which does not scale with the noise, so it rises less.
  - The model's error contains the target noise plus a down-weighted contribution from its inputs, so it grows less in absolute terms than B3's.

#### H10 · Larger spikes: the attention model reacts worse than last-observation
- **Change:** spike magnitude doubled; nothing else changes.
- **Prediction:**
  - Normal-hour MAE changes by **less than 10%** for every model.
  - Event-hour MAE rises for every model, and rises by a larger factor for the attention model than for B1.
  - In the first three hours after a spike starts, B1's MAE is lower than the attention model's.
- **Refuted if:** the attention model's event-hour MAE factor is ≤ B1's, or it beats B1 in the first three hours after onset.
- **Why:**
  - Standardised spike inputs fall outside the training range.
  - The scores are bilinear in the token values, so extreme inputs distort the attention weights.
  - B1 simply copies the latest level and adapts within one hour.

#### H11 · The combined shift is roughly additive
- **Change:** both changes together (§6's example).
- **Prediction:** the attention model's MAE increase under the combined shift is within ±25% of the sum of its increases in H9 and H10.
- **Refuted if:** the combined increase falls outside that range.
- **Why:** the two changes act on different hours (noise everywhere, spikes in about 5% of hours), so they should interact only weakly.

## 5. Failure modes

| ID | Scenario | Predicted behaviour | Why |
|---|---|---|---|
| F1 | First hour of an unannounced spike | Every model misses it by roughly the spike size. The attention model may also lag for the following hours (H10). | Nothing in the window signals the onset, so the error is irreducible. |
| F2 | 24 hours after a spike | B3 and any model relying on t−23 predict a spike that does not happen (the echo error). | The spike sits at position t−23, the "same hour yesterday" (H7). |
| F3 | Weekday ↔ weekend transitions (Saturday, Monday) | Systematic residual error for the attention model, largest in the first hours of the new day type. | A single sin/cos day-of-week pair is smooth, so it cannot represent a step from weekday to weekend exactly (A-05). |
| F4 | Spikes larger than any seen in training (H10) | The attention weights shift sharply, and the error grows faster than for B1. | §20's "instability under extreme inputs": standardised values outside the training range enter bilinear scores (A-21). |
| F5 | Days where "latest hour" and "same hour yesterday" disagree in day type | Attention puts weight on t−23 when it should not, or the reverse, and the error is larger on those days. | Hours t and t+1 are neighbours on the hour circle, so only day-of-week and demand can separate the two positions (A-05). |

Where each mode comes from and how its symptom is measured:

| ID | Generator term or model choice involved | Symptom measured as |
|---|---|---|
| F1 | unannounced onset of E(t) | error in each event's first hour |
| F2 | E(t) at j = 1; readout weight on t − 23 | group-2 error 24 h after event hours |
| F3 | W(d) step; one sin/cos weekday pair | normal-hour MAE by target day and hour |
| F4 | spike multiplier × bilinear scores (XW_Q)(XW_K)ᵀ | in-spike MAE and readout entropy, series 1 vs 3 |
| F5 | near-equal hour features at j = 1 and j = 24 | readout weights on t and t − 23 when the day type changes |

The §20 investigation starts from F2, which arises in normal evaluation (A-21) and has a counterfactual test: replace the window's event hours by λ₀ and predict again. If F2 is not among the largest test errors, it moves to F4 on series 3.

## 6. Evaluation methodology

| Split (A-12) | Weeks, by target hour | Targets | Role |
|---|---|---|---|
| Train | 1–36 | 6,024 | fits θ; supplies μ, s |
| Validation | 37–44 | 1,344 | selects the checkpoint (lowest MAE), the reference baseline (lowest MAE of B1–B3) and any hyperparameter change (as an amendment) |
| Test | 45–52 | 1,344 | scored once, after the configuration is frozen |

A window belongs to the split of its target hour; inputs from an earlier split are past data, so nothing leaks, and whole weeks keep the weekday mix equal (§18). **Test once:** the configuration is committed before the first test-scoring run, which happens only after all seeds are trained and the validation choices are made; reruns of that configuration are allowed, and any later change is an amendment that keeps the original numbers.

**Metrics** (A-14), in orders/hour over target hours T (N = |T|), after ŷ = μ + s ẑ:
- MAE = (1/N) Σ |y(τ) − ŷ(τ)| (primary); RMSE = √((1/N) Σ (y(τ) − ŷ(τ))²) (secondary); mean signed error = (1/N) Σ (ŷ(τ) − y(τ)), positive = over-forecast.
- Reported overall and per A-11 group with counts: (1) τ inside an event; (2) τ outside events with an event hour in the window, i.e. within 24 h after one; (3) others. §17's Difference = model − baseline, in orders/h and %.

*Why.* MAE grows linearly, so the ≈ 2% of targets inside spikes, missed by ≈ 200 orders, do not dominate it; RMSE shows how badly spikes are missed (one 200-order miss weighs as much as ≈ 200 normal-hour misses of 14). The signed error tests a predicted bias: MSE estimates the conditional mean, which onset risk lifts by ≈ +0.7 orders/h at the mean level (p_s·E[m − 1] = 0.8% of λ, net of drops). *Not MAPE:* dividing by y weights a Sunday-night miss (λ = 30) six times a weekday-peak miss (λ = 180), and drops (9–18 orders/h) can give y = 0, where MAPE is undefined.

**Seeds and verdicts.** Seeds 0, 1, 2 set initialisation and batch order; results are mean ± SD with every run shown. A-16: an effect is claimed only with the same sign in all three seeds, paired by seed where arms share one (ablation arms; model and control); otherwise inconclusive. A-27, on test, with Δ_ref(s) = MAE_model(s) − MAE_ref and Δ_ctl(s) = MAE_model(s) − MAE_uniform(s):
- *adds value* if Δ_ref(s) < 0 and Δ_ctl(s) < 0 for every s;
- *does not* if Δ_ref(s) > 0 for every s, or Δ_ref(s) < 0 for every s but Δ_ctl(s) ≥ 0 for every s (gain attributed to the calendar inputs; the confound is stated);
- *inconclusive* otherwise.

**Toy and ablation metrics.**
- Accuracy: share of test sequences whose arg-max logit is the target (chance 1/16).
- Steps to 95%: first evaluation (50-step resolution) with validation accuracy ≥ 0.95, else "not reached by 3,000" (H4).
- Initial entropy fraction: mean over rows of H(A_i)/ln n, H(a) = −Σ_j a_j ln a_j, at step 0 on the diagnostic batch (first 256 validation sequences); paired arms start from identical weights.
- Per-row query gradient: ‖∂L/∂q_9‖₂ at step 0, one readout row per sequence. Only that row enters the loss, so ∂L/∂q_i = 0 exactly for i ≠ 9; counting those rows would make 8/9 "near zero" by construction, so H3 uses the 256 readout rows, as in `prereg/init_stats.py`.
- Gradient check (V1, A-17): an entry passes if |g_auto − g_num| ≤ 1e-8 + 1e-6·|g_num|; every relative difference above 1e-6 is listed with its cause (expected only where g_num ≈ 0), ending "Unexplained entries: 0" (FAC-20).

## 7. Expected baseline behaviour

Baselines (A-15) on the raw window: **B1** ŷ(t + 1) = y(t); **B2** ŷ(t + 1) = (1/24) Σ_(i=0…23) y(t − i); **B3** ŷ(t + 1) = y(t − 23).

In normal hours each error is a deterministic part plus noise (SD ≈ 19.6 at the mean level for B1 and B3, from two observations). The deterministic part is D(h + 1) − D(h) for B1 (mean absolute 10.0, up to 20 on the ramps, ≈ 0 at night); D(h + 1) for B2 (mean |D| = 39.2, as 24 hours average D to zero); and W(d) − W(d − 1) for B3 (0 Tuesday–Thursday, −5 Friday, −15 Saturday, −10 Sunday, +30 Monday). Using E|e| = s_e√(2/π) exp(−μ_e²/2s_e²) + μ_e(1 − 2Φ(−μ_e/s_e)) for e ~ N(μ_e, s_e²), averaged over the week and integrated over event onset, duration and multiplier:

| Expected | B1 | B2 | B3 |
|---|---|---|---|
| MAE, normal hours | 18.4 | 40.5 | 19.1 |
| Extra absolute error per spike (orders) | 460 | 930 | 1,580 |
| MAE, all hours | 20.3 | 44.1 | 25.6 |

In normal hours B3 is 16.5 Tuesday–Thursday but 31.4 on Monday and 19.9 on Saturday; B1 stays at 16.2–19.8 every day.

**Predicted ranking: B1 < B3 < B2, B2/B3 ≈ 1.7.** B2 discards the daily shape. B1 and B3 nearly tie in normal hours (a 0.6 gap, about one split's sampling error), but each spike costs B3 twice, inside the spike and again 24 h later when y(t − 23) echoes it (F2), while B1 misses only onset and end. With ≈ 5 test spikes the overall gap is ≈ 5 ± 2; B1 < B3 should hold even with only one or two test spikes.

**Relation to H5.** This calculation was made on 2 October, when the generator values were fixed. It showed that the first draft of H5 (B3 < B1; B2 ≥ 2 × B3) had left out the Monday step and the spike echo, so H5 was revised to this ranking before the first commit and before any code existed. H6 is scored against the reference baseline that validation selects (A-15).

**Uniform controls.** Toy (H1): Y_9 averages all values, revealing which 8 are present but not which is bound to the queried key, so accuracy ≈ 1/8. Warehouse (H6): the hour features cancel over a full day, so the control sees the hour only through the window's mix of two days' weekday features, which is linear in the hour. It can fit at most a straight ramp, not the 14:00 peak, so its MAE should be near B2's (≈ 40–44), the basis of H6's 1.5× margin.

## 8. Planned generalisation experiment

**Design (A-20).** Evaluation only: the three trained models and their controls are loaded without retraining, with the training μ and s (A-13). Four series come from shift seed 202, each 8 weeks on the test calendar after a 24-hour warm-up (1,344 targets). They share the event stream, so onsets, durations and multiplier draws u are identical (m = 2u with larger spikes); count noise is not paired.

| Series | Parameter | Original | Shifted | Reason |
|---|---|---|---|---|
| 1 Control | none | section 2 | seed 202 only | separates new draws from a changed distribution |
| 2 Higher noise | r | 100 | 14 | noise SD at the mean level 13.8 → 27.8 (×1.6 at λ = 30, ×2.2 at λ = 180), λ unchanged; does the model average inputs or copy them? (H9) |
| 3 Larger spikes | spike m | U(2, 4) | U(4, 8), same draws × 2 | expected spike level up to 1,440 orders/h (z ≈ 23) against at most 720 (z ≈ 11) in training; mean excess 2λ → 5λ; extrapolation through bilinear scores (H10, F4) |
| 4 Both | r, m | 100, U(2, 4) | 14, U(4, 8) | §6's example; additivity (H11); decides §19 |

Everything else is unchanged. **Measured** per series, model seed, control and baseline: MAE, RMSE and signed error, overall and per A-11 group with counts; MAE over the first three spike hours (H10); the mean readout row per group; change from series 1 (orders/h, %).

**Expected impact.** H9–H11 (section 4) predict the model; the section 7 calculation gives the baselines:

| Expected MAE, all hours | 1 Control | 2 Noise | 3 Spikes | 4 Both |
|---|---|---|---|---|
| B1 | 20.3 | 35.1 | 23.2 | 38.6 |
| B2 | 44.1 | 47.9 | 52.3 | 56.0 |
| B3 | 25.6 | 39.5 | 35.9 | 49.7 |

Higher noise raises normal-hour MAE ×1.76 (B3) and ×1.79 (B1), inside H9's ranges; larger spikes leave normal hours unchanged; the combined shift is near-additive (B1 +18.3 vs 14.8 + 2.9; B3 +24.1 vs 13.9 + 10.3), as H11 assumes, because noise acts on every hour and spikes on ≈ 2%.

**§19 verdict (fixed now).** For each seed, R_model(s) = MAE₄/MAE₁ − 1 against the reference baseline's R_ref (expected +90% for B1, +94% for B3): *learned useful structure* if R_model(s) ≤ R_ref for every s; *simply adapted to the training distribution* if R_model(s) > R_ref for every s; otherwise *inconclusive*. The measure is relative because series 4 is harder for every forecaster; the reference shows how much harder for one that learned nothing. B2 rises only 27% because the unchanged daily shape dominates its error.

## Amendments

All three were made on **2 October 2026**, after the toy task and the ablation had run, and **before any warehouse or shift-series result existed**. They came from an AI-assisted design review (see AI_LOG.md) and were approved by the candidate. The text above is unchanged; these entries override it where they conflict.

### Amendment 1 · The §19 verdict rule (section 8; A-20)
- **Replaces:** "learned useful structure if R_model(s) ≤ R_ref for every s", where R is the *relative* MAE increase from series 1 to series 4.
- **Why:**
  - Noise raises every forecaster's unavoidable error, and a good model's error is mostly that error. So the relative rule labels a perfect forecaster (one predicting λ) "simply adapted": its R is about +1.12, against B1's +0.88.
  - It also credits B2: B2's error is dominated by the unchanged daily shape, so B2 rises only about 27%.
- **New rule,** for each seed s:
  - *learned useful structure* if the model beats the reference baseline on series 1, MAE₁(s) < MAE₁(ref), **and** its absolute increase MAE₄(s) − MAE₁(s) is no larger than the reference's;
  - *simply adapted to the training distribution* if the increase is larger than the reference's for every s;
  - *inconclusive* otherwise.
- The original relative rule's outcome is still reported alongside, for transparency.

### Amendment 2 · Length of the shift series (sections 2 and 8; A-20)
- **Replaces:** "each 8 weeks on the test calendar after a 24-hour warm-up (1,344 targets)".
- **New value:** each series is **52 weeks** (8,736 targets) after the 24-hour warm-up, starting on a Monday. Seed 202, the shared event stream and the four conditions are unchanged.
- **Why:** counting the events in the seed-202 8-week series (data only, no model output) found **one** spike and three first-three-hour targets. H10, F4 and the spike half of H11 would have rested on a single event. 52 weeks gives about 34 spikes. The series is evaluation-only, so the cost is seconds.
- **Effect on predictions:** none of the numbers in H9–H11 changes.

### Amendment 3 · A fourth baseline, B4 (section 7; A-15)
- **Adds:** B4, ŷ(t+1) = the mean of the training targets at the same hour of the week as t+1. That is 168 values, computed from training targets only.
- **Why:**
  - The generator repeats exactly every 168 hours, so the weekly average is the obvious simple baseline (§17: "Another simple baseline may be used if justified").
  - The design review's closed-form estimate puts B4 about 23% below B1 on all-hours MAE (about 15.8 against 20.3). That is better than H6's whole predicted range for the attention model.
- **Expectations stated now, before any warehouse result:**
  - A-15's rule (best baseline on validation) will probably select **B4** as the reference.
  - **H6 will then probably be refuted.** The 24-hour window holds one observation per hour of the week, while B4 averages 36.
  - H5 is unchanged, because it concerns B1–B3 only.
  - H6 and the A-27 verdict are scored as written, against whichever reference validation selects.
