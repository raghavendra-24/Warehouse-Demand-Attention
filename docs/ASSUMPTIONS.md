# Assumptions

This document records the assumptions made where the PRD (*From First Principles: Warehouse Demand Attention Model*, v1.0) does not explicitly specify the behaviour. Section references (§) are to the PRD.

Design details that the PRD asks the candidate to choose, such as exact generator parameters, model sizes and hyperparameters, belong in the Phase 0 design document. Where a design choice is recorded here, it is because other assumptions depend on it.

## Clarifications received

These are facts, not assumptions.

| Question | Answer from the assessment owner (1 Oct 2026) |
|---|---|
| Deadline | 3 October 2026, midnight (read as 23:59 IST) |
| When is Phase 0 submitted? | Together with the final submission |
| Submission channel | Git repository |

---

### A-01 · Language and numerical stack
- **PRD context:** §7 allows "NumPy, or raw PyTorch tensor operations" and "You may use PyTorch autograd for gradient computation". §22 asks for the "Python/framework version". §33 prescribes no "programming language or framework".
- **Ambiguity:** §33 suggests a free choice, while §7 restricts the attention implementation to NumPy or PyTorch.
- **Assumption:** Python 3.12, with pinned versions:
  - PyTorch (CPU build) for the attention mechanism and the models;
  - NumPy for data generation.
- **Reasoning:** this stack satisfies §7 and also allows autograd, which keeps manual backpropagation optional (§10). Python 3.12 is the version available in the development environment. Pinning NumPy matters because random streams depend on the version.
- **Impact on implementation:** a pinned dependency file; CPU only.
- **Alternatives considered:**
  - NumPy only: forces manual backprop, which the PRD treats as an optional bonus (§10).
  - JAX: listed as an optional extension (§28).

### A-02 · What counts as "high-level"
- **PRD context:**
  - High-level Transformer abstractions "are therefore not permitted for the core attention implementation" (§2.1).
  - "You must not use high-level attention or Transformer implementations" (§7).
  - The implementation should make the operations "identifiable in the source code" (§7).
- **Ambiguity:** whether general building blocks are allowed: linear layers, a library softmax, optimisers, loss functions.
- **Assumption:**
  - In the attention core, each step is an explicit tensor operation: the Q, K and V projections, QKᵀ, the √d_k scaling, a hand-written numerically stable softmax, and AV.
  - Optimisers, losses and parameter containers are used outside the core.
  - Ready-made attention is never used: `nn.MultiheadAttention`, `F.scaled_dot_product_attention`, `nn.Transformer*`, or third-party attention libraries.
- **Reasoning:** every operation the PRD lists stays visible (§7), and the stable-softmax investigation (§11) becomes concrete. Optimisers and losses are not attention.
- **Impact on implementation:** a test scans the package source (not `tests/` or `docs/`) for imports or calls of the prohibited APIs, and fails if any appear.
- **Alternatives considered:**
  - A library softmax: one listed step would be hidden.
  - Hand-written optimisers: no PRD basis, and they cost time.

### A-03 · Size of the attention model
- **PRD context:** "Implement a minimal scaled dot-product self-attention mechanism" (§7). A full Transformer and multi-head attention are "not required" (§27).
- **Ambiguity:** the number of heads and layers, and whether Transformer extras are expected (residual connections, normalisation, feed-forward blocks).
- **Assumption:** one self-attention layer with one head, followed by a small prediction head. No Transformer block.
- **Reasoning:** this is the minimal model that answers §1's questions. "Additional architecture or features do not automatically receive additional credit" (§2.7).
- **Impact on implementation:** a small, fast CPU model with one attention matrix per input.
- **Alternatives considered:**
  - Multi-head attention: not required (§27).
  - Stacked layers: not needed (§2.7).

### A-04 · Input window alignment
- **PRD context:** "the previous 24 hours" (§4) and "Previous 24 hourly observations" (§15). The §16 diagram shows "t-23 ... t-2 t-1" → ŷ(t), which is 23 positions. The §17 baselines predict ŷ(t+1) from y(t) … y(t−23).
- **Ambiguity:** whether there are 23 or 24 inputs, and exactly how they align with the target.
- **Assumption:**
  - The inputs are the 24 observations y(t−23) … y(t).
  - The target is y(t+1), the next hour (§4, §15).
  - §31's "next-period" is read the same way.
- **Reasoning:** this matches §4, §15 and both §17 formulas. The §16 diagram is treated as a conceptual sketch whose 23 positions are an inconsistency.
- **Impact on implementation:** a windowing test checks the alignment, and the model and the baselines use identical windows.
- **Alternatives considered:** the literal 23-position reading of §16, which contradicts "previous 24".

### A-05 · How each hour is represented to the attention layer
- **PRD context:** §16 names an "Input representation" step without defining it. Positional encoding is never mentioned, although §8, §16 and §20 refer to sequence or "historical positions".
- **Ambiguity:** whether the model sees only the 24 demand values.
- **Assumption:** each hour becomes a token containing standardised demand plus cyclic (sin/cos) hour-of-day and day-of-week features. A learned linear projection maps each token to the model width.
- **Reasoning:**
  - Self-attention is permutation-equivariant: without position or time information, it cannot tell t−1 from t−23.
  - With one number per token, the scores reduce to c·x_i·x_j (or β_i·x_j plus a constant per query, if the projections have biases), so attention is nearly degenerate.
  - Each hour of the day appears exactly once in the window, so hour-of-day also fixes each position's lag.
  - Without day-of-week, the window's level only partly reveals weekday versus weekend, and a change of day type cannot be anticipated (§4 weekly seasonality).
- **Impact on implementation:**
  - The data pipeline builds the features. The simple baselines do not use them.
  - With one sin/cos pair, the time part of each score is a smooth function of lag, which the attention plots will reflect.
  - Lags 0 and 23 (hours t and t+1) are neighbours on the hour circle, so the hour features alone barely separate "latest hour" from "same hour yesterday". Only day-of-week and demand separate them. This is recorded as a predicted failure mode.
  - The model sees calendar features that the baselines do not; A-27 accounts for that.
- **Alternatives considered:**
  - Demand only: the model cannot see order.
  - A generic positional encoding: gives order, but not the weekly pattern.

### A-06 · From attention outputs to one prediction
- **PRD context:** §16 shows "Self-attention → Aggregated context → Prediction" without saying how outputs are aggregated.
- **Ambiguity:** how 24 output vectors become one forecast.
- **Assumption:** the attention output at the most recent position (hour t) goes through a linear head to give ŷ(t+1).
- **Reasoning:** this is the simplest option. The prediction is read from the most recent position, so that row of the attention matrix is the one to inspect (§16). Following §16, it is presented as a visualisation, not a causal explanation.
- **Impact on implementation:**
  - With no residual path, ŷ is a weighted average of per-hour affine values with non-negative weights. That family includes close approximations of all three baselines (A-15), so losing to one is a finding about optimisation or generalisation, not capacity.
  - Plots show that row for typical, event and worst-error test windows.
- **Alternatives considered:**
  - Mean pooling: dilutes the signal.
  - A learned query token: adds a mechanism.
  - Flattening all outputs: bypasses the attention weighting.

### A-07 · Masking
- **PRD context:** masking is not mentioned.
- **Ambiguity:** whether a causal mask is needed.
- **Assumption:** no mask, because every input hour precedes the target and the toy task (A-18) has no future positions.
- **Reasoning:** a causal mask only blocks future positions.
- **Impact on implementation:** simpler attention code.
- **Alternatives considered:** a causal mask, which adds nothing here.

### A-08 · Size and calendar of the synthetic dataset
- **PRD context:** hourly data with daily and weekly seasonality (§4). Large datasets are "not required" (§27). No length is given.
- **Ambiguity:** how much history to generate.
- **Assumption:** 52 weeks of hourly data (8,736 hours) on a synthetic calendar that starts on a Monday, with no holidays and no daylight-saving changes.
- **Reasoning:** each split can hold whole weeks (A-12), and the data still trains in minutes on CPU.
- **Impact on implementation:** about 6,000 training windows.
- **Alternatives considered:**
  - A few weeks: the weekly pattern is barely sampled.
  - Several years: no added value, and longer runs.

### A-09 · Demand values and noise
- **PRD context:**
  - "A conceptual demand-generating process could resemble" Trend + Daily + Weekly + Event + Noise. "The exact formulation should be designed and justified by the candidate" (§4).
  - §6 asks for the noise distribution, its "approximate magnitude", and whether its variance is constant.
- **Ambiguity:** additive symmetric noise can produce negative order counts.
- **Assumption:**
  - The additive components of the §4 sketch define the expected level, floored at a small positive value.
  - Observed demand is a negative-binomial count around that level, with a dispersion parameter that sets the noise magnitude.
  - The target is the observed count.
- **Reasoning:**
  - Counts cannot be negative, and the variance grows with the level, which answers §6's question.
  - The dispersion parameter lets A-20 raise the noise without changing the level. Poisson noise cannot do that, because its variance equals the mean.
- **Impact on implementation:**
  - non-negative integer targets;
  - noise whose variance changes with the level, documented in Phase 0;
  - the true level is stored for diagnostics only, such as the irreducible noise part of MSE in normal hours. It is never a model input or target, and it is not a lower bound for MAE.
- **Alternatives considered:**
  - Gaussian noise clipped at zero: biases the mean and piles values up at 0.
  - A multiplicative model: more realistic seasonality, but it drops the additive structure the design keeps from the §4 sketch.

### A-10 · Trend
- **PRD context:** §6 asks for "trend behavior, if present".
- **Ambiguity:** whether to include a trend.
- **Assumption:** no long-term trend in the base environment.
- **Reasoning:** with a trend, the test period would sit outside the training range, mixing an uncontrolled shift into every result. A stationary base leaves A-20 as the only deliberate change.
- **Impact on implementation:** simpler normalisation and cleaner comparisons.
- **Alternatives considered:** a mild linear trend, which is realistic but confounds the test results.

### A-11 · Unusual events
- **PRD context:** "occasional demand spikes", "occasional unusual demand drops" and "optional operational events" (§4); "Define how demand spikes or drops are generated" (§6).
- **Ambiguity:** how often events happen, how long they last, and whether the model is told about them.
- **Assumption:**
  - Spikes and drops start at random times at a fixed rate and last a few hours.
  - They are not model inputs. Event labels are used only to group errors, never for training, early stopping or model selection.
  - No separate operational events.
- **Reasoning:** unannounced events are the realistic case, and they produce a genuine, explainable failure mode (§20).
- **Impact on implementation:** errors are reported in three groups:
  1. target inside an event;
  2. target within 24 hours after an event, where the event is still in the window and reappears through y(t−23);
  3. all other hours.
- **Alternatives considered:** scheduled events with an input flag, which makes events predictable and changes the task.

### A-12 · Train / validation / test split
- **PRD context:** §18 requires a training, a validation and a test set, says to "avoid random splitting that leaks future information", and states that "The candidate must explain the split strategy". No proportions are given.
- **Ambiguity:** the proportions, which split a boundary window belongs to, and what the validation set decides.
- **Assumption:**
  - A chronological split in whole weeks: weeks 1–36 training, 37–44 validation, 45–52 test (about 69/15/15).
  - A window belongs to the split that contains its target hour. Its inputs may come from the previous split.
  - Validation alone drives early stopping, hyperparameters and A-15's choice of reference baseline.
  - The test split is scored only once the configuration is frozen.
- **Reasoning:**
  - Whole weeks keep the mix of weekdays equal across splits.
  - Earlier hours are genuinely past data, so no future information leaks.
  - Keeping the test set untouched protects the hypotheses-before-results discipline (A-22).
- **Impact on implementation:** a test checks that no training target falls in a later split.
- **Alternatives considered:**
  - A random split: §18 says to "avoid random splitting that leaks future information".
  - A split by hour count: cuts through weeks.
  - Rolling-origin evaluation: more robust, but too costly here.

### A-13 · Normalisation
- **PRD context:** not mentioned.
- **Ambiguity:** whether inputs are scaled, how, and on which data.
- **Assumption:** standardisation statistics come from the training split only. They are applied unchanged to the validation, test and shifted data. Predictions are converted back to orders per hour before scoring.
- **Reasoning:** fitting on later data leaks information, and re-fitting on shifted data would hide the shift being tested (§19).
- **Impact on implementation:** metrics are reported on the original scale.
- **Alternatives considered:**
  - No scaling: unstable training.
  - Re-fitting per dataset: leaks information, or adapts to the shift.

### A-14 · Metrics and training loss
- **PRD context:** "Metrics should be defined before experiments". MAE, MSE, RMSE or "another justified forecasting metric" are listed, and "The candidate should explain why the selected metric is appropriate" (§18). The training loss is not specified.
- **Ambiguity:** which metric is primary, and what the model is trained on.
- **Assumption:**
  - MAE in orders per hour is the primary metric and RMSE is secondary. Both are fixed in Phase 0 and reported in the A-11 groups.
  - Models are trained with MSE.
  - MAPE is not used.
- **Reasoning:**
  - MAE is easy to interpret and robust to spikes; RMSE shows how badly spikes are missed.
  - MSE gives smooth gradients. It estimates the conditional mean, while MAE is minimised by the median. Unannounced spikes skew demand upward, so a small upward bias in normal hours is expected and will be reported.
  - MAPE is unstable when night-time demand is near zero.
- **Impact on implementation:** one comparison-table format for every experiment.
- **Alternatives considered:**
  - Training with MAE: matches the metric, but its gradients are less smooth.
  - RMSE as the primary metric, or MAPE.

### A-15 · Baselines
- **PRD context:** at least one simple baseline is required, with last observation and a 24-hour moving average as examples. "Another simple baseline may be used if justified", and baselines are evaluated on "the same test set" (§17).
- **Ambiguity:** how many baselines, and which one is the reference.
- **Assumption:**
  - Report both PRD baselines.
  - Add seasonal naive, ŷ(t+1) = y(t−23): the same hour yesterday, which is the oldest value in the window.
  - *Amended 2 October 2026 (PHASE0 Amendment 3):* add B4, the mean of the training targets at the same hour of the week. It will probably be the reference.
  - The best baseline on validation is the reference for judging whether attention adds value.
  - "Difference" (§17) = model − baseline, in orders per hour and as a relative change in %. Negative means the model is better.
- **Reasoning:** the 24-hour average spans exactly one daily cycle, so it averages the daily pattern away. Seasonal naive is the natural strong baseline for hourly data, and costs one line.
- **Impact on implementation:** three baselines scored on identical test windows.
- **Alternatives considered:**
  - A single PRD baseline: too easy to beat.
  - A linear autoregressive model: stronger, but not planned in two days.

### A-16 · Random seeds and repeated runs
- **PRD context:** random seeds must be documented, and "Where randomness is intentionally used, explain its purpose" (§22). The number of repeated runs is not specified.
- **Ambiguity:** whether one training run is enough evidence.
- **Assumption:**
  - Fixed, documented seeds for every generated dataset: warehouse, toy task, and the A-20 series.
  - Three training seeds for every reported model configuration, reported as mean ± standard deviation.
  - An effect is claimed only if it has the same sign in all three seeds, paired by seed where both arms share one. Otherwise it is reported as inconclusive. Every run's value is shown.
- **Reasoning:** small models vary between initialisations, and three seeds is the minimum that shows the spread within the time available.
- **Impact on implementation:** every headline number is a mean over three runs.
- **Alternatives considered:**
  - A single seed: cannot separate noise from effect.
  - Five or more seeds: better, but costs time.

### A-17 · Gradient verification
- **PRD context:** §9 says candidates "may use either" Option A (autograd, explaining what gradient is computed) or Option B (comparing analytical and numerical gradients), and "If the result is not exact, explain why." §23 requires a gradient-verification test, and §2.2 values "comparison against independently calculated results".
- **Ambiguity:** with Option A, what the §23 test compares against.
- **Assumption:**
  - Train with autograd (Option A).
  - Compare autograd gradients with float64 central finite differences (h = 1e-6), for every entry of X, W_Q, W_K and W_V, on small tensors with n, d_model, d_k and d_v all different.
  - The checked loss is Σ(R ⊙ Y) for a fixed random R. sum(A) cannot be used: each row of A sums to 1, so its gradient is zero.
  - An entry agrees if |g_auto − g_num| ≤ 1e-8 + 1e-6·|g_num|. The table reports the maximum absolute and relative differences. Any entry that fails is investigated and explained.
  - The table reports the "autograd gradient" next to the numerical one (it is not an analytical gradient).
  - The forward pass is checked separately against values computed by hand on a tiny example.
- **Reasoning:**
  - float64 avoids the round-off error that makes float32 comparisons fail.
  - Distinct sizes expose transposition bugs in the projections.
  - Both gradients come from the same forward code, so a forward bug such as KQᵀ instead of QKᵀ would pass the gradient check. That is why the forward pass needs its own check.
- **Impact on implementation:** gradient and forward-value tests in the suite (§23), plus a results table.
- **Alternatives considered:**
  - Autograd only: no independent check.
  - A library gradient checker only: correct, but hides the method.

### A-18 · Toy learning task (design choice recorded because A-19 depends on it)
- **PRD context:** "The candidate may design the task", with "associative recall" among the examples. It "must demonstrate that the attention mechanism participates in an actual learning process" (§12).
- **Ambiguity:** none in the text; the task is the candidate's choice.
- **Assumption:**
  - An associative-recall task: each token packs one key and its value, and a final query token carries a key. The model must output the matching value.
  - The model is evaluated on held-out sequences.
  - A control with the attention weights fixed to uniform (mean pooling) shows how much attention contributes. Its expected accuracy is about 1/n_pairs, guessing among the values present, not 1/vocabulary size.
- **Reasoning:** with key and value in the same token, one attention layer can solve the task by matching the query to the right key. If keys and values were separate tokens, two layers would be needed.
- **Impact on implementation:** a small task generator, accuracy and loss curves, and a control run.
- **Alternatives considered:**
  - A copy task: mostly tests position handling.
  - Separate key and value tokens: unsolvable with one layer (A-03).

### A-19 · Training dynamics and the ablation
- **PRD context:** §13 asks for analysis of at least one training-dynamics phenomenon. §14 recommends scaled vs unscaled attention as the ablation, and says "Alternative ablations are allowed only when justified". Neither section names the model to use.
- **Ambiguity:** whether to use the toy task or the warehouse model, which d_k values, and how to initialise.
- **Assumption:**
  - Both run on the toy task, before the warehouse model (§35's order).
  - Scaled vs unscaled attention at d_k = 4 and d_k = 64, three seeds each, with the optimiser and learning rate fixed in advance.
  - W_Q and W_K are initialised so that the components of Q and K have about unit variance.
  - One optimiser, Adam, with one learning rate for both variants. Because Adam partly offsets small gradients, the comparison is judged first on initial entropy and gradient statistics, then on steps to target.
  - Reported: attention entropy and gradient norms at initialisation, steps to a target accuracy, and final loss.
  - Softmax saturation in the unscaled variant is the training-dynamics phenomenon.
- **Reasoning:**
  - Unscaled attention equals scaled attention with W_Q multiplied by √d_k, so the two differ mainly in how they train.
  - With unit-variance Q and K components, unscaled scores at d_k = 64 have a standard deviation of about 8. A quick simulation before implementation (a calculation of initial statistics, not an experiment result) suggests:
    - attention starts nearly one-hot (entropy about 15% of its maximum, versus about 85% when scaled);
    - gradients become very uneven: about 9–16% of rows have almost none, while others are several times larger than when scaled.
  - The calculation and its output are committed in `prereg/init_stats.py` and `prereg/init_stats.txt`.
  - With PyTorch's default linear-layer initialisation, the score spread would be about 2.7 and the contrast weak, which is why the initialisation is fixed.
- **Impact on implementation:** 2 variants × 2 values of d_k × 3 seeds = 12 short runs.
- **Alternatives considered:**
  - The warehouse model: attention is not essential there, so the signal is noisier.
  - A single d_k: probably a null result.
  - More d_k values: a better curve, but more time.

### A-20 · Distribution shift
- **PRD context:** §19: "The model must be evaluated under a changed statistical distribution" and "The exact change is up to the candidate". §6's example trains on "moderate noise + normal demand spikes" and tests on "higher noise + larger demand spikes".
- **Ambiguity:** which changes to make, and how many.
- **Assumption:**
  - Evaluation only: no retraining, as §6 and §19 imply, and the training normalisation is reused.
  - Four series with the test period's weekly calendar, each preceded by a 24-hour warm-up. They are 52 weeks long (*amended 2 October 2026, PHASE0 Amendment 2*; originally 8 weeks). Events and count noise use separate random streams from one seed, so all four series share the same event timeline; noise draws are not paired across series:
    1. unchanged parameters, as a control;
    2. higher noise, through the dispersion parameter (A-09);
    3. larger spikes;
    4. both together, which is §6's example.
  - The baselines and all three trained warehouse models (A-16) are evaluated on the same series, reported as mean ± SD.
  - Verdict (*amended 2 October 2026, PHASE0 Amendment 1*): the model "learned useful structure" (§19) if it beats the reference on series 1 and its absolute MAE increase from series 1 to series 4 is no larger than the reference's. If the increase is larger in all three seeds, it "simply adapted" to the training distribution. Otherwise the verdict is inconclusive.
- **Reasoning:**
  - Changing one factor at a time shows which change caused the impact.
  - A shared event timeline makes event-hour comparisons line up across series. Count noise cannot be paired, because the samplers consume a parameter-dependent number of random draws.
  - The baselines show whether the data got harder or the model got worse.
- **Impact on implementation:** four small generated series and one comparison table.
- **Alternatives considered:**
  - Only the combined shift: the effects are confounded.
  - Fully independent seeds per series: event timelines differ, so event-hour comparisons do not line up.

### A-21 · What counts as a genuine failure
- **PRD context:** "Identify at least one genuine failure case". The examples include "instability under extreme inputs". Each failure needs a scenario, expected and actual behaviour, explanation, evidence ("What experiment or mathematical reasoning supports your explanation?") and a potential improvement (§20).
- **Ambiguity:** whether a failure provoked by artificial inputs counts.
- **Assumption:**
  - The failure comes from normal evaluation, such as the windows with the largest error on the test or shifted data, or from extreme inputs the generator can actually produce.
  - Injected NaNs and values outside the generator's range are excluded.
  - The explanation is supported by a targeted experiment or by mathematical reasoning.
- **Reasoning:** "genuine" rules out manufactured breakage, and §20 explicitly accepts extreme inputs and mathematical evidence.
- **Impact on implementation:** predictions are saved per window, so failures can be found and examined without retraining.
- **Alternatives considered:** injected NaNs or impossible values, which belong at most in the numerical-stability notes (§11).

### A-22 · Showing that hypotheses came before results
- **PRD context:** hypotheses must be written before the experiments (§2.3). Phase 0 "is mandatory and is graded separately", and "should reflect the actual implementation" (§5). Phase 0 is submitted with the final submission, through the Git repository.
- **Ambiguity:** how the order can be shown, and whether Phase 0 may be edited after implementation.
- **Assumption:**
  - Phase 0 and the hypotheses are committed before any experiment result.
  - They are not rewritten afterwards; changes go in as dated amendments.
  - The full Git history is kept, with no squashing.
  - Each hypothesis has an ID, a metric and comparison, the expected direction (and size where there is a principled basis), and the result that would refute it. Outcomes go in the results and reflection documents, keyed by ID, never into Phase 0.
- **Reasoning:** commit history is objective evidence of order. Amendments reconcile "design before code" with "should reflect the actual implementation" without erasing the original predictions.
- **Impact on implementation:** commit order matters from the first commit.
- **Alternatives considered:**
  - Editing Phase 0 to match the code: destroys the evidence.
  - A separate early submission: not required, per the clarification.

### A-23 · Material AI assistance
- **PRD context:** "Material AI assistance must be documented" (§2.6), with the log fields listed in §25.
- **Ambiguity:** what "material" covers.
- **Assumption:** any AI output that shaped the code, documents, design or analysis is logged, including AI-assisted analysis of the PRD and drafting of these documents. Routine autocompletion is not logged item by item. Hypotheses, failure modes and other Phase 0 content drafted with an AI tool are logged as such, with what the candidate kept, changed or rejected, and the candidate must be able to defend each one (§34).
- **Reasoning:** the evaluation focuses on whether the candidate understands and can defend the work (§25). A broad reading is the honest one.
- **Impact on implementation:** the AI log is kept up to date throughout.
- **Alternatives considered:** logging only generated code, which misses AI influence on the design and the documents.

### A-24 · Experiment runner and reproducibility
- **PRD context:**
  - The runner "may be a Python CLI, notebook, or small application" (§3).
  - "A reset/regeneration mechanism should be provided" (§4).
  - Evaluation commands, seeds and configurations are documented (§22).
- **Ambiguity:** what form the runner takes, and how exact reproduction must be.
- **Assumption:**
  - Experiments run from command-line scripts driven by configuration files.
  - One documented command regenerates the dataset and the primary results.
  - With pinned versions, generated data is bit-identical for a given seed.
  - Training on one machine with deterministic settings is repeatable. No cross-machine tolerance is promised; the README reports the difference observed between two runs.
  - Before submission, a fresh clone is installed from the pinned file, and the tests and data regeneration are run once.
- **Reasoning:** scripts avoid the hidden state of notebooks. PyTorch does not guarantee bit-identical training across platforms, so promising that would be untestable.
- **Impact on implementation:** configuration files and seeds are committed with the code.
- **Alternatives considered:**
  - Notebooks as the main runner: hidden state, harder to rerun.
  - Promising bit-identical training everywhere: the framework does not guarantee it.

### A-25 · Compute, documents and demo
- **PRD context:**
  - The system runs "locally or against synthetic data" (§3).
  - "Include all project documents in the repository" (§29).
  - The demo may be "a short recording or reproducible live-demo instructions" (§30).
- **Ambiguity:** the runtime budget, the document format and the form of the demo.
- **Assumption:**
  - The full pipeline targets about 30 minutes on a laptop CPU, and measured runtimes are reported. Once dependencies are installed, nothing needs network access at run time (§3).
  - Documents are Markdown files under `docs/`, and the README maps each §29 deliverable to its file.
  - The demo is reproducible live-demo instructions: fast steps (the attention forward pass, the gradient check) run live, and long runs are shown from saved results. A short recording is added if time allows.
  - The README lists what was deliberately not built (§27, §28) and why.
- **Reasoning:** any reviewer can rerun the work, and Markdown renders in the repository host.
- **Impact on implementation:** experiment sizes are chosen to fit the runtime budget.
- **Alternatives considered:**
  - GPU training: not needed, and adds non-determinism.
  - PDF or notebook reports: harder to diff and review.

### A-26 · Reuse between the two problems
- **PRD context:** the warehouse attention "should be based on the implementation from Problem 1. The candidate may simplify or adapt the architecture where necessary" (§15). "Problem 2 must genuinely use the understanding developed in Problem 1" (§21).
- **Ambiguity:** whether to reuse the same code or adapt a copy.
- **Assumption:** the warehouse model calls the same attention function used in Problem 1, not a copy. Its additions, the input projection (A-05) and the readout (A-06), wrap that function.
- **Reasoning:** one implementation means the gradient check and tests cover both problems, and it makes the §21 link concrete in the code.
- **Impact on implementation:** the attention function accepts warehouse-shaped inputs and two experiment switches: scaling on or off (A-19) and uniform weights (A-18, A-27). By default it computes the PRD's scaled attention.
- **Alternatives considered:** a modified copy, which would need its own verification and weakens §21.

### A-27 · Judging whether attention adds value
- **PRD context:** "Can you determine whether attention actually provides value over a simple baseline?" (§1). A model worse than the baseline is "a valid result if the candidate correctly explains why" (§2.4). §17: "Report: Baseline / Attention model / Difference".
- **Ambiguity:** whether "value" means the model beats a baseline, or that the attention weighting is what helps. The model has calendar inputs the baselines lack (A-05).
- **Assumption:**
  - The warehouse comparison ends in one verdict: adds value, does not, or inconclusive. It is judged against the A-15 reference baseline under the A-16 rule.
  - The same model is also trained with uniform attention weights (same features, head and training, three seeds).
  - Attention is credited only if the model beats both the reference baseline and this control. Otherwise the gain is attributed to the model's inputs, and the confound is stated.
- **Reasoning:** this makes §1's question answerable and separates the weighting from the inputs. It reuses A-18's switch and costs three short runs.
- **Impact on implementation:** one extra row in the warehouse table, and a verdict sentence in the results.
- **Alternatives considered:**
  - A calendar-aware baseline (the mean demand at each hour of the week): fair, but a new component.
  - Only stating the confound: honest, but leaves the central question open.
