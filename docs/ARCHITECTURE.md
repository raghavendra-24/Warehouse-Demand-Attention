# Architecture

This document defines the technical architecture of the project. It follows the PRD (*From First Principles: Warehouse Demand Attention Model*, v1.0), the assumptions in [ASSUMPTIONS.md](ASSUMPTIONS.md) (A-01 … A-27) and the Phase 0 design in [PHASE0.md](PHASE0.md), which holds the pre-registered hypotheses (H1–H11, V1–V2) and failure modes (F1–F5). FAC-xx IDs refer to [ACCEPTANCE_CRITERIA.md](ACCEPTANCE_CRITERIA.md). It contains no implementation code. Function and file names are indicative.

Drafted with AI assistance; see [AI_LOG.md](AI_LOG.md).

**Design rule:** the simplest structure that meets every PRD requirement. The PRD calls the solution "intentionally small" (§2.7) and warns against "unrelated infrastructure" (§27). There are no plugin systems, no class hierarchies, no CLI framework, no experiment tracker and no web interface. A helper exists only where at least two scripts need it.

---

## 1. Overview

The project is a small Python package (`wda`, for warehouse demand attention) used by a handful of experiment scripts. Each script runs one PRD stage and writes plain result files. The documents in `docs/` report those files.

```
 wda/config.py ──► experiments/<stage>.py ──► results/<stage>/ ──► docs/*.md
 (all values,          │ uses                  (JSON, CSV, PNG,     (written by hand
  all seeds)           ▼                        Markdown tables,     from the result
   wda: attention · models · warehouse_data · toy_data            weights)              files)
        baselines · train · metrics · gradcheck · run
                       ▲
          tests/ (pytest) check the same modules
```

All data is synthetic, generated from `config.py` and a seed whenever it is needed. Nothing reads a network, a database or an external file.

## 2. Repository layout and run convention

```
README.md                  see Section 2.1
requirements.txt           exact version pins (A-01)
wda/                       the package
  config.py                every parameter and seed (frozen dataclasses); the fixed tiny attention example
  attention.py             the one attention implementation (§7)
  models.py                toy model and warehouse model, both calling attention.py
  warehouse_data.py        generator, windows, split, standardisation, error groups
  toy_data.py              associative-recall generator
  baselines.py             last observation, 24-hour mean, seasonal naive
  train.py                 one training loop for every experiment
  metrics.py               scoring and diagnostics
  gradcheck.py             autograd vs central finite differences
  run.py                   start/finish of every script; weight save/load
experiments/               one script per stage, plus run_all
  generate_data  attention_trace  gradcheck  stability  toy  ablation
  warehouse  shift  failure  run_all
prereg/                    the pre-implementation calculation quoted in H2, H3 and A-19, with its output
tests/                     pytest suite
results/                   committed outputs, one folder per script
docs/                      PHASE0, DERIVATION, RESULTS, REFLECTION, AI_LOG, DEMO, DEBUGGING, ASSUMPTIONS,
                           ARCHITECTURE, IMPLEMENTATION_PLAN, ACCEPTANCE_CRITERIA
```

**Run convention.** Every command runs from the repository root: `python -m pytest` for the tests and `python -m experiments.<stage>` for each stage. `python -m` puts the root on the import path, so `wda` is found without an install step or a packaging file. Plain `pytest` and `python experiments/<stage>.py` would not find it, which is why the README gives only the `-m` forms.

### 2.1 README contents

Purpose; setup (Python 3.12, pinned install, the PyTorch CPU wheel index); the test command and a map from the five §23 minimum tests to test names; per-stage and `run_all` commands; a seed table giving each seed's purpose; headline results; the §29 deliverables map; key decisions (pointing to ASSUMPTIONS and PHASE0); the reproducibility statement (exact: generated data; within the measured tolerance: trained metrics); measured runtime and hardware; **Known issues and limitations** (including the calendar-feature confound and any skipped test); what was deliberately not built (§27, §28) and why; the code search that shows one attention definition with two call sites (FAC-47).

## 3. Components and responsibilities

| Component | Responsibility | PRD / assumptions |
|---|---|---|
| `config.py` | Holds every tunable value and seed as frozen dataclasses: warehouse data, the four shift series, toy task, model sizes, training, seed lists, the `FINAL_TEST` switch (Section 10), and the fixed tiny attention example shared by the trace script, the FAC-14 test and the derivation. It is the committed configuration file A-24 refers to. | §22, A-16, A-24 |
| `attention.py` | The only attention implementation: a hand-written stable softmax, and a function that computes Q, K, V, S, S′, A and Y as separate, named statements. Two switches: `scaled` (on by default) and `uniform`. Always returns all intermediates. Also holds the W_Q, W_K, W_V parameter container with the unit-variance initialisation of A-19. | §7, §8, §11, A-02, A-19, A-26 |
| `models.py` | **Toy model:** tokens → attention → readout at the query position → linear head over values. **Warehouse model:** 5 features per hour → linear input projection → attention → readout at the last position → linear head → ŷ(t+1). Both pass the switches through. | §12, §15, §16, A-05, A-06, A-26, A-27 |
| `warehouse_data.py` | Generates the hourly series, with components stored separately, negative-binomial counts and events on their own random stream. Builds 24-hour windows with their event and calendar fields. Assigns windows to splits by target week, and standardises with given constants. | §4, §6, §18, A-08 … A-13, A-20 |
| `toy_data.py` | Associative-recall sequences: one token per key–value pair plus a query token, distinct values per sequence, separate seeds for train, validation and test. | §12, A-18 |
| `baselines.py` | Three pure functions on raw demand windows: y(t), the mean of the 24 hours, and y(t−23). Plus B4, the mean of the training targets at the same hour of the week (PHASE0 Amendment 3), whose 168-value profile is computed from training targets only. | §17, A-15 |
| `train.py` | One loop for every run: Adam, seeded batch order, a fixed step budget, and an evaluation on validation every k steps. Each evaluation records step, training loss, validation loss, validation metric, row entropy, maximum weight, ‖∂L/∂W_Q‖ and ‖∂L/∂W_K‖. A finite-loss check stops the run. Returns the history and the best-validation weights, which is early stopping on validation (A-12). | §12, §13, A-12, A-19 |
| `metrics.py` | MAE, RMSE, mean signed error (bias) and count, overall and per A-11 group; accuracy; steps to a threshold; row entropy as `torch.special.entr(A).sum(-1)`, also as a fraction of ln n; maximum weight. **`score(models, windows, mean, sd)`** returns per-window predictions for every seed, control and baseline, and the MAE/RMSE table, all in orders per hour. It is used by `warehouse.py`, `shift.py` and `failure.py`, so all three use the same metric code (FAC-35). | §17, §18, A-11, A-14 |
| `gradcheck.py` | Compares autograd with float64 central differences (h = 1e-6) for every entry of the given tensors. Returns one row per entry with an `agrees` column from `torch.isclose(g_auto, g_num, rtol=1e-6, atol=1e-8)`, which is exactly A-17's rule. | §9, A-17 |
| `run.py` | **`start(stage)`** sets the seeds, a fixed thread count and deterministic algorithms, empties `results/<stage>/` (so a failed rerun cannot leave old files looking current) and starts a timer. **`finish(stage, metrics, table)`** writes `metrics.json` (resolved configs and results only), `run.json` (runtime, versions, CPU, git revision, `FINAL_TEST`) and `table.md`, and prints the table. It also saves and loads weights. | §22, A-24, A-25 |
| `experiments/*.py` | One script per stage (Section 8). Each calls `start`, runs, and calls `finish`. Plotting stays inside the script that draws each figure, because no figure is drawn twice. | §3, §22, §30 |
| `prereg/` | The NumPy calculation of initial entropy and per-row ‖∂L/∂q‖ quoted in H2, H3 and A-19, and its saved output. It is labelled "a calculation of initial statistics, not an experiment result" and committed with the hypotheses. | FAC-52, A-19 |
| `tests/` | The §23 minimum tests plus targeted bug-finding tests (Section 11). | §23 |
| `docs/` | PHASE0, DERIVATION, RESULTS, REFLECTION, AI_LOG, DEMO, ASSUMPTIONS, ARCHITECTURE. Numbers are copied from result files. | §5, §8, §25, §26, §29 |

## 4. Data model

Everything is in-memory NumPy arrays or PyTorch tensors, plus the files in Section 4.6. There is no database.

### 4.1 Configuration (`config.py`)

| Entity | Main fields | Notes |
|---|---|---|
| `WarehouseDataConfig` | weeks, start day, base level, daily-profile and weekday parameters, trend (none), negative-binomial dispersion r, event onset rates, spike and drop multiplier ranges, duration ranges, seed | Values in PHASE0 item 2 |
| Shift series (four) | `dataclasses.replace(base, …)` with weeks = 52 (PHASE0 Amendment 2), the calendar starting at the test period's first Monday, a 24-hour warm-up, the **shift seed** (different from the data seed), and, per condition, r and/or the spike multiplier range | Control, higher noise, larger spikes, both (A-20). No separate type. |
| `ToyDataConfig` | pairs, keys, values, train/val/test sizes and seeds | A-18 |
| `ModelConfig` | d_model, d_k, d_v, initialisation scale, `scaled`, `uniform` | One per variant |
| `TrainConfig` | learning rate, batch size, step budget, evaluation interval, training seeds (three), thread count | A-16, A-19 |
| `FINAL_TEST` | boolean, `False` until the frozen-configuration commit | Section 10 |

### 4.2 Warehouse series (one row per hour)

| Field | Type | Meaning |
|---|---|---|
| t, hour_of_day, day_of_week, week | int | hour index and synthetic calendar (A-08) |
| daily, weekly, event | float | the generator's components, stored separately |
| level | float | expected demand λ = max(floor, base + daily + weekly + event) (A-09) |
| demand | int ≥ 0 | negative-binomial draw with mean λ and dispersion r |
| event_label | int | −1 drop, 0 none, +1 spike (analysis only, A-11) |

The negative binomial has variance λ + λ²/r. NumPy's `negative_binomial(n, p)` gives this with n = r and p = r / (r + λ); it accepts a non-integer n. Events and noise use two child streams of one seed (`SeedSequence.spawn`). Each event draws its onset, duration and a uniform base multiplier from the event stream, and a shift condition only rescales that drawn multiplier. The number of draws is therefore the same in every condition, and the four shift series share one event timeline. Noise draws are not paired (A-20).

### 4.3 Windows (one row per target hour)

| Field | Shape / type | Meaning |
|---|---|---|
| target_index | int | τ = t+1 |
| features | 24 × 5, float32 | standardised demand and sin/cos of hour-of-day and day-of-week, for y(t−23) … y(t) (A-05) |
| raw_window | 24, float | unstandardised demand, used by the baselines |
| target | float | y(t+1) in orders per hour |
| split | train / val / test | by the target's week: 1–36, 37–44, 45–52 (A-12) |
| group | inside / after / other | target inside an event, within 24 h after one, or neither (A-11) |
| event_sign, hours_since_onset | int | the sign of the event the group refers to, and hours since its onset (−1 if none). Used for H10, F1 and F2. |
| target_hour, target_day | int | calendar of the target. Used for F3 and F5. |

The first 24 hours produce no target. A shift series has 52 weeks plus the 24-hour warm-up (PHASE0 Amendment 2), so it yields 8,736 targets. Only the demand feature is standardised: mean and standard deviation of the training targets, computed once by `warehouse.py` and saved with the weights (A-13). The sin/cos features are already bounded.

### 4.4 Toy sequences

| Field | Shape | Meaning |
|---|---|---|
| tokens | (n_pairs + 1) × (n_keys + n_values + 1) | one-hot key, one-hot value and a query flag. The query token has a key and the flag, with no value. |
| target | int | index of the value paired with the query key |

### 4.5 Tensor shapes in the attention core

| Tensor | Shape (batch B) | Toy | Warehouse |
|---|---|---|---|
| X | B × n × d_model | n = n_pairs + 1, d_model = n_keys + n_values + 1 | n = 24, after the 5 → d_model projection |
| Q, K | B × n × d_k | | |
| V | B × n × d_v | | |
| S = QKᵀ, S′, A | B × n × n | | 24 × 24 |
| Y | B × n × d_v | | |
| readout | B × d_v | query row | last row (hour t) |

The actual sizes are in PHASE0 and go into the derivation's shape table (FAC-18).

### 4.6 Result files (committed under `results/<stage>/`)

| File | Content |
|---|---|
| `metrics.json` | Resolved config (one per arm in multi-arm stages, so arms can be diffed, FAC-28), seeds, per-seed values, mean ± SD, and the training histories. Only reproducible content, so a rerun on the same machine reproduces it exactly. |
| `run.json` | Runtime, Python/PyTorch/NumPy versions, CPU, git revision and `FINAL_TEST`: what legitimately changes between runs. |
| `metrics.json` (warehouse) | Also holds the readout attention row averaged over test windows, per seed and per group (H8, FAC-33). |
| `table.md` | The summary table as printed. Warehouse and shift tables have rows MAE and RMSE, and columns Reference baseline / Attention model (mean ± SD, every seed shown) / Uniform control / Difference = model − baseline (orders/h and %, negative means better) (A-15, FAC-35). |
| `*.png` | Figures (Section 4.7) |
| `warehouse/predictions_<split>.csv` | Per window: target index, target, `pred_s{s}` and `ctrl_s{s}` per seed, B1, B2, B3, B4, group, event_sign, hours_since_onset, target_hour, target_day. Validation only while `FINAL_TEST` is off; test after. |
| `warehouse/weights_pred_s{s}.pt`, `weights_ctrl_s{s}.pt` | Trained weights, plus μ, s, the B4 profile, the model config and the seed; `wda.models.load_warehouse_models` rebuilds the models from them |
| `shift/predictions_<condition>.csv` | As above, per condition |
| `trace/table.md` | The seven intermediates for the fixed tiny example |
| `failure/failure_modes.md` | Error slices for F1–F5 and the chosen case |

All files are kilobytes to a few hundred kilobytes, so all are committed. That meets FAC-52: every reported number comes from a committed file. No dataset file is written, because the data is regenerated from config (D3).

### 4.7 Figures

1. `generate_data`: one week of demand, and one event.
2. `toy`: loss and accuracy curves, model and control.
3. `ablation`: loss, entropy and gradient-norm curves per arm.
4. `warehouse`: one test week of truth against the model and B1–B4, and the FAC-33 attention figure (average readout row plus a typical, an event and a worst-error window).
5. `shift`: MAE by condition.
6. `failure`: the chosen case.

## 5. API boundaries

The boundaries are Python function calls inside one process. There is no network API.

| Boundary | Inputs | Outputs | Contract |
|---|---|---|---|
| `attention(X, W_Q, W_K, W_V, scaled=True, uniform=False)` | X with shape (…, n, d_model) | named tuple (Y, A, Q, K, V, S, S′) | Rows of A sum to 1. `uniform` sets A = 1/n, so Y is the mean of V. Works in float32 and float64. Callers keep the intermediates they need; H3 uses Q's gradient. |
| `stable_softmax(S)` | any tensor | same shape | over the last axis, max-subtracted |
| `generate_series(cfg)` | a data config (base or shift) | the arrays of Section 4.2 | a pure function of the config, including its seed |
| `make_windows(series)`, `split(windows)`, `standardise(x, mean, sd)` | | the Section 4.3 rows | mean and sd come from the training targets, computed once in `warehouse.py` |
| `ToyModel(cfg)`, `WarehouseModel(cfg)` | config | `forward(batch)` gives logits or ŷ, plus the attention tuple | Parameters are created from a seeded generator, independently of `scaled` and `uniform`, so paired arms start identical (FAC-28). |
| `fit(model, train, val, loss, cfg)` | | per-evaluation history and best-validation weights | stops with an error on a non-finite loss |
| `baselines.*(raw_windows)` | (N, 24) | (N,) | pure functions |
| `score(models, windows, mean, sd)` | trained models, windows, standardisation | per-window predictions and the metric table | inverse-standardises before any metric (A-13) |
| `gradcheck(fn, tensors, h)` | a scalar function and float64 tensors | per-entry table with `agrees` | A-17 |
| Script ↔ script | | | Only through `results/` files. `shift.py` and `failure.py` read what `warehouse.py` wrote, and stop with a clear message if it is missing. |

## 6. Frontend and backend

Not applicable. The PRD lists a "frontend application" and "production deployment" as not required (§27), and allows the runner to be "a Python CLI, notebook, or small application" (§3). The only interfaces are the scripts' console output, the result files and figures, and the Markdown documents.

## 7. External integrations

| Dependency | Use | Rule |
|---|---|---|
| Python 3.12 | runtime | A-01 |
| PyTorch (CPU build) | tensors, autograd, `nn.Module` / `nn.Parameter` containers, `nn.Linear` outside the attention core, Adam, standard losses, `torch.isclose`, `torch.special.entr`, `torch.save` / `torch.load(weights_only=True)` | never `nn.MultiheadAttention`, `F.scaled_dot_product_attention`, `nn.Transformer*`, or a library softmax inside `attention.py` (A-02) |
| NumPy | data generation (`Generator`, `SeedSequence`) | pinned, because random streams depend on the version |
| matplotlib | figures | |
| pytest | tests | |
| Git | the history is the evidence that hypotheses came first (A-22) | no squashing |

All versions are pinned in `requirements.txt`. PyTorch is installed from the official CPU wheel index, and the README gives the command. Nothing needs network access at run time (A-25). No other services, datasets or APIs are used. There is no pandas: the standard library and NumPy handle the CSV files.

## 8. Data flows

`run_all.py` runs the stages in this order; `finish` records each one's runtime.

| # | Script | Reads | Does | Writes | PRD / IDs |
|---|---|---|---|---|---|
| 1 | `generate_data` | config | Generates the base series and prints: mean; autocorrelation at lags 1, 24 and 168; noise SD at the mean level, √(λ + λ²/r), in orders/hour; daily amplitude ÷ that SD; event onsets against the configured rates; share of hours inside an event; realised spike and drop multipliers (minimum, median, maximum). Checks these against the PHASE0 design targets. Plots one week and one event. Rerunning it is the reset/regeneration mechanism (§4). | JSON, PNG | §4, §6, FAC-10 |
| 2 | `attention_trace` | config | Prints Q, K, V, S, S′, A and Y for the fixed tiny example (demo part 1) | `trace/table.md` | §7, §30 |
| 3 | `gradcheck` | config | A-17 comparison in float64 for X, W_Q, W_K and W_V, with loss Σ(R ⊙ Y) | per-entry table; per-tensor max absolute and relative difference and number disagreeing; a final "Unexplained entries: N" line | §9, V1, FAC-20 |
| 4 | `stability` | — | Naive vs stable float32 softmax on [1000, 999, 0], and on maximum logits 80, 85, 88, 89, 90 and 100: whether naive is finite, and the largest difference where it is | table | §11, V2, FAC-21 |
| 5 | `toy` | config | Trains scaled attention and the uniform control, three seeds each, with the same step budget | metrics, PNG | §12, H1 |
| 6 | `ablation` | config | Runs {scaled, unscaled} × d_k {4, 64} × 3 paired seeds. **At step 0, before any update**, it logs on one fixed evaluation batch: row entropy ÷ ln n, logit SD, maximum weight, and ‖∂L/∂q‖ of the **readout row** for every sequence (H2, H3). Only the readout row reaches the loss, so the other rows' query gradients are exactly zero and are not counted (PHASE0 §6). During training it logs the `fit` diagnostics (FAC-27, FAC-30). It reports steps to 95% validation accuracy (H4). | metrics, PNG | §13, §14, H2–H4 |
| 7 | `warehouse` | config | Builds the windows and splits, computes the standardisation and trains the model and the uniform control (three seeds each). Picks the reference baseline on validation. With `FINAL_TEST` off it scores **validation only**. With it on, it also scores test once, overall and by group, and keeps the validation outputs. Saves weights (with the model config and μ, s), predictions, the averaged attention rows and the figures. Shift needs none of the test outputs, so it is built and checked before `FINAL_TEST`. | metrics, CSV, weights, PNG | §15–§18, H5–H8, A-27 |
| 8 | `shift` | step 7 weights (which carry the model config and μ, s) and the reference baseline chosen on validation | Generates the four 52-week shift series and scores the saved models and the baselines (B1–B4) on each, with no retraining. Reports MAE for the first 3 hours after each spike onset (H10), and the §19 verdict under PHASE0 Amendment 1, with the original relative rule's outcome alongside. | metrics, CSV | §19, H9–H11 |
| 9 | `failure` | step 7 and 8 predictions and weights; data and shift configs | Writes the F1–F5 slices: onset hours, the 24 hours after an event, the first hours of Saturday and Monday, and windows where t and t−23 fall on different day types. Investigates the case linked to a failure mode with a targeted dose-response experiment: each spike's excess over its no-event level is scaled by k = 0 … 16 in the same windows (k = 0 removes it), recording the forecast, the attention mass on the spike tokens and their value level. | `failure_modes.md`, metrics, PNG | §20, F1–F5 |

The saturation analysis for FAC-22 reuses the step-6 logs of the unscaled d_k = 64 arm. There is no separate logit-scale sweep: a scope decision to keep the run budget small.

Main warehouse path:

```
config + seed → generate_series → make_windows → split by target week → mean, sd from training targets
   → fit (fixed budget, keep best validation) → [FINAL_TEST] score test once with score()
   → baselines on the same target indices → metrics by group → result files → RESULTS.md
```

## 9. Error-handling strategy

This is research code run by one person or one reviewer, so it fails fast and loudly. Nothing is retried or silently skipped. There is no extra defensive layer: the libraries already raise on shape mismatches and invalid sampler parameters, and the tests in Section 11 cover the invariants.

| Situation | Handling |
|---|---|
| Non-finite loss during training | `fit` stops with the step number and the last finite loss. It never continues on NaN. **Exception, the ablation:** a diverging arm is a legitimate H4 outcome, so `ablation.py` records it as "diverged at step N" and the other arms continue. |
| A script's input is missing (for example, shift before warehouse, or test predictions before `FINAL_TEST`) | Stop with a message naming what to run first |
| A run fails partway | `start` emptied the stage folder and `finish` never ran, so no result file exists for that stage. `run_all.py` stops at the first failing stage and names it. |
| Gradient entries that disagree | Listed in the table, with "Unexplained entries: N". The test fails. The investigation goes in RESULTS (§9 "Uninvestigated discrepancies are not [acceptable]"). |
| A hypothesis is refuted, or attention loses to a baseline | Not an error. It is a result, reported as such (§2.4, §2.5). |

## 10. Validation strategy

| Layer | What is checked | Where |
|---|---|---|
| Invariants | Window alignment, split boundaries, train-only standardisation, non-negative counts, shared event labels across shift series | tests (Section 11) |
| Design targets | lag-1/24/168 autocorrelation, noise SD, amplitude ratio, event share, spike multipliers against PHASE0's design targets | `generate_data` output, quoted in PHASE0 amendments and RESULTS |
| Test-set discipline | `FINAL_TEST` is off during development, so test metrics and test predictions cannot be produced. Switching it on is a commit of its own, made after the configuration is frozen. The test is then scored once (A-12, FAC-32). | `config.py`, Git history |
| Pre-registration | PHASE0, with the hypotheses, committed before any result (A-22, FAC-06). Later changes are dated amendments. | Git history |

## 11. Testing strategy

One command, `python -m pytest`, finishing in under a minute on CPU. Each test is named after the bug it targets.

| Test file | Tests | Bug it catches | Source |
|---|---|---|---|
| `test_attention.py` | Shapes with n, d_model, d_k and d_v all different, batched and unbatched | swapped dimensions, a missing transpose | §23, FAC-54 |
| | Softmax rows sum to 1 and entries are ≥ 0, on random non-constant scores | softmax over the wrong axis | §23, FAC-55 |
| | Stable softmax is finite on [1000, 999, 0] and matches naive softmax on moderate logits | overflow | §11, FAC-21 |
| | Forward pass on the fixed tiny example matches values computed by hand in float64; removing 1/√d_k makes it fail | KQᵀ instead of QKᵀ, a missing scale | FAC-14 |
| | Permuting the rows of X permutes the rows of Y the same way | position leaking into the core | FAC-58 |
| | Unscaled attention equals scaled attention with W_Q·√d_k; uniform gives the mean of V | a switch that changes more than intended | A-19, A-27 |
| `test_gradients.py` | Every entry passes `torch.isclose(rtol=1e-6, atol=1e-8)` in float64 | a broken autograd graph (detach, in-place) | §23, A-17 |
| | One backward pass gives finite, non-zero W_Q, W_K, W_V gradients, and all three change after one step | parameters not registered or not trained | FAC-15 |
| `test_data.py` | Same seed gives an identical series; a different seed gives a different one | hidden global randomness | §23, FAC-57 |
| | Window alignment; no training target in a later split | off-by-one, leakage | FAC-31, FAC-37 |
| | Standardisation uses training statistics; demand ≥ 0; the four shift series share event labels | leakage, invalid counts, unpaired shift | A-09, A-13, A-20 |
| | Baselines on the ramp y(t) = t predict 23, 11.5 and 0 for y(24) | wrong lag or window | FAC-34 |
| `test_models.py` | Each model returns one output per window or sequence | wrong readout or pooling | FAC-54 |
| `test_prohibited_api.py` | No prohibited attention API in `wda/` or `experiments/`; no `torch.softmax`, `F.softmax`, `nn.Softmax` or `.softmax(` in `wda/attention.py` | a hidden implementation | §7, FAC-13 |

**On the gradient criterion.** A pure relative-error test fails on near-zero gradient entries even when the implementation is correct, so the tests and FAC-20/FAC-56 use A-17's combined absolute-and-relative rule. The table still reports the maximum relative difference.

**Pre-submission checks** (manual, done once, results recorded in the README):
1. Fresh clone → install from `requirements.txt` → `python -m pytest` → `python -m experiments.generate_data`. `git diff --exit-code -- results/data ':!results/data/run.json'` must be empty, which shows exact reproduction.
2. Rerun `experiments.warehouse` once. The largest change in any `metrics.json` value, read from `git diff results/warehouse`, is the stated tolerance (A-24).
3. Confirm the assessors can open the repository (FAC-67).

Not included: coverage targets, mutation testing, and automated end-to-end runs. `run_all.py` runs the experiments, and the pre-submission checks cover reproduction.

## 12. Security considerations

The attack surface is small: local, offline, synthetic data, no service and no users.

| Concern | Measure |
|---|---|
| Real or personal data | None is used or accepted (§3). The generator is the only data source. |
| Untrusted file loading | Weights load with `torch.load(weights_only=True)`, so no arbitrary objects are unpickled. Results are JSON and CSV, never pickles. |
| Dependency supply chain | Exact pins, installed only from PyPI and the official PyTorch CPU index |
| Secrets | None are needed. `.gitignore` excludes virtual environments, caches and editor files. |
| Information in the repository | The AI log summarises excerpts and leaves out local paths and unrelated environment details (A-23). The repository stays private until the candidate sets its visibility. |

## 13. Performance considerations

| Aspect | Expectation and measure |
|---|---|
| Data size | 8,736 hours give about 8,700 windows of 24 × 5 float32, about 4 MB. Each shift series gives 8,736 targets. |
| Model size | One attention layer, d_model about 16: a few thousand parameters |
| Compute | Batched matrix products over (B, 24, 24) on CPU, with no Python loops over positions |
| Run budget | 3 toy + 3 control, 12 ablation, and 3 warehouse + 3 control runs, plus evaluation only for the shift. The target is about 30 minutes for `run_all.py`. Measured times are reported (A-25). |
| Determinism vs speed | A fixed, small thread count and deterministic algorithms make reruns repeatable on one machine, at some cost in speed, which is acceptable at this size |
| Gradient check | Float64 on tiny tensors: 45 entries, so 90 forward passes. It runs instantly and live in the demo. |
| Memory | Well under 1 GB |

If runs are slower than budgeted, the first lever is a smaller step budget, recorded as a dated PHASE0 amendment before any result. The grid stays as declared.

---

## 14. Architectural decisions

### D1 · A package plus one script per stage, run with `python -m`
- **Decision:** a small `wda` package, one script per §3 stage, and `run_all.py`, all run from the repository root with `python -m`.
- **Alternatives:** a single CLI with subcommands; notebooks; one monolithic script; an installable package with a `pyproject.toml`.
- **Why:** §3 names the stages, and FAC-50 needs a command per stage. One script per stage gives that with no argument parsing, and `python -m` finds the package with no install step. `run.start` / `run.finish` remove the setup that would otherwise repeat in every script.
- **Trade-offs:** about ten entry points instead of one, and the commands must use the `-m` form.

### D2 · Configuration as frozen dataclasses in one module
- **Decision:** every parameter and seed in `wda/config.py`. The shift series are `dataclasses.replace` copies of the base config.
- **Alternatives:** YAML or JSON files with a loader; command-line flags.
- **Why:** no parser, typed fields, and one place for the reviewer to read every §22 parameter. `config.py` is the committed configuration file A-24 refers to, and every `metrics.json` records the resolved values.
- **Trade-offs:** changing a value means editing Python, and there are no per-run overrides. Neither is needed.

### D3 · Data is a pure function of (config, seed)
- **Decision:** every script regenerates the data it needs in memory. Rerunning `generate_data` is the reset. No dataset file is written.
- **Alternatives:** generate once and store dataset files, with checksums to detect stale data.
- **Why:** generation takes well under a second, so stored files would only create a stale-data problem. Determinism is guaranteed by the FAC-57 test and the version pins.
- **Trade-offs:** reproducibility depends on the pinned NumPy version, which the README states.

### D4 · One attention function with two switches
- **Decision:** `attention.py` holds the only implementation, with `scaled` and `uniform` switches. It always returns all intermediates, and both models call it.
- **Alternatives:** separate implementations per model or experiment; a class hierarchy of attention variants; a third flag to return intermediates.
- **Why:**
  - §15 and §21 ask Problem 2 to build on Problem 1, and FAC-47 requires one definition with two call sites.
  - The gradient check and tests then cover both problems.
  - The two switches are exactly what the ablation and the controls need (A-19, A-18, A-27).
  - Returning the intermediates costs nothing, because they are computed anyway.
- **Trade-offs:** two flags beyond the PRD's formula; both default to the PRD's scaled attention.

### D5 · PyTorch with autograd; explicit operations only inside the core
- **Decision:** the attention core uses explicit tensor operations and a hand-written softmax. `nn.Module`, `nn.Parameter`, `nn.Linear` (for the input projection and the heads), Adam and the standard losses are used around it.
- **Alternatives:** NumPy with manual backpropagation; explicit operations everywhere, optimiser included.
- **Why:** §7 permits autograd, and the prohibition applies to attention and Transformer implementations (§2.1, §7). Hand-writing the rest adds no understanding the PRD asks for.
- **Trade-offs:** the boundary must be visible to a reviewer. The derivation's seven-row table (FAC-12) points to the exact lines, and a test enforces the hand-written softmax.

### D6 · One training loop with one stopping mode
- **Decision:** `train.fit` runs a fixed step budget, evaluates every k steps, logs the diagnostics for every run, and keeps the best-validation weights.
- **Alternatives:** a loop per experiment; patience-based early stopping plus a separate fixed-budget mode; a training framework such as Lightning.
- **Why:**
  - One loop guarantees the paired ablation arms train identically (FAC-28).
  - A fixed budget gives the toy control the same budget as the model (FAC-25).
  - Keeping the best-validation weights is early stopping on validation (A-12).
  - A framework would hide the optimisation details §13 asks to analyse.
- **Trade-offs:** every run pays for the diagnostics, which is negligible at this size.

### D7 · Plain committed result files
- **Decision:** JSON metrics, CSV predictions, PNG figures, Markdown tables and small weight files under `results/`, all committed.
- **Alternatives:** an experiment tracker (MLflow, Weights & Biases); not committing outputs.
- **Why:** FAC-52 needs every reported number to come from a committed file, and plain files need no service.
- **Trade-offs:** no run browser; each script prints its summary table instead.

### D8 · Hand-written documents fed by result files
- **Decision:** `docs/` is written by hand, with numbers copied from `table.md` and `metrics.json`.
- **Alternatives:** generated reports; notebooks as reports.
- **Why:** the PRD grades explanation and honesty (§26, §32), and generated text does not explain.
- **Trade-offs:** numbers copied by hand are re-checked against the result files before submission.

### D9 · Saved weights and per-window predictions
- **Decision:** `warehouse` saves its weights, the standardisation constants and per-window predictions. `shift` and `failure` read them.
- **Alternatives:** retrain inside `shift` and `failure`.
- **Why:** A-20 evaluates without retraining, and A-21 examines failures without retraining. Saving makes both literal and fast.
- **Trade-offs:** the stages depend on each other. `run_all.py`'s order and the missing-input message handle that.

### D10 · Seeding, determinism and dtypes
- **Decision:**
  - NumPy `SeedSequence` child streams for data, with events and noise separate;
  - a separate shift seed;
  - `torch.manual_seed` with a seeded generator for parameters and batch order;
  - deterministic algorithms and a fixed thread count;
  - float32 for training, float64 for the gradient check and the hand-computed forward test.
- **Alternatives:** global seeding only; no determinism settings.
- **Why:** each of these needs explicit streams:
  - paired ablation arms (FAC-28);
  - shift series that share one event timeline but not the training data (A-20);
  - repeatable reruns (A-24, FAC-51).
- **Trade-offs:** slightly slower training; bit-identity holds only on one machine with the same versions, as A-24 states.

### D11 · pytest with bug-targeted unit tests
- **Decision:** the §23 minimum plus the targeted tests in Section 11, all fast.
- **Alternatives:** unittest; automated end-to-end tests; coverage goals.
- **Why:** §23 asks for tests that detect "subtle mathematical bugs, not merely code coverage". Unit tests on small inputs do that quickly.
- **Trade-offs:** whole experiments are not tested automatically. The manual pre-submission checks cover reproduction.

### D12 · A `FINAL_TEST` switch instead of process alone
- **Decision:** test metrics can only be produced when `FINAL_TEST` is on. Turning it on is its own commit.
- **Alternatives:** score test on every run and rely on discipline; a separate evaluation script.
- **Why:** FAC-32 (a MUST) needs the test scored once after the configuration is frozen. One boolean makes that visible in the history at almost no cost.
- **Trade-offs:** one more config field, and a second run of `warehouse` at the end.

### D13 · No frontend, no notebooks
- **Decision:** console output, figures and Markdown are the only interfaces.
- **Alternatives:** a small web app or dashboard; notebooks.
- **Why:** §27 does not require a frontend, and A-24 avoids notebooks' hidden state.
- **Trade-offs:** none that matter for review; the figures cover the visual needs.

---

## 15. Traceability: PRD MUST items to components

| PRD requirement | Where it is met |
|---|---|
| Synthetic environment, reproducible, with reset (§4, §6) | `warehouse_data.py`, `generate_data`, `config.py` |
| Phase 0 with eight items, hypotheses first (§2.3, §5) | `docs/PHASE0.md`, Git history |
| First-principles attention, seven identifiable operations (§7) | `attention.py`, `attention_trace`, the derivation table |
| Derivation (§8), including the gradient method and loss (FAC-19) | `docs/DERIVATION.md` |
| Gradient verification (§9) | `gradcheck.py`, `experiments.gradcheck`, `test_gradients.py`; the table in RESULTS |
| Numerical stability (§11) | `stable_softmax`, `experiments.stability`, the ablation logs |
| Toy task (§12) and training dynamics (§13) | `toy_data.py`, `toy`, `ablation` diagnostics |
| Controlled ablation (§14) | `ablation`, the `scaled` switch |
| Warehouse model and baseline (§15–§17) | `models.py`, `baselines.py`, `warehouse` |
| Evaluation methodology (§18) | `warehouse_data.py`, `metrics.score`, `FINAL_TEST` |
| Distribution shift (§19) | `shift`, the four shift configs |
| Failure investigation (§20) | `failure`, the saved predictions |
| Cross-problem integration (§21) | the shared `attention.py`; the integration section of RESULTS |
| Reproducibility (§22) | `config.py`, `run.py`, `requirements.txt`, `run_all`, README |
| Tests (§23) | `tests/` |
| Debugging (§24), outcomes per H/V/F ID, known issues | `docs/RESULTS.md`, README |
| AI log (§25) | `docs/AI_LOG.md`: the requirements analysis and review, ASSUMPTIONS, the hypotheses and the `prereg` calculation, ARCHITECTURE, PHASE0, and each AI-generated module |
| Reflection (§26), demonstrated vs believed (FAC-64) | `docs/REFLECTION.md` |
| Submission and demo (§29, §30) | README deliverables map, `docs/DEMO.md`, `attention_trace`, `gradcheck` |
