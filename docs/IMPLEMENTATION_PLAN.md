# Implementation plan

Small, ordered tasks that build the design in [ARCHITECTURE.md](ARCHITECTURE.md). Each task lists its files, dependencies, the requirements it serves and its own acceptance criteria. Requirement IDs:
- §n is a PRD section;
- A-xx is an assumption in [ASSUMPTIONS.md](ASSUMPTIONS.md);
- H/V/F-xx is a hypothesis or failure mode in [PHASE0.md](PHASE0.md);
- FAC-xx is in [ACCEPTANCE_CRITERIA.md](ACCEPTANCE_CRITERIA.md).

Design values (sizes, rates, step budgets) come from PHASE0 and live in `wda/config.py`.

Drafted with AI assistance; see [AI_LOG.md](AI_LOG.md).

**Deadline:** 3 October 2026, 23:59 IST. **Rule throughout:** no task may produce a model result on validation or test before T-001 is committed (A-22). Tasks marked **[live]** are run in the demo.

---

## Phase 0 — Pre-registration

### T-001 · Commit Phase 0 and the design documents
- **Files:** `docs/PHASE0.md`, `docs/ASSUMPTIONS.md`, `docs/ARCHITECTURE.md`, `docs/ACCEPTANCE_CRITERIA.md`, `docs/IMPLEMENTATION_PLAN.md`, `docs/AI_LOG.md`, `prereg/*`, `.gitignore`
- **Depends on:** —
- **Requirements:** §2.3, §5; A-22; FAC-01 … FAC-06 (Phase 0 contents and commit order)
- **Acceptance criteria:**
  - `PHASE0.md` contains the hypotheses and failure modes; no separate `HYPOTHESES.md` is ever committed.
  - It is the first commit in the repository, and it is pushed.
  - It contains no code under `wda/` or `experiments/`.
- **Estimate:** 10 min

---

## Phase 1 — Project setup

### T-101 · Environment and pins
- **Files:** `requirements.txt`, `README.md` (setup section only)
- **Depends on:** T-001
- **Requirements:** §22; A-01; FAC-49
- **Acceptance criteria:**
  - A Python 3.12 virtual environment, outside Git, installs `torch` (CPU wheel index), `numpy`, `matplotlib` and `pytest` at exact versions.
  - `python -c "import torch, numpy, matplotlib"` works.
  - The README states the install command, including the CPU index URL.
- **Estimate:** 20 min

### T-102 · Package skeleton and run convention
- **Files:** `wda/__init__.py`, `experiments/` (empty package), `tests/`
- **Depends on:** T-101
- **Requirements:** FAC-50, FAC-53; ARCHITECTURE §2
- **Acceptance criteria:**
  - From the repository root, `python -m pytest` runs (no tests yet) and `python -m experiments.<name>` can import `wda`.
  - The README gives only the `-m` forms.
- **Estimate:** 10 min

### T-103 · Configuration module
- **Files:** `wda/config.py`
- **Depends on:** T-102
- **Requirements:** §22; A-16, A-20, A-24; PHASE0 items 2 and the design-values table; FAC-08, FAC-11
- **Acceptance criteria:**
  - Frozen dataclasses hold every value in PHASE0's tables: data, toy, models and training, three training seeds, a data seed and a separate shift seed.
  - The four shift series are built with `dataclasses.replace`.
  - The fixed tiny attention example has n, d_model, d_k and d_v all different.
  - `FINAL_TEST = False`.
  - Every value matches PHASE0, checked by reading both side by side.
- **Estimate:** 30 min

### T-104 · Run start/finish helper
- **Files:** `wda/run.py`
- **Depends on:** T-103
- **Requirements:** §22; A-24, A-25; FAC-49, FAC-52; ARCHITECTURE D10
- **Acceptance criteria:**
  - `start(stage)` sets the seeds, the thread count and deterministic algorithms, and empties `results/<stage>/`.
  - `finish(stage, metrics, table)` writes `metrics.json` (config, seeds, versions, CPU model, runtime) and `table.md`, and prints the table.
  - Weights save, and load with `weights_only=True`.
- **Estimate:** 30 min

---

## Phase 2 — Attention core (Problem 1)

### T-201 · Stable softmax and the attention function
- **Files:** `wda/attention.py`
- **Depends on:** T-102
- **Requirements:** §7, §11; A-02, A-19, A-26; FAC-12, FAC-13
- **Acceptance criteria:**
  - `stable_softmax` subtracts the row maximum and normalises over the last axis.
  - `attention(X, W_Q, W_K, W_V, scaled=True, uniform=False)` computes Q, K, V, S, S′, A and Y on separate named lines and returns all of them.
  - `uniform` sets A = 1/n.
  - A parameter container creates W_Q, W_K and W_V from a seeded generator with the PHASE0 initialisation scale, independently of the switches.
  - No library softmax or attention API is used.
- **Estimate:** 45 min

### T-202 · Attention tests
- **Files:** `tests/test_attention.py`, `tests/test_prohibited_api.py`
- **Depends on:** T-201, T-103
- **Requirements:** §23; FAC-13, FAC-14, FAC-21, FAC-54, FAC-55, FAC-58
- **Acceptance criteria:** tests, each named after the bug it targets, pass for:
  - shapes, batched and unbatched;
  - rows of A sum to 1 and entries ≥ 0 on random scores;
  - finite output on [1000, 999, 0];
  - the hand-computed float64 forward pass on the tiny example, with a check that removing 1/√d_k makes it fail;
  - permutation equivariance;
  - unscaled = scaled with W_Q·√d_k;
  - uniform gives the mean of V;
  - a source scan for prohibited APIs, including any library softmax in `attention.py`.
- **Estimate:** 60 min

### T-203 · Gradient check
- **Files:** `wda/gradcheck.py`, `tests/test_gradients.py`
- **Depends on:** T-201
- **Requirements:** §9, §23; A-17; V1; FAC-15, FAC-19, FAC-20, FAC-56
- **Acceptance criteria:**
  - Central differences in float64 with h = 1e-6, for every entry of X, W_Q, W_K and W_V, with loss Σ(R ⊙ Y).
  - An `agrees` column from `torch.isclose(rtol=1e-6, atol=1e-8)`.
  - The test asserts that every entry agrees.
  - A second test: one backward pass gives finite, non-zero gradients for W_Q, W_K and W_V, and all three change after one step.
- **Estimate:** 40 min

### T-204 · Trace, gradient-check and stability scripts **[live]**
- **Files:** `experiments/attention_trace.py`, `experiments/gradcheck.py`, `experiments/stability.py`
- **Depends on:** T-104, T-202, T-203
- **Requirements:** §7, §9, §11, §30 parts 1–2; V1, V2; FAC-20, FAC-21, FAC-69
- **Acceptance criteria:**
  - `results/trace/table.md` holds the seven intermediates.
  - `results/gradcheck/` holds the per-entry table, the per-tensor maxima and "Unexplained entries: N".
  - `results/stability/` shows naive vs stable float32 on [1000, 999, 0] and on maximum logits 80–100: whether naive is finite, and the largest difference.
  - Each script runs in seconds.
- **Estimate:** 40 min

**Checkpoint A:** commit and push (A-28 advice).

---

## Phase 3 — Synthetic data

### T-301 · Warehouse generator
- **Files:** `wda/warehouse_data.py` (generation part)
- **Depends on:** T-103
- **Requirements:** §4, §6; A-08, A-09, A-10, A-11, A-20; FAC-08, FAC-09
- **Acceptance criteria:**
  - `generate_series(cfg)` returns the ARCHITECTURE §4.2 arrays.
  - Negative-binomial draws use n = r and p = r/(r+λ).
  - Events come from their own `SeedSequence` child stream, and each event draws onset, duration and base multiplier in a fixed number of draws.
  - A shift condition only rescales the drawn multiplier.
- **Estimate:** 60 min

### T-302 · Windows, split, standardisation, groups
- **Files:** `wda/warehouse_data.py` (windows part)
- **Depends on:** T-301
- **Requirements:** §15, §18; A-04, A-05, A-11, A-12, A-13; FAC-31, FAC-37, FAC-38, FAC-40
- **Acceptance criteria:**
  - Each window holds 24 × 5 features, a raw window, the target y(t+1), its split by target week (1–36, 37–44, 45–52), the group (inside / after / other), event_sign, hours_since_onset, target_hour and target_day.
  - `standardise(x, mean, sd)` touches only the demand feature.
- **Estimate:** 45 min

### T-303 · Data tests
- **Files:** `tests/test_data.py`
- **Depends on:** T-302
- **Requirements:** §23; FAC-31, FAC-37, FAC-38, FAC-57; A-09, A-20
- **Acceptance criteria:** tests pass for:
  - the same seed giving identical series, and a different seed giving a different one;
  - target index = last input index + 1;
  - no training target in a later week;
  - standardisation statistics from training targets only;
  - demand ≥ 0;
  - identical event labels across the four shift series.
- **Estimate:** 40 min

### T-304 · Data generation script and design-target check
- **Files:** `experiments/generate_data.py`
- **Depends on:** T-104, T-302
- **Requirements:** §4, §6; FAC-10; PHASE0 design targets
- **Acceptance criteria:**
  - Prints and saves: mean; autocorrelation at lags 1, 24 and 168; noise SD at the mean level; amplitude ÷ SD; onsets against the configured rates; share of event hours; realised multipliers.
  - Two PNGs: one week, and one event.
  - If a design target is missed, PHASE0 gets a dated amendment **before** T-501.
- **Estimate:** 30 min

### T-305 · Toy-task generator
- **Files:** `wda/toy_data.py`, plus toy tests in `tests/test_data.py`
- **Depends on:** T-103
- **Requirements:** §12; A-18; FAC-23
- **Acceptance criteria:**
  - Each token packs a one-hot key, a one-hot value and a query flag; the query token has no value.
  - Values are distinct within a sequence.
  - Train, validation and test use separate seeds.
  - A test checks that the target is the query key's value.
- **Estimate:** 30 min

---

## Phase 4 — Models, baselines, training, metrics

### T-401 · Models
- **Files:** `wda/models.py`, `tests/test_models.py`
- **Depends on:** T-201
- **Requirements:** §12, §15, §16, §21; A-05, A-06, A-26, A-27; FAC-31, FAC-47, FAC-54
- **Acceptance criteria:**
  - The toy model reads out at the query position; the warehouse model projects 5 features to d_model and reads out at the last position.
  - Both call `attention.attention` and pass `scaled` and `uniform` through.
  - The test checks one output per window or sequence.
  - A code search shows one attention definition and two call sites.
- **Estimate:** 40 min

### T-402 · Baselines
- **Files:** `wda/baselines.py`, ramp test in `tests/test_data.py`
- **Depends on:** T-302
- **Requirements:** §17; A-15; FAC-34
- **Acceptance criteria:** on y(t) = t, the three baselines predict 23, 11.5 and 0 for y(24).
- **Estimate:** 15 min

### T-403 · Training loop
- **Files:** `wda/train.py`
- **Depends on:** T-401, T-104
- **Requirements:** §12, §13; A-12, A-19; FAC-15, FAC-25, FAC-27, FAC-28
- **Acceptance criteria:**
  - Adam, seeded batch order and a fixed step budget, with an evaluation every k steps.
  - Each evaluation records step, training loss, validation loss, validation metric, row entropy, maximum weight, ‖∂L/∂W_Q‖ and ‖∂L/∂W_K‖.
  - It keeps the best-validation weights and stops with the step number on a non-finite loss.
  - Two runs with the same seed give identical histories.
- **Estimate:** 45 min

### T-404 · Metrics and shared scoring
- **Files:** `wda/metrics.py`
- **Depends on:** T-402, T-403
- **Requirements:** §17, §18; A-11, A-14, A-15; FAC-35, FAC-38, FAC-40
- **Acceptance criteria:**
  - MAE, RMSE, mean signed error and count, overall and per group.
  - Accuracy, steps to a threshold, and entropy via `torch.special.entr`, also ÷ ln n.
  - `score(models, windows, mean, sd)` inverse-standardises, returns per-window predictions for every seed, control and baseline, and builds the table with Difference = model − baseline (orders/h and %).
- **Estimate:** 45 min

---

## Phase 5 — Experiments (run in this order)

### T-501 · Toy task **[demo part 3]**
- **Files:** `experiments/toy.py`
- **Depends on:** T-305, T-403, T-404
- **Requirements:** §12; A-18; H1; FAC-23, FAC-24, FAC-25
- **Acceptance criteria:**
  - Scaled attention and the uniform control are trained, three seeds each, with the same step budget.
  - `results/toy/` holds the loss and accuracy curves and the held-out accuracy, mean ± SD with every seed shown.
- **Estimate:** 30 min, including the run

### T-502 · Ablation and training dynamics **[demo part 5]**
- **Files:** `experiments/ablation.py`
- **Depends on:** T-501
- **Requirements:** §13, §14; A-19; H2, H3, H4; FAC-22, FAC-26 … FAC-30
- **Acceptance criteria:**
  - The grid is {scaled, unscaled} × d_k {4, 64} × 3 paired seeds; the arms share data, initial weights and batch order, and the per-arm configs differ only in `scaled`.
  - At step 0, before any update, on one fixed batch: entropy ÷ ln n, logit SD, maximum weight and the readout row's ‖∂L/∂q‖ per sequence (the other rows do not reach the loss).
  - A diverging arm is recorded as "diverged at step N", not a crash (H4).
  - Training diagnostics are logged, and steps to 95% are reported.
  - `results/ablation/` holds the curves and the table.
- **Estimate:** 45 min, including the runs

### T-503 · Warehouse model, validation only
- **Files:** `experiments/warehouse.py`
- **Depends on:** T-304, T-401, T-404
- **Requirements:** §15–§18; A-12, A-15, A-27; FAC-31, FAC-32, FAC-36
- **Acceptance criteria:**
  - With `FINAL_TEST` off, it trains the model and the uniform control (three seeds each), picks the reference baseline on validation, and writes validation metrics and `predictions_val.csv`.
  - No test metrics exist anywhere.
- **Estimate:** 45 min

### T-504 · Freeze and score the test once **[demo part 4]**
- **Files:** `wda/config.py` (`FINAL_TEST = True`), `results/warehouse/`
- **Depends on:** T-503
- **Requirements:** §17, §18; A-12, A-16, A-27; H5–H8; FAC-32, FAC-33, FAC-35, FAC-36, FAC-39, FAC-40
- **Acceptance criteria:**
  - A commit "freeze configuration" comes first, then a commit that sets `FINAL_TEST = True` and contains the single test run's results.
  - The table shows Reference baseline / Attention model / Uniform control / Difference, overall and per group.
  - The FAC-33 attention figure and the test-week forecast figure are written.
  - Weights and `predictions_test.csv` are saved.
- **Estimate:** 20 min

### T-505 · Distribution shift **[demo part 6]**
- **Files:** `experiments/shift.py`
- **Depends on:** T-503 (built and checked before the freeze; rerun after T-504)
- **Requirements:** §19; A-20; H9, H10, H11; FAC-41 … FAC-44
- **Acceptance criteria:**
  - It loads the saved weights (with their model config, μ and s) with no retraining, takes the reference baseline chosen on validation, and generates the four series with the shift seed.
  - One table covers every model and baseline, with the change from the control.
  - MAE in the first 3 hours after spike onsets is reported.
  - `predictions_<condition>.csv` files are written.
- **Estimate:** 40 min

### T-506 · Failure investigation **[demo part 7]**
- **Files:** `experiments/failure.py`
- **Depends on:** T-505
- **Requirements:** §20; A-21; F1–F5; FAC-45, FAC-46
- **Acceptance criteria:**
  - `failure_modes.md` slices the errors for F1–F5.
  - One case is chosen and linked to a failure mode, reproducible by dataset, seed and window index.
  - A targeted experiment (for example, removing the event component and predicting again) is run, with its numbers saved.
- **Estimate:** 60 min

### T-507 · Run-all
- **Files:** `experiments/run_all.py`
- **Depends on:** T-204, T-304, T-501 … T-506
- **Requirements:** §3, §22; A-24, A-25; FAC-50, FAC-51
- **Acceptance criteria:**
  - It runs every stage in order, stops at the first failure and names it, and records each stage's runtime.
  - The total runtime is recorded for the README.
- **Estimate:** 15 min

---

## Phase 6 — Testing and reproducibility

### T-601 · Full suite and test map
- **Files:** `tests/*`, README test section
- **Depends on:** T-202, T-203, T-303, T-305, T-401, T-402
- **Requirements:** §23; FAC-53, FAC-58
- **Acceptance criteria:**
  - `python -m pytest` passes in under a minute.
  - The README maps the five §23 minimum items to test names.
- **Estimate:** 15 min

### T-602 · Pre-submission checks
- **Files:** README reproducibility section
- **Depends on:** T-507
- **Requirements:** §22; A-24; FAC-51, FAC-67
- **Acceptance criteria:**
  - A fresh clone installs, passes the tests, and regenerates `results/data` with an empty `git diff` (excluding `run.json`, which holds the runtime).
  - One warehouse rerun gives the stated tolerance.
  - Assessor access is confirmed.
- **Estimate:** 30 min

---

## Phase 7 — Documentation

### T-701 · Derivation
- **Files:** `docs/DERIVATION.md`
- **Depends on:** T-201 (can be written in parallel with Phases 3–5)
- **Requirements:** §8, §9, §11; FAC-12, FAC-16 … FAC-19, FAC-21
- **Acceptance criteria:**
  - One section per §8 topic.
  - The seven-row operation → formula → source-line table.
  - The variance chain behind √d_k.
  - The symbolic shape table and a table at the actual sizes.
  - The gradient method and loss.
  - Stable softmax: softmax(s − c) = softmax(s).
  - A worked example equal to `results/trace/table.md`.
- **Estimate:** 75 min

### T-702 · Results
- **Files:** `docs/RESULTS.md`
- **Depends on:** T-204, T-502, T-504, T-505, T-506
- **Requirements:** §9, §12–§14, §17, §19–§21, §24; FAC-20, FAC-24, FAC-26, FAC-27, FAC-30, FAC-35, FAC-36, FAC-44, FAC-45, FAC-48, FAC-59
- **Acceptance criteria:**
  - Sections in this order: gradient check, toy task, training dynamics, ablation (six headings), warehouse with the A-27 verdict, shift (five §19 items and the verdict), failure (six headings), integration (§21), debugging episode (§24).
  - An outcome for every H, V and F ID.
  - Every number is copied from `results/`.
- **Estimate:** 90 min

### T-703 · Reflection
- **Files:** `docs/REFLECTION.md`
- **Depends on:** T-702
- **Requirements:** §26; FAC-63, FAC-64
- **Acceptance criteria:**
  - Eleven numbered answers, written by the candidate.
  - A "demonstrated vs believed" table with one row per hypothesis ID.
- **Estimate:** 40 min

### T-704 · Demo instructions
- **Files:** `docs/DEMO.md`
- **Depends on:** T-702
- **Requirements:** §30; A-25; FAC-68, FAC-69
- **Acceptance criteria:**
  - The seven parts in order.
  - Parts 1–2 run live; the others point to committed results.
- **Estimate:** 20 min

### T-705 · README
- **Files:** `README.md`
- **Depends on:** T-602, T-702
- **Requirements:** §22, §29; A-25; FAC-49, FAC-60, FAC-65, FAC-66
- **Acceptance criteria:** every item in ARCHITECTURE §2.1, including:
  - the seed table;
  - the reproducibility statement;
  - the measured runtime and hardware;
  - **Known issues and limitations**;
  - what was not built;
  - the §29 deliverables map.
- **Estimate:** 40 min

### T-706 · AI log
- **Files:** `docs/AI_LOG.md`
- **Depends on:** T-702 (and kept up to date after every AI-assisted task)
- **Requirements:** §2.6, §25; A-23; FAC-61, FAC-62
- **Acceptance criteria:**
  - One dated entry per material AI use, including each AI-generated module.
  - The candidate's modifications and understanding are written in their own words.
- **Estimate:** 30 min

---

### T-707 · Candidate study of the code and the maths
- **Files:** —
- **Depends on:** T-701, T-702
- **Requirements:** §2.6, §34; A-23
- **Acceptance criteria:**
  - The candidate reads `wda/attention.py`, `wda/gradcheck.py`, `wda/train.py` and the derivation.
  - The candidate reruns the tiny-example trace by hand and can explain every line of the attention core and every choice in ASSUMPTIONS.
  - The AI-log "resulting understanding" fields are written in the candidate's own words.
- **Estimate:** 90 min

---

## Phase 8 — Submission

### T-801 · Final checks and submit
- **Files:** —
- **Depends on:** T-703 … T-706
- **Requirements:** §29; A-22; FAC-06, FAC-07, FAC-65, FAC-67
- **Acceptance criteria:**
  - Each Definition-of-Done row in ACCEPTANCE_CRITERIA.md is ticked.
  - The hypothesis text is unchanged since T-001, and every change is a dated amendment.
  - History is unsquashed; the final push is before 23:59 IST on 3 October; the assessors can open the repository.
- **Estimate:** 20 min

---

## Schedule

| Block | When | Tasks |
|---|---|---|
| A | 2 Oct, 20:45 – 00:30 | T-001, T-101 … T-104, T-201 … T-204 → **Checkpoint A** |
| B | 3 Oct, 08:00 – 12:30 | T-301 … T-305, T-401 … T-404, T-501, T-502; T-701 in parallel |
| C | 3 Oct, 13:30 – 18:00 | T-503 … T-507, T-601, T-602 |
| D | 3 Oct, 18:00 – 22:30 | T-702 … T-707, T-801; 22:30 – 23:59 is buffer |

**If behind schedule, cut in this order.** Each cut is a dated PHASE0 amendment, made before the affected result exists:
1. the demo recording;
2. the F5 slice;
3. the single-factor shift conditions, keeping the control and the combined condition (FAC-42 still holds);
4. the warehouse uniform control (A-27): the verdict then names the calendar-feature confound instead;
5. the toy-task step budget, reduced.

**Never cut:** gradient verification, the tests, the baseline comparison, the shift experiment, the failure investigation, the derivation, the reflection or the AI log.
