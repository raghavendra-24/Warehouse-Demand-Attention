# Warehouse Demand Attention

Scaled dot-product self-attention built from first principles, verified, trained on a toy task and ablated, then reused to forecast next-hour demand in a synthetic warehouse environment. It is compared against simple baselines, tested under distribution shift, and taken apart where it fails.

The project follows one chain (PRD §35):

**Phase 0 → derivation → dataset → hypotheses → attention → gradient check → toy task → ablation → warehouse forecasting → baselines → generalisation → failure analysis → reflection.**

The hypotheses were committed before any result existed: commit `2cba495` comes first in the history.

## Headline results

| Experiment | Result | Pre-registered prediction |
|---|---|---|
| Toy task (associative recall) | Attention reaches **0.9999** held-out accuracy; the uniform-attention control reaches **0.1249** (= 1/8) | H1 ✅ |
| Ablation, d_k = 64 | Unscaled attention starts almost one-hot (entropy 0.15 of ln n, against 0.83 when scaled), needs **2.07×** the steps to reach 95%, and one seed never learns. At d_k = 4 there is no difference. | H2, H3, H4 ✅ |
| Warehouse, test MAE (orders/h) | B4 hour-of-week mean **15.42** (the reference, chosen on validation); attention **14.97 ± 0.39** (seeds 14.55, 14.88, 15.49); uniform control 43.21. RMSE 25.23 against 30.65. | H5 ✅, H6 ❌ (the gain is not the same sign in every seed), H7 ❌, H8 ❌ |
| Distribution shift (four 52-week series) | Noise and spike effects as predicted. The §19 verdict is **inconclusive**: the model's MAE rises +14.8 to +15.1, against B4's +14.9. | H9, H10, H11 ✅ |
| Failure | During large spikes the forecast saturates near 350 orders/h while demand is near 590. The value path carries almost no demand magnitude (dg/dz 0.02–0.06), so the convex-combination readout can only re-weight tokens. | F4 ✅, case investigated |

Full results, with an outcome for every pre-registered ID: [docs/RESULTS.md](docs/RESULTS.md).

## Setup

Python 3.12, CPU only.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` pins every package exactly.
- **Linux and Windows:** PyTorch 2.14.1 comes from the official CPU wheel index (`--extra-index-url` in the file).
- **macOS:** the same version comes from PyPI, through a platform marker, because macOS wheels are CPU-only and have no `+cpu` label.

Versions used: Python 3.12.3, torch 2.14.1+cpu, numpy 2.5.3, matplotlib 3.11.2, pytest 9.1.1.

## Running

Every command runs from the repository root, in the `-m` form, so that the `wda` package is found without an install step. Prefix each with `.venv/bin/`, or activate the environment first.

| Stage (§3) | Command | Runtime* |
|---|---|---|
| Tests | `python -m pytest` | ~10 s |
| Dataset generation (also the reset/regeneration mechanism, §4) | `python -m experiments.generate_data` | 1 s |
| Attention trace (demo part 1) | `python -m experiments.attention_trace` | < 1 s |
| Gradient verification | `python -m experiments.gradcheck` | < 1 s |
| Numerical stability | `python -m experiments.stability` | < 1 s |
| Toy task (training) | `python -m experiments.toy` | ~70–90 s |
| Ablation | `python -m experiments.ablation` | ~4 min |
| Warehouse training and evaluation | `python -m experiments.warehouse` | ~3 min |
| Generalisation (distribution shift) | `python -m experiments.shift` | ~5 s |
| Failure analysis | `python -m experiments.failure` | ~2 s |
| **Everything, in order** | `python -m experiments.run_all` | **486 s** |

\*Measured on an Intel Core i5-1335U laptop with 4 PyTorch threads. Every stage writes `results/<stage>/`: `metrics.json`, `table.md`, figures, and `run.json` (runtime, versions, CPU, git revision).

The test split is scored only when `FINAL_TEST = True` in `wda/config.py`. It was switched on once, in commit `abf1a8e`, after the "freeze configuration" commit (A-12, FAC-32).

## Tests

63 tests, each named after the bug it would catch. The five §23 minimum items:

| §23 item | Test |
|---|---|
| Attention tensor shapes | `tests/test_attention.py::test_shapes_catch_swapped_dimensions_or_missing_transpose` |
| Softmax normalisation | `tests/test_attention.py::test_softmax_rows_sum_to_one_catches_wrong_axis` |
| Attention output dimensions | `tests/test_attention.py::test_shapes_catch_swapped_dimensions_or_missing_transpose`, `tests/test_models.py::test_warehouse_model_gives_one_prediction_per_window` |
| Gradient verification | `tests/test_gradients.py::test_autograd_matches_float64_central_differences_for_every_entry` |
| Deterministic dataset generation | `tests/test_data.py::test_same_seed_gives_identical_series_and_different_seed_differs` |

The extra tests target subtle bugs:
- a forward pass checked against values computed by hand, which catches KQᵀ, AᵀV and a missing 1/√d_k;
- an element-by-element reference;
- permutation equivariance;
- unscaled attention equal to scaled attention with W_Q·√d_k;
- a negative control for the gradient check;
- leakage-free splits and standardisation;
- the FAC-34 baseline ramp;
- paired-arm initialisation;
- an AST scan for prohibited attention APIs.

A mutation check, which plants bugs in the attention core, confirmed that each of these bugs makes a test fail ([docs/DEBUGGING.md](docs/DEBUGGING.md)).

**One attention implementation, two call sites (FAC-47):**
- `grep -n "^def attention" wda/attention.py` finds the single definition;
- `grep -n "self.attn(" wda/models.py` finds the toy model (line 41) and the warehouse model (line 60), both through `AttentionWeights`.

## Seeds

| Seed | Purpose |
|---|---|
| 101 | warehouse series: train, validation and test (event and noise streams split by `SeedSequence`) |
| 202 | the four distribution-shift series, sharing one event timeline |
| 301 / 302 / 303 | toy task: train / validation / test sequences |
| 0, 1, 2 | model initialisation and batch order. Paired arms share a seed, and so start bit-identical. |
| 7 | gradient-check tensors |
| 0, 1 | `prereg/init_stats.py`, the pre-implementation calculation |

## Reproducibility

- **Exact:** the generated data, and every result table, metrics file and figure, are reproduced by `python -m experiments.run_all` on the same machine. `git diff` shows no change apart from `run.json`.
- **Within a tolerance:** when the machine is busy, float32 model quantities can differ in their last bit, about 1e-7 relative, because the maths library changes its summation order. This was observed once, in four ablation gradient norms.
- **Not promised:** bit-identical training on a different machine (A-24).
- **Pre-submission check (T-602), done on 2 October 2026:** a fresh clone installed from `requirements.txt`, all 63 tests passed, and `run_all` regenerated every result in 446 s. Every result file matched the committed version except four ablation gradient norms, which differed in the last float32 bit. The committed run had been made while the machine was busy; the committed ablation file now comes from that clean rerun.

## Deliverables map (§29)

| # | Deliverable | Where |
|---|---|---|
| 1 | Source code | [wda/](wda), [experiments/](experiments) |
| 2 | README | this file |
| 3 | Phase 0 design document | [docs/PHASE0.md](docs/PHASE0.md) (with dated amendments) |
| 4 | Mathematical derivation | [docs/DERIVATION.md](docs/DERIVATION.md) |
| 5 | Synthetic dataset generator | [wda/warehouse_data.py](wda/warehouse_data.py), [experiments/generate_data.py](experiments/generate_data.py) |
| 6 | First-principles attention | [wda/attention.py](wda/attention.py) |
| 7 | Gradient verification | [wda/gradcheck.py](wda/gradcheck.py), [experiments/gradcheck.py](experiments/gradcheck.py), [results/gradcheck/](results/gradcheck) |
| 8 | Toy learning experiment | [experiments/toy.py](experiments/toy.py), [results/toy/](results/toy) |
| 9 | Warehouse demand model | [wda/models.py](wda/models.py), [experiments/warehouse.py](experiments/warehouse.py) |
| 10 | Baselines | [wda/baselines.py](wda/baselines.py) |
| 11 | Ablation | [experiments/ablation.py](experiments/ablation.py), [results/ablation/](results/ablation) |
| 12 | Generalisation experiment | [experiments/shift.py](experiments/shift.py), [results/shift/](results/shift) |
| 13 | Failure investigation | [experiments/failure.py](experiments/failure.py), [results/failure/](results/failure) |
| 14 | Test suite | [tests/](tests) |
| 15 | Experiment results | [results/](results), [docs/RESULTS.md](docs/RESULTS.md) |
| 16 | Reflection | [docs/REFLECTION.md](docs/REFLECTION.md) |
| 17 | AI assistance log | [docs/AI_LOG.md](docs/AI_LOG.md) |
| 18 | Demo instructions | [docs/DEMO.md](docs/DEMO.md) |

Also included:
- [docs/ASSUMPTIONS.md](docs/ASSUMPTIONS.md): A-01 … A-27, with the reason for each;
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md);
- [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md);
- [docs/ACCEPTANCE_CRITERIA.md](docs/ACCEPTANCE_CRITERIA.md);
- [docs/DEBUGGING.md](docs/DEBUGGING.md);
- [prereg/](prereg): the calculations quoted in Phase 0.

## Key decisions

The reasoning behind each is in [docs/ASSUMPTIONS.md](docs/ASSUMPTIONS.md) and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

- **Stack and rules:** PyTorch with autograd. The attention core uses only explicit tensor operations and a hand-written stable softmax (A-01, A-02).
- **Model:** one layer and one head. Each hour is a token of standardised demand plus sin/cos hour-of-day and day-of-week; the readout is the last position; there is no residual path (A-03, A-05, A-06).
- **Evaluation:**
  - A chronological split in whole weeks (36 / 8 / 8). The standardisation and the B4 profile use training targets only (A-12, A-13).
  - MAE is the primary metric (A-14).
  - Three seeds, and an effect is claimed only if it has the same sign in all three (A-16).
- **Experiments:**
  - The ablation is on the toy task: d_k ∈ {4, 64} × scaled or unscaled × 3 paired seeds (A-19).
  - The shift uses four 52-week series, with no retraining (A-20).
- **Changes made before the results they affect:** three dated Phase 0 amendments, all made before any warehouse or shift result:
  1. the §19 verdict rule;
  2. 52-week shift series;
  3. baseline B4.

## Known issues and limitations

- **Calendar-feature confound.** The model sees calendar features that B1–B3 do not. B4 addresses this directly, and the model's MAE advantage over B4 is small and not the same sign in every seed (A-27 verdict: inconclusive).
- **Spike saturation.** The model cannot forecast spikes much larger than those seen in training; see the failure investigation. A residual path for demand magnitude is the proposed fix. It was not implemented: the PRD does not require a perfect fix, and the configuration was frozen.
- **Three hypotheses refuted.**
  - H6: a 5–20% gain in every seed was predicted.
  - H7: the gain was predicted to come mostly from post-event hours, but it comes everywhere.
  - H8: attention was predicted to concentrate on t−23 and t, but it spreads out.
  - V2 is partly refuted: naive softmax can be finite and still wrong.
  - All of these are reported, not tuned away.
- **Reproducibility tolerance:** float32 last-bit differences under machine load (see above).
- **No skipped or expected-failure tests.**

## Deliberately not built

Each item below is either not required by the PRD (§27, §28) or deliberately left out of scope:
- a full Transformer, multi-head attention, stacked layers, residual or normalisation blocks;
- GPU support, distributed training, deployment, or a frontend;
- LoRA, JAX, a paper reproduction, or manual backpropagation;
- an experiment tracker, a CLI framework, config hashing or checksum machinery;
- a separate softmax-saturation sweep (the ablation logs cover it).

The goal, following §2.7, was a small, verified core rather than breadth.

## AI assistance

AI tools were used throughout and are documented entry by entry in [docs/AI_LOG.md](docs/AI_LOG.md), as §2.6 and §25 require.
