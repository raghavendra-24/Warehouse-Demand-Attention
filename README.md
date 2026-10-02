# Warehouse Demand Attention

First-principles scaled dot-product self-attention, verified, then reused to forecast next-hour demand in a synthetic warehouse environment.

The design is in [docs/PHASE0.md](docs/PHASE0.md), [docs/ASSUMPTIONS.md](docs/ASSUMPTIONS.md) and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). The rest of this README is completed as the implementation lands ([docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md), T-705).

## Setup

Requires Python 3.12. Everything runs on CPU.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` pins every package exactly. PyTorch comes from the official CPU wheel index, which the file names with `--extra-index-url`.

## Running

All commands run from the repository root, using the `-m` form so that the `wda` package is found without an install step:

```bash
.venv/bin/python -m pytest                 # test suite
.venv/bin/python -m experiments.<stage>    # one stage, for example experiments.gradcheck
```
