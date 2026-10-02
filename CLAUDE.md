# Working rules for this repository

- The PRD is the source of truth; then `docs/ASSUMPTIONS.md`, `docs/PHASE0.md` and `docs/ARCHITECTURE.md`. Work follows `docs/IMPLEMENTATION_PLAN.md`, one task per commit.
- If a change would alter the architecture, an assumption or a Phase 0 value: stop and explain first. Phase 0 hypotheses are frozen; changes are dated amendments in `docs/PHASE0.md`.
- No model result on validation or test before its task; no test-split result while `FINAL_TEST` is off.
- Attention core (`wda/attention.py`): explicit tensor operations and a hand-written softmax only. Never `nn.MultiheadAttention`, `F.scaled_dot_product_attention`, `nn.Transformer*`, a library softmax, or third-party attention code.
- Python 3.12 from `.venv`. Run everything from the repository root with `python -m pytest` and `python -m experiments.<stage>`.
- No company or employer names anywhere in the repository. Commit messages carry no AI attribution lines.
- Every material AI contribution gets an entry in `docs/AI_LOG.md`.
