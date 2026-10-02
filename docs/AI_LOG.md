# AI-assistance log

PRD §2.6: "Material AI assistance must be documented." Each entry gives the six §25 fields. "Material" means any AI output that shaped the code, documents, design or analysis (ASSUMPTIONS A-23). Excerpts are summarised. The **candidate modifications** and **resulting understanding** fields are written by the candidate.

Tool for every entry so far: **Claude Code** (Claude Opus 5.5), including its multi-agent workflows, in which several AI agents drafted, reviewed and cross-checked one another.

---

### 1 · 2026-10-01 · Requirements analysis of the PRD
- **Task requested:** analyse the PRD into functional and non-functional requirements, user flows, data model, API, UI, integrations, validation rules, error cases, ambiguities, assumptions (each justified), acceptance criteria, technical constraints, risks and clarification questions. No implementation.
- **Generated output:** a long requirements analysis (about 57,000 words). It was kept as working notes and is not part of the submission; its main findings are summarised here:
  - self-attention needs positional or time information;
  - the §16 diagram is off by one against §17;
  - seasonal naive y(t−23) is the oldest value in the window;
  - scaled and unscaled attention are reparameterisations of each other;
  - the gradient check needs float64.
- **Candidate modifications:** answered the blocking questions with the assessment owner's information: deadline 3 October 2026; Phase 0 submitted with the final submission; submission through a Git repository. *(Candidate to complete.)*
- **Verification performed:** independent AI critique passes; PRD quotations checked against the PRD text; an automated check that every cross-reference in the analysis resolves.
- **Resulting understanding:** *(Candidate to complete.)*

### 2 · 2026-10-01 · Critical review of the analysis
- **Task requested:** review the analysis as a senior engineer would. Sort its requirements into PRD-confirmed and inferred, and list ambiguities, missing requirements, recommended assumptions, scope boundaries and a final list of acceptance criteria.
- **Generated output:** a review, kept as working notes and not part of the submission. Its acceptance criteria became [ACCEPTANCE_CRITERIA.md](ACCEPTANCE_CRITERIA.md). Its main findings:
  - the analysis is faithful to the PRD but heavily over-scoped;
  - seven requirements are misattributed;
  - its oracle and its gradient tolerance have technical flaws;
  - a reduced list of 69 objective acceptance criteria.
- **Candidate modifications:** *(Candidate to complete.)*
- **Verification performed:** five AI reviewers, one of which re-extracted the PRD's obligations before reading the analysis. The verifier pass was cut for time, so each synthesising agent checked the claims it used against the PRD. An automated check confirmed that every PRD section with a deliverable is covered by an acceptance criterion.
- **Resulting understanding:** *(Candidate to complete.)*

### 3 · 2026-10-01 · ASSUMPTIONS.md
- **Task requested:** document every assumption needed where the PRD does not specify the behaviour, concisely.
- **Generated output:** [ASSUMPTIONS.md](ASSUMPTIONS.md), A-01 … A-27.
- **Candidate modifications:** *(Candidate to complete.)*
- **Verification performed:**
  - A PRD-fidelity check and a technical check. These found, among other things, that §20 accepts "instability under extreme inputs" as a genuine failure, and that a one-layer model cannot solve recall if keys and values are separate tokens.
  - Reconciliation with the review's recommended assumptions: 14 changes and A-27 added.
  - A check of the development environment: Python 3.11 was unavailable, so A-01 was changed to Python 3.12.
- **Resulting understanding:** *(Candidate to complete.)*

### 4 · 2026-10-01 · Hypotheses, failure modes and the pre-registration calculation
- **Task requested:** the candidate asked the AI to draft the Phase 0 hypotheses.
- **Generated output:** hypotheses H1–H11, expected verification results V1–V2, failure modes F1–F5, and [prereg/init_stats.py](../prereg/init_stats.py) with its output [prereg/init_stats.txt](../prereg/init_stats.txt), and the design calculation [prereg/design_expectations.py](../prereg/design_expectations.py). These are now in [PHASE0.md](PHASE0.md) items 4 and 5.
- **Corrections made during drafting:**
  - The calculation showed that unscaled attention at d_k = 64 does **not** shrink gradients uniformly: the median is about the same or larger, while about 9–16% of rows nearly vanish. H3 was rewritten to predict uneven gradients.
  - The uniform-attention control's expected accuracy was corrected from 1/vocabulary to about 1/n_pairs.
  - The baseline-ordering design target was raised to a daily amplitude of 5× the noise.
  - A closed-form calculation of the baselines' expected errors (`prereg/design_expectations.py`) showed that the drafted H5 ordering was wrong. The Monday step and the 24-hour spike echo make seasonal naive worse than last observation. With the candidate's agreement, H5 was revised before the first commit.
- **Candidate modifications:** *(Candidate to complete: which predictions were kept, changed or rejected.)*
- **Verification performed:** the calculation was rerun and its output saved. Each prediction's reasoning was re-derived. A PRD-fidelity and technical review cross-checked the hypotheses against ASSUMPTIONS.
- **Resulting understanding:** *(Candidate to complete.)*

### 5 · 2026-10-02 · ARCHITECTURE.md
- **Task requested:** design the simplest technical architecture that satisfies the PRD, the analysis and the assumptions, with each decision's alternatives and trade-offs. No code.
- **Generated output:** [ARCHITECTURE.md](ARCHITECTURE.md).
- **Candidate modifications:** *(Candidate to complete.)*
- **Verification performed:** two AI critics, one for coverage against the acceptance criteria and one for simplicity, found 40 issues, which were applied. A third, technical critic failed to run (an authentication error), so its lens was not applied. Among the fixes:
  - the `python -m` run convention, reproduced as necessary on a scratch layout;
  - H3's per-row ∂L/∂q logging;
  - the `FINAL_TEST` switch;
  - a separate shift seed;
  - the stability sweep dropped as out of scope.
- **Resulting understanding:** *(Candidate to complete.)*

### 6 · 2026-10-02 · PHASE0.md
- **Task requested:** write the Phase 0 design document (§5), with the drafted hypotheses carried over unchanged and concrete design values.
- **Generated output:** [PHASE0.md](PHASE0.md).
- **Candidate modifications:** *(Candidate to complete.)*
- **Verification performed:** an AI verifier checked it against the Phase 0 acceptance criteria and the design targets, and confirmed that the hypotheses were copied verbatim.
- **Resulting understanding:** *(Candidate to complete.)*

### 7 · 2026-10-02 · Block A code: setup and attention core (T-101 … T-204)
- **Task requested:** implement the plan task by task: pinned environment, package skeleton, configuration, run helper, attention core, its tests, gradient check, and the trace and stability scripts.
- **Generated output:** `requirements.txt`, `CLAUDE.md`, `wda/config.py`, `wda/run.py`, `wda/attention.py`, `wda/gradcheck.py`, `experiments/attention_trace.py`, `experiments/gradcheck.py`, `experiments/stability.py`, and the tests `tests/test_run.py`, `tests/test_attention.py`, `tests/test_gradients.py`, `tests/test_prohibited_api.py`.
- **Candidate modifications:** *(Candidate to complete.)*
- **Verification performed:**
  - The tiny example was worked out by hand and matches the code at every step.
  - 22 tests pass.
  - A mutation check planted five bugs in the attention core (KQᵀ, no scaling, softmax over the wrong axis, division by d_k, AᵀV). It found that the first tiny example gave a symmetric score matrix, so KQᵀ passed every test. The example was changed and an element-by-element reference test was added; all five bugs are now caught.
  - The gradient check's negative control (a detached W_Q) is reported as disagreeing.
- **Resulting understanding:** *(Candidate to complete.)*

### 8 · 2026-10-02 · Data, models, training and metrics (T-301 … T-404), and review fixes
- **Task requested:** continue the plan: warehouse generator, windows and splits, toy task, models, baselines, training loop, metrics; then apply the findings of two independent AI reviews (of the Block A code and of the architecture and plan).
- **Generated output:**
  - modules: `wda/warehouse_data.py`, `wda/toy_data.py`, `wda/models.py`, `wda/baselines.py`, `wda/train.py`, `wda/metrics.py`;
  - `experiments/generate_data.py`;
  - tests: `tests/test_data.py`, `tests/test_models.py`, `tests/test_train_metrics.py`;
  - changes to `wda/run.py`, `requirements.txt`, the test suite, ARCHITECTURE and the plan.
- **Candidate modifications:** the candidate approved six design changes from the architecture review:
  1. a separate `run.json`;
  2. a macOS-safe PyTorch pin;
  3. ablation divergence recorded as a result;
  4. shift built before the freeze;
  5. the H3 wording fix;
  6. study task T-707.

  *(Candidate to complete.)*
- **Verification performed:**
  - The generator was checked against the PHASE0 design calculation: autocorrelation 0.84/0.66/0.68 at lags 1/24/168, amplitude 5.42 × the noise SD, 2.5% of hours in events. Split counts are 6,024 / 1,344 / 1,344, as designed.
  - 61 tests pass, and a full rerun reproduces every result file exactly, apart from the runtime file.
  - The Block A review found three real problems, all fixed:
    - V2's "finite means correct" claim is false: at logits [88.5, 87.5, 0], naive softmax returns finite zeros. It will be reported as refuted.
    - The FAC-15 test could not see an unregistered parameter.
    - The text-based prohibited-API scan missed aliases. It was replaced by an AST scan.
- **Resulting understanding:** *(Candidate to complete.)*

### 9 · 2026-10-02 · Toy task, ablation, and three Phase 0 amendments
- **Task requested:**
  - run the toy experiment (T-501) and the ablation (T-502);
  - then review the architecture and plan with five sceptical AI reviewers, each checked by an adversarial verifier.
- **Generated output:**
  - `experiments/toy.py` and `experiments/ablation.py`, with their results;
  - three PHASE0 amendments, implemented in `wda/config.py`, `wda/baselines.py` and `wda/metrics.py`:
    1. the §19 verdict rule;
    2. 52-week shift series;
    3. baseline B4.
- **Candidate modifications:** the candidate reviewed and approved the three amendments before any warehouse or shift result existed. *(Candidate to complete.)*
- **Verification performed:**
  - H1–H4 were scored against their frozen thresholds; all four hold.
  - The verifiers rejected several reviewer claims, and the reasons are recorded with each claim. For example, the claim that paired arms start from different weights was refuted by the existing test.
  - The upheld claims were checked by direct computation. The 8-week seed-202 shift series really has one spike, and the relative verdict rule really does label a perfect forecaster "simply adapted".
- **Resulting understanding:** *(Candidate to complete.)*

### 10 · 2026-10-02 · Warehouse, shift and failure experiments; run_all; the documents
- **Task requested:** finish the plan:
  - the warehouse model (validation, then a single test evaluation after a freeze commit), the distribution shift, the failure investigation and `run_all`;
  - the fresh-clone check;
  - DERIVATION, RESULTS, README, DEMO, DEBUGGING and the reflection scaffold.
- **Generated output:**
  - experiment scripts: `experiments/warehouse.py`, `experiments/shift.py`, `experiments/failure.py`, `experiments/run_all.py`;
  - shared helpers: `select`, `attention_rows`, `load_warehouse_models`;
  - their results;
  - documents: `docs/DERIVATION.md` and `docs/RESULTS.md` (each drafted by one AI agent and checked by another), `README.md`, `docs/DEMO.md`, `docs/DEBUGGING.md`, and the factual parts of `docs/REFLECTION.md`.
- **Candidate modifications:** *(Candidate to complete.)*
- **Verification performed:**
  - The test split was scored once, in commit `abf1a8e`, after an empty "freeze configuration" commit. Later reruns reproduce those numbers deterministically.
  - A fresh clone installed from the pins, passed the tests, and reproduced every result with `run_all`. The only exception was last-bit float32 differences under machine load.
  - The failure explanation was tested by a dose-response experiment, after two explanations from the AI assistant were refuted by its own experiments (DEBUGGING episode 4).
  - The DERIVATION verifier made 13 corrections.
  - Two AI agents answered a chat question instead of doing their writing task, and their output was discarded. That episode shows why each document needed an independent check.
  - The AI assistant mistakenly stopped a `run_all` that the candidate had started in their own terminal. No results were lost: they were restored from Git.
- **Resulting understanding:** *(Candidate to complete.)*

### 11 · 2026-10-02 · Reflection answers
- **Task requested:** the candidate asked the AI to write all the reflection answers, including the personal questions (1, 4, 5, 7, 8 and 11).
- **Generated output:** the complete `docs/REFLECTION.md`. The answers are drawn only from the project record (PHASE0, RESULTS, DEBUGGING and this log), in the candidate's voice; they describe no experience that did not happen in this project.
- **Candidate modifications:** *(Candidate to complete: what was kept, changed or rewritten.)*
- **Verification performed:** every factual claim was checked against the results and the debugging journal.
- **Resulting understanding:** *(Candidate to complete.)*

