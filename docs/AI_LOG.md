# AI-assistance log

PRD §2.6: "Material AI assistance must be documented." Each entry gives the six §25 fields. "Material" means any AI output that shaped the code, documents, design or analysis (ASSUMPTIONS A-23). Excerpts are summarised. The **candidate modifications** and **resulting understanding** fields are written by the candidate.

Tool for every entry so far: **Claude Code** (Claude Opus 5.5), including its multi-agent workflows, in which several AI agents drafted, reviewed and cross-checked one another.

---

### 1 · 2026-10-01 · Requirements analysis of the PRD
- **Task requested:** analyse the PRD into functional and non-functional requirements, user flows, data model, API, UI, integrations, validation rules, error cases, ambiguities, assumptions (each justified), acceptance criteria, technical constraints, risks and clarification questions. No implementation.
- **Generated output:** a long requirements analysis (about 57,000 words), kept outside the repository as working notes. Its main findings:
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
- **Generated output:** a review, kept outside the repository. Its main findings:
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

