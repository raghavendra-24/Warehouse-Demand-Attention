# AI-assistance log

PRD §2.6: "Material AI assistance must be documented." Each entry gives the six §25 fields. "Material" means any AI output that shaped the code, documents, design or analysis (ASSUMPTIONS A-23). Excerpts are summarised. The **candidate modifications** fields record the candidate's decisions. The **resulting understanding** fields were drafted with AI assistance at the candidate's request, from the project record.

## Tools used to complete this project

| Tool | Kind | Used for |
|---|---|---|
| **Claude Code** with the Claude Opus 5.5 model, run from the VS Code extension | AI assistant | Every entry below: requirements analysis and review; ASSUMPTIONS, PHASE0, ARCHITECTURE and the plan; all code, tests and experiment scripts; the documents; the PDF versions and the demo video |
| Claude Code multi-agent workflows (same model) | AI assistant | Several agents working in parallel: drafting, independent review and adversarial verification of each other's output (entries 1–3, 5, 6, 8–10, 12) |
| Python 3.12, PyTorch 2.14.1 (CPU), NumPy 2.5.3, matplotlib 3.11.2, pytest 9.1.1 | Software, not AI | Implementation, experiments, figures, tests |
| Git and GitHub | Software, not AI | Version history: the pre-registration commit, the freeze commit, and submission |
| Python-Markdown and Google Chrome (headless); pyte, Pillow and an ffmpeg encoder | Software, not AI | Converting the Markdown documents to PDF; recording the terminal session and assembling the demo video |

No other AI tool was used: no code-completion plugin and no other chatbot. Each entry's **Tool used** line names the tool for that entry.

---

### 1 · 2026-10-01 · Requirements analysis of the PRD
- **Tool used:** Claude Code (Claude Opus 5.5), with a multi-agent workflow.
- **Task requested:** analyse the PRD into functional and non-functional requirements, user flows, data model, API, UI, integrations, validation rules, error cases, ambiguities, assumptions (each justified), acceptance criteria, technical constraints, risks and clarification questions. No implementation.
- **Generated output:** a long requirements analysis (about 57,000 words). It was kept as working notes and is not part of the submission; its main findings are summarised here:
  - self-attention needs positional or time information;
  - the §16 diagram is off by one against §17;
  - seasonal naive $y(t-23)$ is the oldest value in the window;
  - scaled and unscaled attention are reparameterisations of each other;
  - the gradient check needs float64.
- **Candidate modifications:** No edits to the generated analysis. The candidate supplied the answers to the blocking questions: deadline 3 October 2026 at midnight (read as 23:59 IST), Phase 0 submitted with the final submission, submission through a Git repository whose visibility the candidate sets at the end.
- **Verification performed:** independent AI critique passes; PRD quotations checked against the PRD text; an automated check that every cross-reference in the analysis resolves.
- **Resulting understanding:** The PRD grades method over accuracy: formulation, verification, hypotheses stated before results, and honest failure analysis. Self-attention needs position or time information, because without it the model cannot tell $t-1$ from $t-23$.

### 2 · 2026-10-01 · Critical review of the analysis
- **Tool used:** Claude Code (Claude Opus 5.5), with a multi-agent workflow.
- **Task requested:** review the analysis as a senior engineer would. Sort its requirements into PRD-confirmed and inferred, and list ambiguities, missing requirements, recommended assumptions, scope boundaries and a final list of acceptance criteria.
- **Generated output:** a review, kept as working notes and not part of the submission. Its acceptance criteria became [ACCEPTANCE_CRITERIA.md](ACCEPTANCE_CRITERIA.md). Its main findings:
  - the analysis is faithful to the PRD but heavily over-scoped;
  - seven requirements are misattributed;
  - its oracle and its gradient tolerance have technical flaws;
  - a reduced list of 69 objective acceptance criteria.
- **Candidate modifications:** No edits. The candidate asked for the review and accepted its recommendations as the basis for the assumptions, the scope and the acceptance criteria.
- **Verification performed:** five AI reviewers, one of which re-extracted the PRD's obligations before reading the analysis. The verifier pass was cut for time, so each synthesising agent checked the claims it used against the PRD. An automated check confirmed that every PRD section with a deliverable is covered by an acceptance criterion.
- **Resulting understanding:** An analysis can be faithful to the PRD and still be over-scoped. Separating what the PRD requires from what is merely good practice is what made a two-day plan feasible.

### 3 · 2026-10-01 · ASSUMPTIONS.md
- **Tool used:** Claude Code (Claude Opus 5.5), with a multi-agent workflow.
- **Task requested:** document every assumption needed where the PRD does not specify the behaviour, concisely.
- **Generated output:** [ASSUMPTIONS.md](ASSUMPTIONS.md), A-01 … A-27.
- **Candidate modifications:** No edits to the text. The candidate asked for the document to be written while the review was running, and accepted the corrections from the two checks and from the review.
- **Verification performed:**
  - A PRD-fidelity check and a technical check. These found, among other things, that §20 accepts "instability under extreme inputs" as a genuine failure, and that a one-layer model cannot solve recall if keys and values are separate tokens.
  - Reconciliation with the review's recommended assumptions: 14 changes and A-27 added.
  - A check of the development environment: Python 3.11 was unavailable, so A-01 was changed to Python 3.12.
- **Resulting understanding:** Where the PRD is silent, every choice needs a stated reason and a considered alternative. Choices interact: Poisson noise, for example, cannot support a "higher noise" shift without also changing the demand level.

### 4 · 2026-10-01 · Hypotheses, failure modes and the pre-registration calculation
- **Tool used:** Claude Code (Claude Opus 5.5).
- **Task requested:** the candidate asked the AI to draft the Phase 0 hypotheses.
- **Generated output:** hypotheses H1–H11, expected verification results V1–V2, failure modes F1–F5, and [prereg/init_stats.py](../prereg/init_stats.py) with its output [prereg/init_stats.txt](../prereg/init_stats.txt), and the design calculation [prereg/design_expectations.py](../prereg/design_expectations.py). These are now in [PHASE0.md](PHASE0.md) items 4 and 5.
- **Corrections made during drafting:**
  - The calculation showed that unscaled attention at $d_k = 64$ does **not** shrink gradients uniformly: the median is about the same or larger, while about 9–16% of rows nearly vanish. H3 was rewritten to predict uneven gradients.
  - The uniform-attention control's expected accuracy was corrected from $1/\text{vocabulary}$ to about $1/n_{\text{pairs}}$.
  - The baseline-ordering design target was raised to a daily amplitude of 5× the noise.
  - A closed-form calculation of the baselines' expected errors (`prereg/design_expectations.py`) showed that the drafted H5 ordering was wrong. The Monday step and the 24-hour spike echo make seasonal naive worse than last observation. With the candidate's agreement, H5 was revised before the first commit.
- **Candidate modifications:** No edits to the wording. The candidate asked the AI to draft the hypotheses. After the design calculation, the candidate chose to amend H5 before the first commit rather than commit a prediction that the calculation already contradicted.
- **Verification performed:** the calculation was rerun and its output saved. Each prediction's reasoning was re-derived. A PRD-fidelity and technical review cross-checked the hypotheses against ASSUMPTIONS.
- **Resulting understanding:** A pre-registered prediction needs a number and a condition that would refute it. Checking the predictions against our own design, by simulation and closed-form calculation, before committing caught two wrong drafts: H3's "smaller gradients" and H5's baseline order.

### 5 · 2026-10-02 · ARCHITECTURE.md and IMPLEMENTATION_PLAN.md
- **Tool used:** Claude Code (Claude Opus 5.5), with a multi-agent workflow.
- **Task requested:**
  - design the simplest technical architecture that satisfies the PRD, the analysis and the assumptions, with each decision's alternatives and trade-offs, and no code;
  - then turn it into an implementation plan of numbered tasks, following the candidate's "TASK-xxx" template.
- **Generated output:**
  - [ARCHITECTURE.md](ARCHITECTURE.md);
  - [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md): 34 tasks in nine phases (Phase 0 to Phase 8), each with its files, dependencies, requirement IDs, acceptance criteria and time estimate, plus a schedule and an ordered list of what to cut if time ran short. Task T-707 was added later (entry 8);
  - [ACCEPTANCE_CRITERIA.md](ACCEPTANCE_CRITERIA.md): the review's 69 criteria (entry 2), copied into the repository so that the FAC references resolve.
- **Candidate modifications:** No edits. The candidate approved applying the critics' fixes and committing the pre-registration, which includes all three documents.
- **Verification performed:**
  - Architecture: two AI critics, one for coverage against the acceptance criteria and one for simplicity, found 40 issues, which were applied. A third, technical critic failed to run (an authentication error), so its lens was not applied. Among the fixes:
    - the `python -m` run convention, reproduced as necessary on a scratch layout;
    - H3's per-row $\partial L / \partial q$ logging;
    - the `FINAL_TEST` switch;
    - a separate shift seed;
    - the stability sweep dropped as out of scope.
  - Plan: a script checked that every task's dependencies exist, that there are no cycles, and that each of the 69 acceptance criteria is assigned to at least one task. It caught a dependency cycle in the AI-log task and four Phase 0 criteria with no task; both were fixed. The plan was later reviewed together with the architecture by the sceptical design review (entries 8 and 9).
- **Resulting understanding:** Even the simplest design needs explicit decisions on seeding, test-set discipline and reproducibility. Reviews showed that details such as the run convention and measuring H3 on the readout row decide whether an experiment can be scored at all. A plan can be checked mechanically: every acceptance criterion needs a task, and the task order must keep the test split unscored until after the freeze.

### 6 · 2026-10-02 · PHASE0.md
- **Tool used:** Claude Code (Claude Opus 5.5), with a multi-agent workflow.
- **Task requested:** write the Phase 0 design document (§5), with the drafted hypotheses carried over unchanged and concrete design values.
- **Generated output:** [PHASE0.md](PHASE0.md).
- **Candidate modifications:** No edits. To save time, the candidate approved skipping the separate AI verifier, so the AI's own checks were used instead (a verbatim-copy check of the hypotheses and the design-target checks). The H5 revision follows the candidate's choice in entry 4.
- **Verification performed:** the AI's own checks only: a verbatim-copy check of the hypotheses against the drafts, and the design-target checks. The separate verifier was stopped before it ran (see Candidate modifications).
- **Resulting understanding:** Phase 0 must be specific to this generator: concrete parameter values, design targets and expected baseline errors. Only then can each result be compared with a prediction made before it existed.

### 7 · 2026-10-02 · Block A code: setup and attention core (T-101 … T-204)
- **Tool used:** Claude Code (Claude Opus 5.5).
- **Task requested:** implement the plan task by task: pinned environment, package skeleton, configuration, run helper, attention core, its tests, gradient check, and the trace and stability scripts.
- **Generated output:** `requirements.txt`, `CLAUDE.md`, `wda/config.py`, `wda/run.py`, `wda/attention.py`, `wda/gradcheck.py`, `experiments/attention_trace.py`, `experiments/gradcheck.py`, `experiments/stability.py`, and the tests `tests/test_run.py`, `tests/test_attention.py`, `tests/test_gradients.py`, `tests/test_prohibited_api.py`.
- **Candidate modifications:** No edits to the code. The candidate approved starting the implementation and working task by task. They asked for the repository under their personal account, kept private, and for commit messages without AI attribution lines.
- **Verification performed:**
  - The tiny example was worked out by hand and matches the code at every step.
  - 22 tests pass.
  - A mutation check planted five bugs in the attention core ($KQ^\top$, no scaling, softmax over the wrong axis, division by $d_k$, $A^\top V$). It found that the first tiny example gave a symmetric score matrix, so $KQ^\top$ passed every test. The example was changed and an element-by-element reference test was added; all five bugs are now caught.
  - The gradient check's negative control (a detached $W_Q$) is reported as disagreeing.
- **Resulting understanding:** I can trace the seven attention operations by hand on the tiny example. A gradient check only compares autograd with the same forward code, so the forward pass needs its own independent test, and the test example must be asymmetric to catch a transposed $QK^\top$ or $A^\top V$.

### 8 · 2026-10-02 · Data, models, training and metrics (T-301 … T-404), and review fixes
- **Tool used:** Claude Code (Claude Opus 5.5), with a multi-agent workflow.
- **Task requested:** continue the plan: warehouse generator, windows and splits, toy task, models, baselines, training loop, metrics; then apply the findings of two independent AI reviews (of the Block A code and of the architecture and plan).
- **Generated output:**
  - modules: `wda/warehouse_data.py`, `wda/toy_data.py`, `wda/models.py`, `wda/baselines.py`, `wda/train.py`, `wda/metrics.py`;
  - `experiments/generate_data.py`;
  - tests: `tests/test_data.py`, `tests/test_models.py`, `tests/test_train_metrics.py`;
  - changes to `wda/run.py`, `requirements.txt`, the test suite, ARCHITECTURE and the plan.
- **Candidate modifications:** No edits to the code. The candidate approved six design changes from the architecture review: a separate `run.json`; a macOS-safe PyTorch pin; ablation divergence recorded as a result; shift built before the freeze; the H3 wording fix; and study task T-707.
- **Verification performed:**
  - The generator was checked against the PHASE0 design calculation: autocorrelation 0.84/0.66/0.68 at lags 1/24/168, amplitude 5.42 × the noise SD, 2.5% of hours in events. Split counts are 6,024 / 1,344 / 1,344, as designed.
  - 61 tests pass, and a full rerun reproduces every result file exactly, apart from the runtime file.
  - The Block A review found three real problems, all fixed:
    - V2's "finite means correct" claim is false: at logits $[88.5, 87.5, 0]$, naive softmax returns finite zeros. It will be reported as refuted.
    - The FAC-15 test could not see an unregistered parameter.
    - The text-based prohibited-API scan missed aliases. It was replaced by an AST scan.
- **Resulting understanding:** For paired experiments, every parameter must come from a seeded generator, because nn.Linear's default initialisation uses the global random state. The standardisation and the B4 profile must be computed from training targets only.

### 9 · 2026-10-02 · Toy task, ablation, and three Phase 0 amendments
- **Tool used:** Claude Code (Claude Opus 5.5), with a multi-agent workflow.
- **Task requested:**
  - run the toy experiment (T-501) and the ablation (T-502);
  - then review the architecture and plan with five sceptical AI reviewers, each checked by an adversarial verifier.
- **Generated output:**
  - `experiments/toy.py` and `experiments/ablation.py`, with their results;
  - three PHASE0 amendments, implemented in `wda/config.py`, `wda/baselines.py` and `wda/metrics.py`:
    1. the §19 verdict rule;
    2. 52-week shift series;
    3. baseline B4.
- **Candidate modifications:** No edits. The candidate read and approved the three Phase 0 amendments (the §19 verdict rule, 52-week shift series, baseline B4) before any warehouse or shift result existed.
- **Verification performed:**
  - H1–H4 were scored against their frozen thresholds; all four hold.
  - The verifiers rejected several reviewer claims, and the reasons are recorded with each claim. For example, the claim that paired arms start from different weights was refuted by the existing test.
  - The upheld claims were checked by direct computation. The 8-week seed-202 shift series really has one spike, and the relative verdict rule really does label a perfect forecaster "simply adapted".
- **Resulting understanding:** Scaling by $\sqrt{d_k}$ changes how attention trains, not what it can represent. Unscaled attention at $d_k = 64$ starts almost one-hot, gets very uneven gradients, and in one seed never learned. A verdict rule can be wrong too, so it should be checked before any result exists.

### 10 · 2026-10-02 · Warehouse, shift and failure experiments; run_all; the documents
- **Tool used:** Claude Code (Claude Opus 5.5), with a multi-agent workflow.
- **Task requested:** finish the plan:
  - the warehouse model (validation, then a single test evaluation after a freeze commit), the distribution shift, the failure investigation and `run_all`;
  - the fresh-clone check;
  - DERIVATION, RESULTS, README, DEMO, DEBUGGING and the reflection scaffold.
- **Generated output:**
  - experiment scripts: `experiments/warehouse.py`, `experiments/shift.py`, `experiments/failure.py`, `experiments/run_all.py`;
  - shared helpers: `select`, `attention_rows`, `load_warehouse_models`;
  - their results;
  - documents: `docs/DERIVATION.md` and `docs/RESULTS.md` (each drafted by one AI agent and checked by another), `README.md`, `docs/DEMO.md`, `docs/DEBUGGING.md`, and the factual parts of `docs/REFLECTION.md`.
- **Candidate modifications:** No edits to the code or the documents.
- **Verification performed:**
  - The test split was scored once, in commit `abf1a8e`, after an empty "freeze configuration" commit. Later reruns reproduce those numbers deterministically.
  - A fresh clone installed from the pins, passed the tests, and reproduced every result with `run_all`. The only exception was last-bit float32 differences under machine load.
  - The failure explanation was tested by a dose-response experiment, after two explanations from the AI assistant were refuted by its own experiments (DEBUGGING episode 4).
  - The DERIVATION verifier made 13 corrections.
  - Two AI agents answered a chat question instead of doing their writing task, and their output was discarded. That episode shows why each document needed an independent check.
  - The AI assistant mistakenly stopped a `run_all` that the candidate had started in their own terminal. No results were lost: they were restored from Git.
- **Resulting understanding:** A one-line hour-of-week baseline almost matched the attention model, so an advantage counts only if it has the same sign in every seed. The model saturates on large spikes because its readout is a weighted average of values that barely depend on demand.

### 11 · 2026-10-02 · Reflection answers
- **Tool used:** Claude Code (Claude Opus 5.5).
- **Task requested:** the candidate asked the AI to write all the reflection answers, including the personal questions (1, 4, 5, 7, 8 and 11).
- **Generated output:** the complete `docs/REFLECTION.md`. The answers are drawn only from the project record (PHASE0, RESULTS, DEBUGGING and this log), in the candidate's voice; they describe no experience that did not happen in this project.
- **Candidate modifications:** No edits. The candidate accepted the drafted answers as written.
- **Verification performed:** every factual claim was checked against the results and the debugging journal.
- **Resulting understanding:** An honest reflection lists what was wrong (H6–H8, part of V2) next to what was right, and keeps what was demonstrated separate from what is only believed.

### 12 · 2026-10-03 · Final check, submission PDFs and the demo recording
- **Tool used:** Claude Code (Claude Opus 5.5), with a multi-agent workflow.
- **Task requested:**
  - check the whole repository once more against the PRD and the acceptance criteria;
  - convert the documents the submission form asks for into PDFs (Phase 0, derivation, experiment report, reflection, AI tools);
  - make a demo video covering the seven §30 parts;
  - write the equations in mathematical form and make the PDFs easier to read.
- **Generated output:**
  - two compliance-audit reports (one against the PRD and the submission form, one against FAC-01 … FAC-69), kept as working notes;
  - the fixes they called for, all to documents, with no change to code or results:
    - corrected reflection claims: the attention peak differs by seed (it is not "19–21 hours" in every seed), ×1.71 not ×1.72, within 1.5% not 1%, and the repository's visibility;
    - AI-log entry numbers and RESULTS sections cited in the reflection, and the remaining rows of its demonstrated-vs-believed table;
    - a **Tool used** line in every entry of this log, the implementation plan added to entry 5, and entry 6's verification corrected;
    - PHASE0 Record 4 (the measured autocorrelation, which section 3 had promised);
    - a README configuration section (§22), six figures embedded in RESULTS.md, and the Definition-of-Done status;
  - the equations in DERIVATION, RESULTS, REFLECTION, this log, DEMO and the README rewritten as typeset LaTeX mathematics. The notation changes; no wording, number or claim does;
  - five PDFs rendered from the committed Markdown documents, each with a title page, a guide to the labels (H, V, F, A, FAC, §, B1–B4), a table of contents, page numbers and typeset equations. The Phase 0 PDF is a typeset edition: its wording, numbers and hypotheses are identical to `docs/PHASE0.md`, which is left exactly as pre-registered;
  - two terminal recordings of the seven parts, with no voice-over. Every command is typed and run live in a fresh clone of `v1.0-submission`, including the full training runs; the waits are shortened and marked "time-lapse", and the committed figures are shown after each stage. The first (4 minutes) shows a neutral prompt. The second (4.5 minutes), made at the candidate's request, shows the candidate's name in the prompt and also prints the Phase 0 predictions next to the measured ablation results.
- **Candidate modifications:** The candidate rejected the first video, a captioned slide walkthrough, and asked instead for a recording of the commands being typed and run, which replaced it; they then asked for their name in the prompt and for the earlier take to be kept. They asked for the equations in mathematical form and for more readable PDFs, and chose to keep `docs/PHASE0.md` unchanged and to typeset only the submission documents and the README. No edits to the documents. The candidate made the repository public for submission.
- **Verification performed:**
  - Each audit finding was checked against the files and the result JSON before it was fixed. For example, the per-seed attention peaks were recomputed from `results/warehouse/metrics.json`.
  - The final take's rerun reproduced every committed `metrics.json` bit for bit. An earlier take, on a busy machine, differed in eight failure-slice means by at most 7e-9 relative, the last-bit float32 effect described in the README. Every frame was checked for private paths, and stills of every part and figure were reviewed. Two earlier takes were discarded: one had truncated tables, and one cut the reading pauses short.
  - The equation rewrite was checked by a script that compares every number and ID with the previous commit, renders each document with GitHub's own Markdown renderer, and compiles every formula; an independent AI verifier then reviewed each document's changes for meaning and notation.
  - The PDFs were checked page by page against the Markdown by independent AI reviewers. The rendering defects they found were fixed at the source and the PDFs re-rendered: two reflection tables that did not render, a softmax formula and the symbol $k^{\ast}$ whose characters were read as emphasis, a list paragraph out of place, and the ≠ glyph.
- **Resulting understanding:** A final audit against the original requirements still finds real errors, such as a claim true for one seed written as if true for all three. The submission documents are the repository's own documents, so every number in them traces back to a committed result file.
