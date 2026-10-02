# Acceptance criteria

Pass/fail criteria the implementation is checked against (FAC-01 … FAC-69). Sources are PRD sections and the assumptions in [ASSUMPTIONS.md](ASSUMPTIONS.md). Prepared with AI assistance (see [AI_LOG.md](AI_LOG.md)); "RJ" marks a criterion that needs reviewer judgement, with an objective proxy.

**How to read this list**
- **MUST** rows test a PRD requirement; a submission that fails one is incomplete. **SHOULD** rows test a PRD "should", a §31 SHOULD item or an assumption's plan item.
- The only thresholds are the PRD's own counts and values fixed in ASSUMPTIONS.md: the A-12 split weeks, A-16's three seeds, A-17's gradient tolerance, A-19's d_k = 4 vs 64, and A-20's four shift conditions.
- **RJ** (reviewer judgement) marks a row where the PRD itself calls for judgement. The objective proxy in the row is the pass condition.
- The Definition of Done at the end lists the PRD §29 deliverables and the §30 demo parts.

## 1. Phase 0

| FAC | Criterion (objective, observable) | Evidence | Source | Priority |
|---|---|---|---|---|
| FAC-01 | The Phase 0 document has eight headed sections matching §5 items 1–8: formulation, dataset design, statistical assumptions, hypotheses, failure modes, evaluation methodology, expected baseline behaviour, planned generalisation experiment. Each section holds content, not "see code" | Headings in the Phase 0 file | §5 | MUST |
| FAC-02 | The formulation defines, in symbols: the input window y(t−23) … y(t), the target y(t+1), the model ŷ = f_θ(x) with θ listed, and the training loss | Phase 0, item 1 | §5(1), §15, A-04, A-14 | MUST |
| FAC-03 | At least 2 hypotheses. Each has an ID, a prediction, a metric, an expected direction and a "refuted if …" condition; magnitude is optional. Together they include the ablation hypothesis and the expected shift impact. A statement with no refutation condition ("attention may help") does not count | Hypotheses section | §5(4), §2.3; §14 "state the hypothesis before running the experiment"; §19 "expected impact" | MUST |
| FAC-04 | At least 3 failure modes. Each names the generator term or model choice involved, the mechanism and an observable symptom. RJ ("system-specific"); proxy: all three fields filled for every mode | Failure-modes section | §5(5) | MUST |
| FAC-05 | *Evaluation methodology* gives the split boundaries, the metric formulas and units, and the role of each split: train fits, validation selects, test is scored once. *Expected baseline behaviour* gives each baseline's formula and its predicted ranking or weakness. *Planned generalisation experiment* gives the parameters to change and the expected direction of the effect | Phase 0, items 6–8 | §5(6)–(8), §18, A-12, A-14 | MUST |
| FAC-06 | In the Git history, the commit that adds the hypotheses and metrics comes before the first commit containing any experiment result: toy, ablation, warehouse or baseline scores on validation or test, or shift | `git log` | §2.3, §5, A-22 | MUST |
| FAC-07 | After that commit, the hypothesis text is unchanged. Every later Phase 0 change is a dated amendment (what, when, why) | `git diff` from the FAC-06 commit to the final one | §5 "should reflect the actual implementation"; A-22 | SHOULD |

## 2. Data generator

| FAC | Criterion (objective, observable) | Evidence | Source | Priority |
|---|---|---|---|---|
| FAC-08 | Every dataset (train, validation, test, shift series, toy data) is produced by code in the repository from a config and a documented seed. No external data file or network read; no real warehouse, customer or employee data | Code; search for file and network reads | §3, §4, §31, A-16 | MUST |
| FAC-09 | The generator implements the components chosen in A-08 to A-11: daily and weekly seasonality, negative-binomial count noise, spikes and drops, no trend. Each has a one-line reason. RJ ("justified"); proxy: one reason per component | Generator code; Phase 0 dataset design | §4 "guidance rather than a mandatory schema", "should be designed and justified"; A-08 to A-11 | SHOULD |
| FAC-10 | The §6 documentation has all five headings, with these contents:<br>• Temporal structure: hourly, daily and weekly patterns; trend = none (A-10).<br>• Noise: distribution, approximate magnitude in orders/hour, and whether variance is constant or level-dependent.<br>• Unusual events: onset rule, rate, magnitude and duration; not model inputs (A-11).<br>• Correlation: which lags are expected to be informative and why, with the measured autocorrelation at lags 1, 24 and 168 next to the expectation.<br>• Distribution shift: points to FAC-11.<br>Parameter values match the shipped config or a dated amendment | Dataset section | §6 "must explicitly define"; §5(2)–(3); A-09 to A-11 | MUST |
| FAC-11 | For each shift condition, a table of parameter / original value / shifted value / reason | Shift-definition table | §6 "Define how the generalization test will alter the data distribution"; §19; A-20 | MUST |

## 3. Attention core

| FAC | Criterion (objective, observable) | Evidence | Source | Priority |
|---|---|---|---|---|
| FAC-12 | Q = XW_Q, K = XW_K, V = XW_V, S = QKᵀ, S′ = S/√d_k, A = softmax(S′) and Y = AV each appear as a separate, named statement. The derivation has a seven-row table: operation → formula → source location. The function can return A (and Q, K, S′) for inspection. One switch changes S′ = S/√d_k to S′ = S and changes nothing else | Source; derivation table | §7 "should make these operations identifiable in the source code"; §14; §16 | MUST |
| FAC-13 | A search of the source finds no use of `nn.MultiheadAttention`, `F.scaled_dot_product_attention`, `nn.Transformer*` or a third-party attention library. The softmax inside the core is hand-written | Search output (a test or the README) | §2.1, §7 "must not use high-level attention or Transformer implementations"; A-02 | MUST |
| FAC-14 | The forward pass matches values computed by hand for a tiny float64 example in which n, d_model, d_k and d_v all differ. Removing 1/√d_k makes this test fail | Test; worked example in the derivation | §2.2 "comparison against independently calculated results"; A-17 | SHOULD |
| FAC-15 | With a non-constant input and a non-degenerate loss, one backward pass gives finite, non-zero gradients for W_Q, W_K and W_V, and all three matrices change after one optimiser step | Test | §12 "participates in an actual learning process" (INFERRED) | MUST |

## 4. Derivation

| FAC | Criterion (objective, observable) | Evidence | Source | Priority |
|---|---|---|---|---|
| FAC-16 | The derivation has one section per §8 topic and covers:<br>• what Q, K and V represent, and why there are three projections rather than one shared one;<br>• why q_i·k_j measures how well position j matches what position i is looking for;<br>• how softmax turns logits into weights: exponentiation makes them positive, normalising each row makes it sum to 1, the order is preserved, and the scale acts as a temperature;<br>• what each row of AV is: a weighted average of the value rows.<br>RJ (correctness); proxy: every listed point is present | Derivation document | §8; §31 "Mathematical derivation" | MUST |
| FAC-17 | The scaling section gives the chain:<br>1. With independent zero-mean components of variance σ², Var(q·k) = d_k·σ⁴.<br>2. So the logit spread grows like √d_k.<br>3. So the softmax approaches one-hot.<br>4. Dividing by √d_k makes the variance independent of d_k.<br>The effect on gradients is optional context for §13 | Derivation document | §8 "connect dimensionality with the magnitude of the dot product and the behavior of softmax" | MUST |
| FAC-18 | A shape table lists X, Q, K, V, QKᵀ, A and Y both symbolically (n×d_model, n×d_k, n×d_k, n×d_v, n×n, n×n, n×d_v, plus the batch dimension if one is used) and at the actual toy and warehouse sizes. It agrees with the FAC-54 test | Derivation document | §8 "Document the dimensions of the major tensors" | MUST |

## 5. Gradient verification

| FAC | Criterion (objective, observable) | Evidence | Source | Priority |
|---|---|---|---|---|
| FAC-19 | The gradient section states:<br>• the method: autograd (Option A), cross-checked against float64 central finite differences;<br>• the loss that is differentiated, as a formula;<br>• the tensors differentiated (X, W_Q, W_K, W_V);<br>• why the loss is non-degenerate. For example, sum(A) is not used, because each row of A sums to 1, so its gradient is zero | Gradient section | §9 Option A "explain what gradient is being calculated and why"; A-17 | MUST |
| FAC-20 | A results table for each of X, W_Q, W_K and W_V, with the columns autograd gradient / numerical gradient / absolute difference / relative difference. Every entry agrees under A-17 (|g_auto − g_num| ≤ 1e-8 + 1e-6·|g_num|), or every disagreeing entry is listed with its explanation, and the table ends with "Unexplained entries: 0" | Saved table and paragraph | §9 "A representative result should include …", "If the result is not exact, explain why", "Uninvestigated discrepancies are not [acceptable]"; A-17 | MUST |

## 6. Numerical stability

| FAC | Criterion (objective, observable) | Evidence | Source | Priority |
|---|---|---|---|---|
| FAC-21 | The softmax subtracts the row maximum. The derivation shows that softmax(s − c) = softmax(s) and explains why choosing c = max(s) prevents overflow. A float32 demonstration on large logits, such as the row [1000, 999, 0], shows the naive version giving inf or NaN, and the stable version giving finite weights that sum to 1 | Derivation; saved demonstration output | §11 "must consider numerical stability"; "explain why a numerically stable softmax may subtract the maximum logit" | MUST |
| FAC-22 | At least one §11 issue (softmax saturation at large logits) is investigated with saved numbers and linked to FAC-17, for example maximum weight, row entropy and gradient norm against logit scale. Reusing the logs of the unscaled d_k = 64 ablation arm counts | Results and paragraph | §11 "should investigate at least one issue"; §31 SHOULD "Strong numerical-stability analysis" | SHOULD |

## 7. Toy task

| FAC | Criterion (objective, observable) | Evidence | Source | Priority |
|---|---|---|---|---|
| FAC-23 | The toy task is stated: input format, target, loss, metric, a seeded data generator and a held-out set (A-18, associative recall). The toy model calls the shared attention function | Toy section; code | §12 "The candidate may design the task", "clear learning objective"; A-18; A-26 | MUST |
| FAC-24 | The toy report contains all five §12 items:<br>1. loss curves, training and held-out;<br>2. training behaviour described with numbers, such as steps to a stated accuracy or a plateau;<br>3. the held-out metric next to a stated trivial reference;<br>4. the model configuration: sizes, initialisation, optimiser, learning rate, batch size, steps, seeds;<br>5. a paragraph of observations | Toy section | §12 "must report: loss; training behavior; relevant evaluation metric; model configuration; observations" | MUST |
| FAC-25 | The uniform-attention control (A-18) is trained with the same budget, and its held-out accuracy is reported next to the full model's over the seeds declared in Phase 0 (default A-16: 3). Whether the full model beats it is a hypothesis outcome, not a pass condition | Toy table | §12; A-18; A-16 | SHOULD |

## 8. Training dynamics

| FAC | Criterion (objective, observable) | Evidence | Source | Priority |
|---|---|---|---|---|
| FAC-26 | One phenomenon is named (default A-19: softmax saturation in unscaled attention). It is explained as a chain from observation to mechanism to effect on learning, in the form of §13's example: large logits → concentrated softmax → small gradients → changed learning. A description of the curve alone fails. RJ (correctness); proxy: the chain is present | Dynamics section | §13 "must go beyond 'The graph became unstable'", "connect the observation to the underlying mathematics"; §31; A-19 | MUST |
| FAC-27 | Each link in that chain cites a quantity logged in the candidate's own runs: attention entropy, maximum weight, logit spread, or W_Q/W_K gradient norms against step | Logged data; plot | §13 "should be supported by the candidate's experiment"; A-19 | SHOULD |

## 9. Ablation

| FAC | Criterion (objective, observable) | Evidence | Source | Priority |
|---|---|---|---|---|
| FAC-28 | The scaled and unscaled arms differ in exactly one factor, the FAC-12 scaling switch. For each seed, the two arms share data, initial weights and batch order. A config diff shows this | Configs | §14 "controlled"; "Compare: softmax(QKᵀ/√d_k) against: softmax(QKᵀ)" | MUST |
| FAC-29 | The grid is the one declared in Phase 0 (default A-19: d_k = 4 and 64 × 2 arms × 3 seeds = 12 runs). The optimiser, learning rate and unit-variance Q/K initialisation are fixed in advance. Any reduction is a dated amendment, not a silent cut | Phase 0; configs; results | §14; A-19; A-16 | MUST |
| FAC-30 | The ablation write-up has six headings:<br>1. Hypothesis, quoted from Phase 0.<br>2. Experimental setup.<br>3. Observed behaviour: a paragraph per arm.<br>4. Quantitative results per arm and d_k: mean ± SD of final loss and accuracy, steps to the target accuracy, initial entropy, and W_Q/W_K gradient norms.<br>5. Comparison with the hypothesis: supported, refuted or inconclusive.<br>6. Mathematical explanation, including that unscaled attention with W_Q equals scaled attention with √d_k·W_Q, so the arms differ in optimisation, not in what they can represent | Ablation section | §14 "The final analysis must contain:" items 1–6 | MUST |

## 10. Warehouse model

| FAC | Criterion (objective, observable) | Evidence | Source | Priority |
|---|---|---|---|---|
| FAC-31 | The model predicts y(t+1) from y(t−23) … y(t). Each hour is a token of standardised demand plus sin/cos hour-of-day and day-of-week features (A-05). The tokens pass through the shared attention function (A-26), and a linear head reads the last position (A-06). On a ramp series y(t) = t, a test confirms that the window for target τ holds exactly τ−24 … τ−1. Every adaptation of the §16 pipeline is listed | Code; test; model section | §15; §16; A-04 to A-06; A-26 | MUST |
| FAC-32 | Training uses only training-split targets. Early stopping, hyperparameters and the reference baseline are chosen on validation only. The test split is scored once, after the configuration is frozen | Code; run log | §18 leakage rule (INFERRED); A-12 | MUST |
| FAC-33 | One attention-weight figure: the readout row averaged over test windows, with the x-axis running t−23 … t, plus a typical, an event and a worst-error example (A-06). The caption states that the weights are descriptive, not causal | Figure and caption | §16 "may be inspected", "should distinguish between visualization and causal claims"; §31 SHOULD | SHOULD |

## 11. Baselines

| FAC | Criterion (objective, observable) | Evidence | Source | Priority |
|---|---|---|---|---|
| FAC-34 | At least one §17 baseline is implemented; the A-15 plan has three: last observation y(t), the 24-hour moving average, and seasonal naive y(t−23). On a ramp series y(t) = t, predicting y(24) from y(0) … y(23) gives 23, 11.5 and 0 respectively | Code; test | §2.4 "at least one simple baseline"; §17; A-15 | MUST |
| FAC-35 | The baselines and the attention model are scored on identical test target indices, with the same metric code, on the original scale. The table has one row per metric (MAE, RMSE) and the columns Baseline / Attention model / Difference, where Difference = model − baseline in orders/hour. The attention model is reported as mean ± SD over the training seeds declared in Phase 0 (default A-16: 3) | Results table | §17 "same test set", "Report: Baseline / Attention model / Difference"; A-14; A-16 | MUST |
| FAC-36 | The reference baseline is the best one on validation (A-15, A-12). The results:<br>• state whether attention adds value over it, without claiming differences smaller than the spread between seeds (A-16);<br>• explain the outcome with evidence: errors by A-11 group or by hour of day, and attention weights;<br>• compare the observed baseline ranking with Phase 0's expected behaviour.<br>Passing does not depend on beating the baseline. RJ ("correctly explains why"); proxy: a verdict and its evidence are present | Results section | §1 "whether attention actually provides value over a simple baseline"; §2.4; §17 "should explain the result"; §5(7) | MUST |

## 12. Evaluation

| FAC | Criterion (objective, observable) | Evidence | Source | Priority |
|---|---|---|---|---|
| FAC-37 | The split is chronological by target hour, in whole weeks, as declared in Phase 0 (default A-12: weeks 1–36 / 37–44 / 45–52). A test asserts that no training target falls in a later split. The split strategy is explained in writing | Test; text | §18 "must explain the split strategy", "avoid random splitting that leaks future information"; A-12 | MUST |
| FAC-38 | Standardisation statistics come from the training split only and are applied unchanged to the validation, test and shifted data. Reported metrics are in orders/hour | Code; test | §18 leakage rule (INFERRED); A-13 | MUST |
| FAC-39 | MAE (primary) and RMSE (secondary) are defined in Phase 0 before any result (FAC-06), and the headline results use exactly these. A paragraph explains why they suit this data: spikes inflate RMSE, and MAPE is unstable when night-time demand is near zero. Any metric added later is labelled post hoc | Phase 0; results | §5(6); §18 "Metrics should be defined before experiments", "should explain why the selected metric is appropriate"; A-14 | MUST |
| FAC-40 | Errors are reported in the three A-11 groups, each with its count: target inside an event; target up to 24 h after an event; all other hours | Results table | A-11; A-14 | SHOULD |

## 13. Generalisation

| FAC | Criterion (objective, observable) | Evidence | Source | Priority |
|---|---|---|---|---|
| FAC-41 | The shift evaluation loads the trained models without retraining and reuses the training standardisation. The shift series are freshly generated by the same generator (A-20: 8-week series with the test calendar, a 24-hour warm-up and one shared seed) | Code; run log | §19 "evaluated under a changed statistical distribution"; the §6 Training/Testing example; A-13; A-20 | MUST |
| FAC-42 | The attention model and every baseline are scored on at least the control (unchanged parameters) and one shifted condition (default: higher noise plus larger spikes, §6's example), in one table that shows the change from the control | Shift table | §19; §31 "One distribution-shift/generalization experiment"; A-20 | MUST |
| FAC-43 | The two single-factor A-20 conditions (higher noise only, larger spikes only) are in the same table, so the combined effect can be attributed | Shift table | A-20 | SHOULD |
| FAC-44 | The shift write-up documents the five §19 items:<br>1. original distribution;<br>2. modified distribution;<br>3. expected impact, quoted from Phase 0;<br>4. observed impact;<br>5. mathematical interpretation.<br>It ends with an explicit verdict on §19's question, "learned useful structure or simply adapted to the exact training distribution", citing the deciding numbers: for example, the model's error relative to the baselines under the control and under the shift | Shift section | §19 "The experiment must document:", "This should test whether …" | MUST |

## 14. Failure investigation

| FAC | Criterion (objective, observable) | Evidence | Source | Priority |
|---|---|---|---|---|
| FAC-45 | At least one failure case, written up under six headings:<br>1. Scenario, reproducible from the dataset, seed and window index.<br>2. Expected behaviour.<br>3. Actual behaviour, with numbers.<br>4. Explanation.<br>5. Evidence: an experiment or mathematical reasoning.<br>6. Potential improvement.<br>The case comes from the model's real behaviour, in normal evaluation or on extreme inputs the generator can produce (A-21). Per-window predictions are saved, so the case can be re-examined without retraining | Failure section; saved predictions | §20 "Identify at least one genuine failure case", "What experiment or mathematical reasoning supports your explanation?"; §2.5; A-21 | MUST |
| FAC-46 | The explanation is tested by at least one targeted experiment, such as removing the spike from the window and predicting again. The case is linked to a Phase 0 failure mode, or the write-up says no mode predicted it | Failure section | A-21; the §5 comparison chain "Phase 0 prediction → … → Final explanation" | SHOULD |

## 15. Cross-problem integration

| FAC | Criterion (objective, observable) | Evidence | Source | Priority |
|---|---|---|---|---|
| FAC-47 | There is exactly one attention implementation, and both the toy model and the warehouse model call it (one definition, two call sites). The main warehouse model is not an off-the-shelf forecaster | Code search | §15 "should be based on the implementation from Problem 1"; §21 "must genuinely use", "A completely disconnected Problem 2 … does not satisfy"; A-26 | MUST |
| FAC-48 | An integration section:<br>• states the §21 assumption;<br>• names at least one Problem 2 choice justified by a Problem 1 result, for example that standardised inputs and √d_k scaling keep the logits of order one;<br>• explains what the warehouse model learns from the history (FAC-33);<br>• gives a verdict on whether the model's behaviour supported the assumption | Integration section | §16 "should be able to explain what the model is learning"; §21 "should document the connection explicitly", "should discuss whether the actual model behavior supported this assumption"; §35 | MUST |

## 16. Reproducibility

| FAC | Criterion (objective, observable) | Evidence | Source | Priority |
|---|---|---|---|---|
| FAC-49 | The README documents the seven §22 items:<br>1. the Python version (3.12, A-01) and framework versions;<br>2. dependencies, with exact pins in a dependency file;<br>3. a seed table that gives each seed's purpose;<br>4. dataset-generation parameters;<br>5. the model configuration;<br>6. the training configuration;<br>7. the evaluation commands | README; dependency file | §22 "Document:", "Where randomness is intentionally used, explain its purpose"; A-01; A-16 | MUST |
| FAC-50 | Each §3 stage has a documented command that runs, once its inputs exist, without manual edits. The stages are dataset generation (which is also the reset/regeneration mechanism), training, evaluation, ablation, generalisation and failure analysis | README; commands | §3; §4 "A reset/regeneration mechanism should be provided"; §22 | MUST |
| FAC-51 | From a fresh clone with the pinned dependencies, one documented command regenerates the dataset and the primary results. The README says what reproduces exactly (the generated data) and what reproduces within a stated tolerance (trained metrics, A-24) | Fresh-clone run | §22 "must be reproducible", "should allow a reviewer to regenerate the primary results"; §31 SHOULD runner; A-24 | MUST |
| FAC-52 | Every reported number and figure comes from a committed result file (a table, a metrics file or a figure) | Result files in the repository | §29(15) "Experiment results"; §31 SHOULD "Reproducible result generation" | MUST |

## 17. Tests

| FAC | Criterion (objective, observable) | Evidence | Source | Priority |
|---|---|---|---|---|
| FAC-53 | One documented test command runs the suite, and the suite passes. The README maps each of the five §23 minimum items to a named test | Test output; README | §23 "At minimum, test:" | MUST |
| FAC-54 | Shape test, with n, d_model, d_k and d_v all different: Q and K are n×d_k, V is n×d_v, S and A are n×n, and Y is n×d_v (plus a batched variant if batching is used). The warehouse model returns one prediction per window | Test | §23 "attention tensor shapes", "attention output dimensions" | MUST |
| FAC-55 | Softmax test on random, non-constant scores: every row of A sums to 1 within floating-point tolerance, and every entry is ≥ 0. A softmax over the wrong axis fails it; constant scores would hide that bug | Test | §23 "softmax normalization" | MUST |
| FAC-56 | Gradient test: float64 central differences against autograd, asserting that every entry agrees under A-17 (|g_auto − g_num| ≤ 1e-8 + 1e-6·|g_num|) | Test | §23 "gradient verification"; §9; A-17 | MUST |
| FAC-57 | Determinism test: the same seed and config produce identical series (exact equality), and a different seed produces a different series | Test | §23 "deterministic dataset generation when a seed is fixed"; §4; A-24 | MUST |
| FAC-58 | Extra debugging tests, each labelled with the bug it targets. At minimum, permutation equivariance of the core: permuting the rows of X permutes the rows of Y in the same way. The tests from FAC-14, FAC-31, FAC-34 and FAC-37 also count here | Tests | §23 "Additional tests are encouraged", "subtle mathematical bugs"; §31 SHOULD | SHOULD |

## 18. Debugging evidence and honesty

| FAC | Criterion (objective, observable) | Evidence | Source | Priority |
|---|---|---|---|---|
| FAC-59 | One debugging episode documented in §24's six steps: initial implementation, unexpected result, hypothesis about the bug, diagnostic experiment, correction (with its commit), verification (the test that now passes) | Debugging section; commit | §24 "should document at least one debugging process" | SHOULD |
| FAC-60 | A "Known issues and limitations" section lists:<br>• the unresolved problems;<br>• each skipped or expected-failure test, with its reason;<br>• any gradient discrepancy still open (FAC-20);<br>• the calendar-feature confound: the attention model sees hour and day features that the baselines do not | README or results | §24 "Do not hide meaningful implementation problems from the final report"; §26 "encouraged to report unresolved issues"; §32 "Honest" | MUST |

## 19. AI assistance log

| FAC | Criterion (objective, observable) | Evidence | Source | Priority |
|---|---|---|---|---|
| FAC-61 | One AI-log entry per material use, each dated and filled in with the six §25 fields: tool, task requested, generated output (a summary or an excerpt), candidate modifications, verification performed, resulting understanding.<br>• A field that does not apply says "n/a" and why.<br>• Where an entry touches maths or code, "verification" names an objective check: a test name, the gradient table, or a derivation step.<br>• "Resulting understanding" is in the candidate's own words | AI-log file | §2.6; §25 | MUST |
| FAC-62 | There are entries for:<br>• the AI-produced requirements analysis and its review;<br>• the drafting of ASSUMPTIONS.md;<br>• the AI-drafted Phase 0 hypotheses, recording the candidate's changes and acceptance;<br>• each AI-generated code module or document | AI-log file | §2.6 "Material AI assistance must be documented"; A-23 | MUST |

## 20. Reflection

| FAC | Criterion (objective, observable) | Evidence | Source | Priority |
|---|---|---|---|---|
| FAC-63 | The reflection gives eleven numbered answers, to §26 Q1–Q11:<br>• Q2–Q3 classify every hypothesis ID as correct, wrong or inconclusive, with a pointer to its result.<br>• Q6 names the FAC-45 failure.<br>• Q9–Q10 cite AI-log entries; Q10 covers AI suggestions that were rejected or corrected | Reflection document | §26 "must include a reflection covering:" | MUST |
| FAC-64 | A "demonstrated vs believed" table with one row per hypothesis ID and per headline claim. Each row gives a status (demonstrated, believed or unresolved) and the evidence it rests on | Reflection or results | §1 "distinguish what you have demonstrated from what you merely believe"; §35 | SHOULD |

## 21. Submission

| FAC | Criterion (objective, observable) | Evidence | Source | Priority |
|---|---|---|---|---|
| FAC-65 | One Git repository holds all 18 §29 deliverables and every project document: Phase 0, derivation, results, reflection, AI log, demo and assumptions. The README maps each §29 item to its path | Repository; README map | §29 "Submit a single repository/archive", "Include all project documents in the repository" | MUST |
| FAC-66 | The README covers:<br>• purpose;<br>• setup (Python 3.12, pinned install);<br>• the test command;<br>• the per-stage and run-all commands;<br>• a headline results table;<br>• the deliverables map;<br>• key decisions with their reasons, pointing to ASSUMPTIONS.md and Phase 0;<br>• known issues (FAC-60);<br>• the measured runtime and hardware, with no limit;<br>• a "deliberately not built" list with reasons (§27, §28) | README | §29(2); §22; §27; §2.7; §33 "Candidates should explain important decisions" | MUST |
| FAC-67 | The full commit history is kept, with no squashing. The final submitted commit is dated no later than 3 Oct 2026, 23:59 IST. At that time the assessors can open the repository; the candidate sets access or visibility at the end | `git log`; access check | Confirmed facts; §29; A-22 | MUST |

## 22. Demo

| FAC | Criterion (objective, observable) | Evidence | Source | Priority |
|---|---|---|---|---|
| FAC-68 | The demo instructions (or a recording) cover the seven §30 parts in order:<br>1. the attention implementation;<br>2. the gradient-verification result;<br>3. the toy task's learning behaviour;<br>4. the attention model and baseline results;<br>5. the ablation, predicted vs observed;<br>6. the distribution shift;<br>7. one failure case, explained.<br>Each part has a command or a committed artifact, and one or two sentences on what the output shows | Demo document | §30 "must provide either a short recording or reproducible live-demo instructions", "should show", "prioritize technical understanding over presentation polish" | MUST |
| FAC-69 | The fast parts (the attention forward trace and the gradient check) run live from the instructions. Long runs are shown from committed results. A short recording is made only if time remains | Demo document | §30 "reproducible live-demo instructions"; A-25 | SHOULD |

**Totals.** 69 criteria: 55 MUST and 14 SHOULD.

## 23. Coverage: PRD section → FAC

| PRD § | FACs | PRD § | FACs |
|---|---|---|---|
| §1 central questions | 36, 64 | §18 evaluation | 05, 32, 37–39 |
| §2.1 understand first | 13, 16 | §19 shift | 11, 41–44 |
| §2.2 verification | 14, 20 | §20 failure | 45, 46 |
| §2.3 hypotheses first | 03, 06, 07 | §21 integration | 47, 48 |
| §2.4 baselines | 34–36 | §22 reproducibility | 49–52 |
| §2.5 failure | 45 | §23 testing | 53–58 |
| §2.6 AI assistance | 61, 62 | §24 debugging | 59, 60 |
| §2.7 scope | 66 (not-built list) | §25 AI log | 61, 62 |
| §3 actors, local | 08, 50 | §26 reflection | 60, 63, 64 |
| §4 environment | 08, 09, 50, 57 | §27 not required | 66 (not-built list) |
| §5 Phase 0 | 01–07 | §28 optional | 66 (none attempted) |
| §6 statistical design | 10, 11 | §29 submission | 52, 65–67 |
| §7 attention core | 12, 13 | §30 demo | 68, 69 |
| §8 derivation | 16–18 | §31 MUST | each one appears in the DoD (G.24) |
| §9 gradients | 19, 20, 56 | §31 SHOULD | 22, 33, 35, 51, 52, 58 |
| §10 manual backprop | 66 (optional; not attempted) | §31 NICE | 66 (not attempted) |
| §11 stability | 21, 22 | §32 NFRs | 13 Controlled; 51 Reproducible; 53 Testable; 60 Honest; 66 Explainable; 47 Maintainable (RJ) |
| §12 toy task | 15, 23–25 | §33 own decisions | 66 |
| §13 dynamics | 26, 27 | §34 live review | none: not a deliverable |
| §14 ablation | 03, 12, 28–30 | §35 coherence | 48, 64 |
| §15 warehouse | 02, 31, 47 | | |
| §16 model | 31, 33, 48 | | |
| §17 baselines | 34–36 | | |

## 24. Definition of Done

**§29's 18 deliverables.** Every box must be ticked. A box is ticked when all the MUST FACs listed on it pass.

- [ ] 1. Source code: FAC-12, FAC-31, FAC-47, FAC-65
- [ ] 2. README: FAC-49, FAC-66
- [ ] 3. Phase 0 design document: FAC-01 to FAC-06
- [ ] 4. Mathematical derivation: FAC-16, FAC-17, FAC-18
- [ ] 5. Synthetic dataset generator: FAC-08, FAC-10, FAC-11
- [ ] 6. First-principles attention implementation: FAC-12, FAC-13, FAC-15, FAC-21
- [ ] 7. Gradient verification: FAC-19, FAC-20, FAC-56
- [ ] 8. Toy learning experiment: FAC-23, FAC-24
- [ ] 9. Warehouse demand model: FAC-31, FAC-32
- [ ] 10. Baseline implementation: FAC-34, FAC-35, FAC-36
- [ ] 11. Ablation experiment: FAC-28, FAC-29, FAC-30
- [ ] 12. Generalization experiment: FAC-41, FAC-42, FAC-44
- [ ] 13. Failure investigation: FAC-45
- [ ] 14. Test suite: FAC-53 to FAC-57
- [ ] 15. Experiment results: FAC-52
- [ ] 16. Reflection: FAC-63
- [ ] 17. AI assistance log: FAC-61, FAC-62
- [ ] 18. Demo instructions or recording: FAC-68

**Required items without a §29 entry.** Also gating:

- [ ] Training dynamics (§13, §31 MUST): FAC-26
- [ ] Numerical stability considered (§11 "must consider"): FAC-21
- [ ] Cross-problem integration (§21 "must"): FAC-47, FAC-48
- [ ] Reproducibility documented (§22 "must"): FAC-49 to FAC-51
- [ ] Evaluation methodology (§18): FAC-37, FAC-38, FAC-39
- [ ] No hidden problems (§24): FAC-60
- [ ] Submitted on time, and readable by the assessors: FAC-67

**§30's seven demo parts.** Each must point at a working command or a committed artifact.

- [ ] Part 1, Attention: the FAC-12 trace
- [ ] Part 2, Gradient verification: the FAC-20 table
- [ ] Part 3, Training: the FAC-24 curves and metric
- [ ] Part 4, Warehouse prediction: the FAC-35 table
- [ ] Part 5, Ablation: the FAC-30 hypothesis vs result
- [ ] Part 6, Generalization: the FAC-44 table and verdict
- [ ] Part 7, Failure: the FAC-45 case

The 14 SHOULD FACs do not block Done. Debugging evidence (§24 "should", FAC-59) is listed among them, not among the gates.
