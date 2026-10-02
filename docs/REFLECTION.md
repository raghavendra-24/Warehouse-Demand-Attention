# Reflection

The eleven §26 questions. Every factual claim points to [RESULTS.md](RESULTS.md), [DEBUGGING.md](DEBUGGING.md) or [AI_LOG.md](AI_LOG.md).

These answers were drafted with AI assistance from the project record, at the candidate's request, and are logged as such ([AI_LOG.md](AI_LOG.md), entry 11).

---

### 1. What did you initially expect?

I expected attention to be clearly useful on the warehouse data, not just on the toy task. The Phase 0 hypotheses, drafted with AI help and adopted by me, said:
- the model would beat the best simple baseline by 5–20%;
- most of that gain would come from the hours after events;
- attention would concentrate on two positions, the same hour yesterday and the latest hour.

On the toy task I expected attention to solve associative recall while a uniform-attention control failed. On the ablation I expected unscaled attention to start saturated and train worse at d_k = 64, but not at d_k = 4. I also first expected seasonal naive to beat last observation. A design calculation showed, before any code existed, that the Monday step and the spike echo make it worse, and H5 was revised.

### 2. Which hypotheses were correct?

| ID | Outcome | Evidence | RESULTS § |
|---|---|---|
| H1 | Confirmed | held-out accuracy 0.9999 (attention) vs 0.1249 (uniform control, = 1/8) | §4 |
| H2 | Confirmed | initial entropy ÷ ln n: 0.15 for unscaled at d_k = 64, ≥ 0.82 for scaled | §5, §6 |
| H3 | Confirmed | at d_k = 64, 11–15% of unscaled readout rows have < 1% of the scaled median gradient; p90/p10 is 1,270–4,250 vs 3–4 | §5, §6 |
| H4 | Confirmed | 2.07× the steps at d_k = 64, and one unscaled seed never reaches 95%; ratio 0.95 at d_k = 4 | §6 |
| V1 | Confirmed | every gradient entry agrees under A-17; 0 unexplained | §3.1 |
| H5 (as amended before commit) | Confirmed | test MAE B1 20.59 < B3 24.91 < B2 43.41; B2/B3 = 1.74; B3 Monday ratio 1.98 | §7.3 |
| H9, H10, H11 | Confirmed | higher noise: B3 ×1.71, B1 ×1.75, and the model rises less than B3; larger spikes: B1 beats the model in the first 3 h (230 vs 324–343); combined increase additive within 1.5% | §8 |
| F1, F4 | Confirmed | onset error ≈ spike size for every forecaster; in-spike MAE grows ×2.86 for the model vs ×2.06 for B1 | §9 |
| F2 | Confirmed for B3 | B3's echo error is 197 orders/h, 24–29 h after a spike; the model does not have the echo (11.7) | §9.7 |

### 3. Which hypotheses were wrong?

| ID | Outcome | What happened | RESULTS § |
|---|---|---|
| H6 | Refuted | Against B4 (the reference chosen on validation), the gain is 5.6%, 3.5% and −0.5%: not 5–20%, and not the same sign in every seed. Amendment 3 had predicted this before the test run. | §7.5 |
| H7 | Wrong in both numbers | The post-event gain over B3 is only 1.5–1.6× the normal-hour gain (predicted ≥ 2×), and the normal-hour gain is ≈ 35% (predicted < 10%). The "refuted if" clause as written (normal ≥ post-event) is not triggered. The model learns the calendar pattern, so it beats B3 everywhere, not mainly after events. | §7.5 |
| H8 | Refuted | The weight on t−23 and t together is 6–9%, not ≥ 50%. Averaged over test windows the readout row is nearly flat (2.6–6.3% per position); its small peak is at t−19/t−20 in seed 0, t−7 in seed 1 and t−9 in seed 2. | §7.5, §7.6 |
| V2 | Partly refuted | Naive float32 softmax can be **finite and wrong**: at [88.5, 87.5, 0] each exp is finite, but the sum overflows. | §3.2 |
| F3, F5 | Weak / not observed | The model's error on weekday ↔ weekend transitions is close to its error on other days (10.2 vs 9.8; 11.9 vs 11.8). | §9.7 |
| H5 (first draft) | Wrong, corrected before commit | The draft predicted B3 < B1. A design calculation showed the Monday step and the spike echo make B3 worse, and H5 was revised before any code existed. | §7.3; [PHASE0.md](PHASE0.md) H5 |
| §19 verdict | Inconclusive | The model's absolute MAE increase (+14.8 to +15.1) matches B4's (+14.9). | §8 |

### 4. What surprised you?

- **The model reads demand through the attention scores, not the values.** A token's projected value changes by only 0.02–0.06 per standardised unit of demand. The forecast rises mainly by moving attention between hours, which is also why it saturates during large spikes.
- **How strong the hour-of-week mean (B4) is.** A one-line baseline came within 3% of the attention model's MAE, and matched it in one of three seeds.
- **Where attention looked.** It did not focus on "same hour yesterday" or the latest hour. It spread its weight almost evenly, and where its small peak fell differed between seeds.
- **How easily a test can be blind.** The first hand-computed example produced a symmetric score matrix, so a transposed QKᵀ passed every test until a deliberate mutation check exposed it.
- **Saturation can stop learning altogether.** One unscaled seed at d_k = 64 never learned the toy task, which is more than "learns slower".
- **"Finite" does not mean "correct" in floating point.** At [88.5, 87.5, 0], naive softmax returns finite zeros.

### 5. What was the hardest implementation problem?

Explaining the spike failure. The symptom was clear: forecasts near 350 orders/h while demand was near 590. But the first two explanations were plausible and wrong:
1. that very large spikes push attention away from the spike hours;
2. that the readout's all-token ceiling was binding.

Separating them needed counterfactual experiments rather than more plots:
- swapping the attention weights between spike sizes;
- breaking each forecast into per-token contributions;
- a dose-response that scales the spike in the same windows.

A close second was making the tests sensitive enough. Planting bugs on purpose showed that passing tests are not evidence until you have seen them fail.

### 6. What failure did you investigate?

During large spikes, the forecast saturates far below demand. On the doubled-spike series it predicts about 350 orders/h when demand is about 590; B1 is close to demand. The cause is that the value path carries almost no demand magnitude, so the convex-combination readout (no residual path, A-06) can raise the forecast only by moving attention between tokens. That route saturates.

The evidence:
- a dose-response experiment, with the spike scaled from k = 0 to 16 in the same windows;
- the measured value gain;
- the attention mass on spike tokens.

Two earlier explanations were refuted on the way: attention being pushed away from spikes, and the all-token ceiling binding. The full write-up is in RESULTS.md (Failure investigation).

### 7. What do you now understand better?

- **Why √d_k matters.**
  - Unscaled attention is scaled attention with W_Q multiplied by √d_k, so it can represent the same functions.
  - The difference is optimisation: large logits make softmax near one-hot.
  - Saturation makes gradients **uneven**, not uniformly small. My first draft claimed they shrink uniformly; a simulation corrected that before the hypotheses were committed.
- **What a gradient check proves.** It verifies autograd against the same forward code. A forward bug such as KQᵀ passes it, so the forward pass needs its own independent check.
- **Floating point.** Subtracting the maximum keeps every exponent ≤ 0. The sum can overflow even when each term does not.
- **Architecture limits behaviour.** With no residual path, the output is a convex combination of the values, which bounds what attention can produce.
- **Evaluation discipline.**
  - Pre-registering predictions, scoring the test split once after a freeze, and claiming an effect only when all seeds agree.
  - A verdict rule can be wrong too: comparing relative increases would have labelled a perfect forecaster "simply adapted".

### 8. What remains uncertain?

- **Why training produces such a small value gain on demand.** It could be the rarity of spikes, MSE on standardised targets, or the calendar features explaining most of the variance. I have not separated these.
- **Whether a residual path would fix the saturation.** I believe it would, but it is untested.
- **Whether the results generalise.** Only one data seed was used, so the warehouse comparison could change with another draw of the data.
- **Whether the model's small edge over B4 is real.** It appears in 2 of 3 seeds, and how much of it comes from the attention weighting rather than the calendar inputs is unresolved (A-27).
- **What the attention weights mean.** They are descriptive, not causal.
- **The toy task's learning curve.** I did not investigate the plateau followed by a sudden drop in loss.

### 9. Where did AI assistance help?

See [AI_LOG.md](AI_LOG.md), entries 1–12. AI tools drafted the requirements analysis, the assumptions, the hypotheses, the architecture and plan, the code, and these documents. They also ran independent reviews that found real defects:
- the symmetric tiny example hiding K Qᵀ (entry 7);
- the "finite means correct" claim in V2 (entry 8);
- the relative §19 rule mislabelling a perfect forecaster (entry 9);
- the one-spike shift series (entry 9);
- the missing hour-of-week baseline (entry 9).

### 10. Where did you need to correct or reject AI-generated suggestions?

From the log, AI suggestions were corrected or rejected at least these times:
- The first draft of H3 assumed that unscaled attention shrinks gradients. A simulation showed they become uneven instead, and H3 was rewritten (entry 4).
- The first draft of H5 predicted the wrong order of baselines (entry 4).
- The first two explanations of the failure case were refuted by experiment (entry 10).
- Several reviewer claims were refuted by the verifiers. One example: the claim that paired ablation arms start from different weights, which an existing test disproves (entry 9).
- The first stability grid missed the 88.5 band (entry 8).
- Two AI agents answered a chat question instead of writing their document. Their output was discarded and the work redone (entry 10).

My own decisions on AI output:
- I chose to amend H5 before the first commit rather than keep a prediction already contradicted by our own calculation (entry 4).
- I approved three Phase 0 amendments and six design changes, but only after reading why each was needed and confirming that no warehouse or shift result existed yet (entries 8 and 9).
- I kept the repository private during development and made it public for submission. I kept AI attribution out of commit messages and documented AI use in the log instead (entries 7 and 12).

### 11. If given one additional day, what would you investigate?

1. **Give demand magnitude a direct path to the output.** Add a residual connection, or y(t) as an input to the head. Pre-register what it should change, then rerun the failure dose-response and the warehouse comparison.
2. **Repeat the warehouse comparison over 3–5 data seeds**, to see whether the edge over B4 is real.
3. **Try a 168-hour window**, which holds a full week, to see whether attention can match B4 without calendar features.
4. **Look into the toy task's plateau**, using the logged entropy and gradient histories.

---

## Demonstrated vs believed

| Claim | Status | Evidence |
|---|---|---|
| The attention core computes §7 correctly | Demonstrated | hand-computed tiny example; element-by-element reference; five planted bugs caught; gradient check with 0 unexplained entries |
| Attention is what solves the toy task (H1) | Demonstrated | the uniform control stays at 1/8 under the same budget |
| Scaling matters for optimisation, not capacity (H2–H4) | Demonstrated | ablation table; the unscaled = rescaled-W_Q test |
| Naive softmax fails only by returning inf or NaN (V2) | Refuted | at [88.5, 87.5, 0], naive float32 softmax is finite and wrong (RESULTS §3.2) |
| Seasonal naive is worse than last observation on this generator (H5, as amended) | Demonstrated | test MAE B1 20.59 < B3 24.91 < B2 43.41, as the design calculation predicted (§7.3) |
| The warehouse model beats the best simple baseline (H6) | Not demonstrated | the MAE gain over B4 is not the same sign in every seed |
| The model's gain over B3 comes mainly after events (H7) | Not demonstrated | the gain is 34.7–36.5% in normal hours too, and post-event only 1.5–1.6× that (§7.5) |
| Attention concentrates on t−23 and t (H8) | Refuted | 6.2–9.1% of the weight on those two positions, against 8.3% for uniform (§7.5) |
| The model's advantage comes from attention weighting rather than calendar inputs (A-27) | Unresolved | it beats the uniform control by about 3×, but that control also loses the calendar signal; B4, which uses the calendar only, is almost as good |
| The model learned structure rather than adapting to the training distribution (§19) | Unresolved | inconclusive verdict |
| Higher noise and larger spikes affect the forecasters as predicted (H9–H11) | Demonstrated for these four series | one shift seed, no retraining (§8) |
| Every forecaster misses a spike's first hour, and B3 echoes it 24 h later (F1, F2) | Demonstrated | onset MAE 205–212 for all; B3 echo 197 vs the model's 11.7 (§9, §9.7) |
| Weekday ↔ weekend transitions hurt the model (F3, F5) | Not observed | 10.2 vs 9.8 and 11.9 vs 11.8 (§9.7) |
| Large spikes hurt the model more than B1 (F4) | Demonstrated | in-spike MAE grows ×2.86 for the model vs ×2.06 for B1 (§9) |
| The spike failure is caused by the value path's small demand gain | Demonstrated for this model | dose-response experiment, value gains, attention mass |
| A residual path would fix it | Believed | not tested |
