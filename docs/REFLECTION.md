# Reflection

The eleven §26 questions. The answers marked *(candidate)* are written by the candidate in their own words. The notes under them are facts taken from the results and the debugging journal, to draw on. Questions 2, 3, 6, 9 and 10 are answered from the evidence. Every claim points to [RESULTS.md](RESULTS.md), [DEBUGGING.md](DEBUGGING.md) or [AI_LOG.md](AI_LOG.md).

---

### 1. What did you initially expect? *(candidate)*

Notes, from the hypotheses as written in Phase 0 before any result:
- Attention would solve associative recall, and the uniform control would not (H1).
- Unscaled attention at d_k = 64 would start saturated and train worse (H2–H4).
- On the warehouse data, the model would beat the best simple baseline by 5–20% (H6).
- Most of that gain would come from the hours after events (H7).
- Attention would concentrate on the same hour yesterday and the latest hour (H8).

### 2. Which hypotheses were correct?

| ID | Outcome | Evidence |
|---|---|---|
| H1 | Confirmed | held-out accuracy 0.9999 (attention) vs 0.1249 (uniform control, = 1/8) |
| H2 | Confirmed | initial entropy ÷ ln n: 0.15 for unscaled at d_k = 64, ≥ 0.82 for scaled |
| H3 | Confirmed | at d_k = 64, 11–15% of unscaled readout rows have < 1% of the scaled median gradient; p90/p10 is 1,270–4,250 vs 3–4 |
| H4 | Confirmed | 2.07× the steps at d_k = 64, and one unscaled seed never reaches 95%; ratio 0.95 at d_k = 4 |
| V1 | Confirmed | every gradient entry agrees under A-17; 0 unexplained |
| H5 (as amended before commit) | Confirmed | test MAE B1 20.59 < B3 24.91 < B2 43.41; B2/B3 = 1.74; B3 Monday ratio 1.98 |
| H9, H10, H11 | Confirmed | higher noise: B3 ×1.72, B1 ×1.75, and the model rises less than B3; larger spikes: B1 beats the model in the first 3 h (230 vs 324–343); combined increase additive within 1% |
| F1, F4 | Confirmed | onset error ≈ spike size for every forecaster; in-spike MAE grows ×2.86 for the model vs ×2.06 for B1 |
| F2 | Confirmed for B3 | B3's echo error is 197 orders/h, 24–29 h after a spike; the model does not have the echo (11.7) |

### 3. Which hypotheses were wrong?

| ID | Outcome | What happened |
|---|---|---|
| H6 | Refuted | Against B4 (the reference chosen on validation), the gain is 5.6%, 3.5% and −0.5%: not 5–20%, and not the same sign in every seed. Amendment 3 had predicted this before the test run. |
| H7 | Refuted | The gain over B3 is large in normal hours too (≈ 35%, not < 10%). The model learns the calendar pattern, so it beats B3 everywhere, not mainly after events. |
| H8 | Refuted | The weight on t−23 and t together is 6–9%, not ≥ 50%. Attention spreads over earlier positions, peaking around 19–21 hours before t. |
| V2 | Partly refuted | Naive float32 softmax can be **finite and wrong**: at [88.5, 87.5, 0] each exp is finite, but the sum overflows. |
| F3, F5 | Weak / not observed | The model's error on weekday ↔ weekend transitions is close to its error on other days (10.2 vs 9.8; 11.9 vs 11.8). |
| H5 (first draft) | Wrong, corrected before commit | The draft predicted B3 < B1. A design calculation showed the Monday step and the spike echo make B3 worse, and H5 was revised before any code existed. |
| §19 verdict | Inconclusive | The model's absolute MAE increase (+14.8 to +15.1) matches B4's (+14.9). |

### 4. What surprised you? *(candidate)*

Notes:
- The warehouse model reads demand level through the attention **scores**, not the **values**: its value path changes by only 0.02–0.06 per standardised unit of demand.
- The hour-of-week mean (B4) is very hard to beat.
- A symmetric tiny example hid a K Qᵀ bug from every test (debugging episode 1).
- One unscaled seed at d_k = 64 never learned at all.

### 5. What was the hardest implementation problem? *(candidate)*

Notes: the four debugging episodes in [DEBUGGING.md](DEBUGGING.md); the explanation of the failure case, which took two refuted hypotheses before the right one.

### 6. What failure did you investigate?

During large spikes, the forecast saturates far below demand. On the doubled-spike series it predicts about 350 orders/h when demand is about 590; B1 is close to demand. The cause is that the value path carries almost no demand magnitude, so the convex-combination readout (no residual path, A-06) can raise the forecast only by moving attention between tokens. That route saturates.

The evidence:
- a dose-response experiment, with the spike scaled from k = 0 to 16 in the same windows;
- the measured value gain;
- the attention mass on spike tokens.

Two earlier explanations were refuted on the way: attention being pushed away from spikes, and the all-token ceiling binding. The full write-up is in RESULTS.md (Failure investigation).

### 7. What do you now understand better? *(candidate)*

Notes:
- why √d_k matters for optimisation but not for capacity;
- why a gradient check cannot catch a forward-pass bug;
- why "finite" does not mean "correct" in floating point;
- how a convex-combination readout limits what attention can output.

### 8. What remains uncertain? *(candidate)*

Notes:
- Why training produces such a small value gain on demand.
- Whether a residual path would fix the saturation.
- Whether the results hold for other data seeds (only one data seed was used, A-16).
- How much of the model's small edge over B4 comes from the calendar features rather than from the attention weighting.

### 9. Where did AI assistance help?

See [AI_LOG.md](AI_LOG.md). AI tools drafted the requirements analysis, the assumptions, the hypotheses, the architecture and plan, the code, and these documents. They also ran independent reviews that found real defects:
- the symmetric tiny example hiding K Qᵀ;
- the "finite means correct" claim in V2;
- the relative §19 rule mislabelling a perfect forecaster;
- the one-spike shift series;
- the missing hour-of-week baseline.

### 10. Where did you need to correct or reject AI-generated suggestions?

From the log, AI suggestions were corrected or rejected at least these times:
- The first draft of H3 assumed that unscaled attention shrinks gradients. A simulation showed they become uneven instead, and H3 was rewritten.
- The first draft of H5 predicted the wrong order of baselines.
- The first two explanations of the failure case were refuted by experiment.
- Several reviewer claims were refuted by the verifiers. One example: the claim that paired ablation arms start from different weights, which an existing test disproves.
- The first stability grid missed the 88.5 band.

*(Candidate: add your own corrections and rejections.)*

### 11. If given one additional day, what would you investigate? *(candidate)*

Notes:
- Add a residual path for demand magnitude and rerun the failure dose-response.
- Repeat the warehouse comparison over several data seeds.
- Test whether a longer window (a full week) lets attention match B4 without calendar features.

---

## Demonstrated vs believed

| Claim | Status | Evidence |
|---|---|---|
| The attention core computes §7 correctly | Demonstrated | hand-computed tiny example; element-by-element reference; five planted bugs caught; gradient check with 0 unexplained entries |
| Attention is what solves the toy task (H1) | Demonstrated | the uniform control stays at 1/8 under the same budget |
| Scaling matters for optimisation, not capacity (H2–H4) | Demonstrated | ablation table; the unscaled = rescaled-W_Q test |
| The warehouse model beats the best simple baseline (H6) | Not demonstrated | the MAE gain over B4 is not the same sign in every seed |
| The model's advantage comes from attention weighting rather than calendar inputs (A-27) | Unresolved | it beats the uniform control by about 3×, but that control also loses the calendar signal; B4, which uses the calendar only, is almost as good |
| The model learned structure rather than adapting to the training distribution (§19) | Unresolved | inconclusive verdict |
| The spike failure is caused by the value path's small demand gain | Demonstrated for this model | dose-response experiment, value gains, attention mass |
| A residual path would fix it | Believed | not tested |
