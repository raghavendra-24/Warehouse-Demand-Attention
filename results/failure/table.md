# Failure case: the forecast saturates during large spikes

**Scenario.** larger_spikes (shift seed 202), the 104 targets from the second hour of a spike onward. Worst window: target index 7718 (demand 1285, seed-0 forecast 398, B1 1030).

**Actual.** Mean demand 589; B1 602; model 351, 328, 308. The forecast follows y(t) with slope 0.21 (demand: 0.92).

**Evidence.**

| Measure | Value |
|---|---|
| Value gain on demand ∂g/∂z, pred_s0 | 0.058 |
| Value gain on demand ∂g/∂z, pred_s1 | 0.019 |
| Value gain on demand ∂g/∂z, pred_s2 | 0.040 |
| control: mean forecast / mean ceiling / mean demand (seed 0) | 232 / 583 / 296 |
| control: forecasts within 5% of the ceiling | 0% |
| larger_spikes: mean forecast / mean ceiling / mean demand (seed 0) | 351 / 585 / 589 |
| larger_spikes: forecasts within 5% of the ceiling | 0% |

**Dose-response** (control series, 104 spike windows; each spike's excess over its no-event level scaled by k):

| k | Expected demand | B1 | Model (seeds 0, 1, 2) | Seed 0: attention mass on spike tokens | Seed 0: spike tokens' value level | Seed 1: mass / value level | All-token ceiling (seed 0) |
|---|---|---|---|---|---|---|---|
| 0 | 95 | 97 | 94, 94, 94 | 0.089 | 316 | 0.111 / 255 | 581 |
| 0.5 | 196 | 199 | 154, 156, 142 | 0.064 | 322 | 0.113 / 257 | 582 |
| 1 | 296 | 302 | 232, 223, 193 | 0.070 | 328 | 0.137 / 259 | 583 |
| 1.5 | 397 | 404 | 285, 270, 237 | 0.095 | 334 | 0.160 / 261 | 584 |
| 2 | 497 | 507 | 321, 302, 274 | 0.130 | 340 | 0.175 / 263 | 585 |
| 3 | 698 | 712 | 366, 343, 333 | 0.195 | 352 | 0.184 / 266 | 587 |
| 4 | 899 | 917 | 384, 367, 374 | 0.311 | 364 | 0.175 / 270 | 590 |
| 8 | 1704 | 1738 | 411, 410, 452 | 0.785 | 412 | 0.107 / 286 | 605 |
| 16 | 3312 | 3380 | 530, 435, 490 | 0.999 | 508 | 0.045 / 317 | 655 |

**Explanation.** The learned value path carries almost no demand magnitude: a token's projected value g changes by only 0.02–0.06 per standardised unit of demand, so values are set mostly by the calendar features. Because the readout is a convex combination with no residual path (A-06), the forecast can rise only by moving attention between tokens, a bounded route that saturates. Seed 0 moves attention onto the spike tokens (mass → 1 as k grows) and its forecast converges to their value level; seed 1 moves attention elsewhere. Both plateau far below demand, while B1 tracks it.

**Refuted explanations (kept for the record).** (1) Large spikes push attention away from the spike tokens: refuted, the mass on them rises for seed 0. (2) The all-token convex-combination ceiling binds: refuted, forecasts stay far below it (no forecast within 5%).

**Potential improvement.** Give demand magnitude a direct path to the output (a residual connection, or y(t) as an input to the head), so attention chooses where to look and the value carries how much; or train with heavier-tailed spikes. Not implemented (§20: a perfect fix is not required).

## Failure-mode slices (MAE, orders/h; model = mean of 3 seeds)

| Slice | Count | model | B1 | B2 | B3 | B4 |
|---|---|---|---|---|---|---|
| F1 spike onset hour (control) | 26 | 211.5 | 211.4 | 209.8 | 205.2 | 208.0 |
| F1 spike hours 1-2 (control) | 52 | 73.4 | 59.4 | 178.9 | 194.9 | 192.9 |
| F2 echo: 24-29 h after a spike starts (control) | 125 | 11.7 | 18.0 | 40.3 | 197.4 | 10.5 |
| F3 Saturday/Monday 00-11 h, normal (control) | 1162 | 10.2 | 15.4 | 34.3 | 24.1 | 10.7 |
| F3 reference: Tue-Thu 00-11 h, normal (control) | 1602 | 9.8 | 15.2 | 35.7 | 13.4 | 9.7 |
| F5 day type differs between t+1 and t-23 (Sat, Mon), normal | 2261 | 11.9 | 18.3 | 40.5 | 25.4 | 12.4 |
| F5 reference: same day type (Tue-Fri, Sun), normal | 5402 | 11.8 | 18.2 | 40.8 | 16.4 | 11.4 |

F4: in-spike MAE grows ×2.86 for the model and ×2.06 for B1 when spikes are doubled (130 in-spike targets).
