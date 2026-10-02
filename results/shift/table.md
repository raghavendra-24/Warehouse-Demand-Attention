# Distribution shift: four 52-week series, shift seed 202 (reference baseline B4, chosen on validation)

| Series | Spike onsets | First-3-h spike targets | Measured noise SD near the mean level |
|---|---|---|---|
| control | 27 | 78 | 13.0 |
| higher_noise | 27 | 78 | 25.7 |
| larger_spikes | 27 | 78 | 13.4 |
| both | 27 | 78 | 25.6 |

### control: all hours (orders/h)

| Metric | Reference (B4) | Attention model | Uniform control | Difference (model − reference) |
|---|---|---|---|---|
| MAE | 14.63 | 14.20 ± 0.21 (14.08, 14.02, 14.49) | 43.32 ± 0.02 | -0.43 (-3.0%) |
| RMSE | 31.81 | 24.89 ± 0.49 (24.50, 24.60, 25.59) | 54.96 ± 0.10 | -6.92 (-21.7%) |

#### control: MAE by A-11 group

| Group | Count | B1 | B2 | B3 | B4 | pred_s0 | pred_s1 | pred_s2 |
|---|---|---|---|---|---|---|---|---|
| all | 8736 | 19.77 | 43.88 | 24.57 | 14.63 | 14.08 | 14.02 | 14.49 |
| inside | 168 | 69.35 | 156.38 | 165.12 | 165.02 | 87.07 | 88.86 | 104.71 |
| after | 905 | 23.79 | 49.79 | 45.26 | 11.53 | 16.41 | 16.14 | 15.64 |
| other | 7663 | 18.21 | 40.71 | 19.04 | 11.70 | 12.21 | 12.13 | 12.38 |

### higher_noise: all hours (orders/h)

| Metric | Reference (B4) | Attention model | Uniform control | Difference (model − reference) |
|---|---|---|---|---|
| MAE | 24.98 | 25.61 ± 0.25 (25.31, 25.60, 25.92) | 46.18 ± 0.02 | +0.63 (+2.5%) |
| RMSE | 43.09 | 40.43 ± 0.54 (39.94, 40.17, 41.18) | 62.13 ± 0.08 | -2.67 (-6.2%) |

#### higher_noise: MAE by A-11 group

| Group | Count | B1 | B2 | B3 | B4 | pred_s0 | pred_s1 | pred_s2 |
|---|---|---|---|---|---|---|---|---|
| all | 8736 | 33.94 | 47.02 | 37.88 | 24.98 | 25.31 | 25.60 | 25.92 |
| inside | 168 | 107.36 | 160.42 | 166.97 | 168.02 | 110.26 | 104.83 | 115.57 |
| after | 905 | 37.25 | 54.70 | 58.10 | 21.35 | 26.88 | 26.27 | 25.88 |
| other | 7663 | 31.94 | 43.63 | 32.66 | 22.27 | 23.26 | 23.78 | 23.96 |

### larger_spikes: all hours (orders/h)

| Metric | Reference (B4) | Attention model | Uniform control | Difference (model − reference) |
|---|---|---|---|---|
| MAE | 19.00 | 17.90 ± 0.24 (17.69, 17.77, 18.24) | 48.07 ± 0.16 | -1.10 (-5.8%) |
| RMSE | 68.28 | 51.35 ± 0.18 (51.60, 51.16, 51.29) | 81.82 ± 0.35 | -16.93 (-24.8%) |

#### larger_spikes: MAE by A-11 group

| Group | Count | B1 | B2 | B3 | B4 | pred_s0 | pred_s1 | pred_s2 |
|---|---|---|---|---|---|---|---|---|
| all | 8736 | 22.10 | 50.83 | 33.14 | 19.00 | 17.69 | 17.77 | 18.24 |
| inside | 168 | 138.79 | 361.74 | 388.65 | 392.18 | 239.01 | 248.03 | 260.28 |
| after | 905 | 31.42 | 79.01 | 87.03 | 11.17 | 21.58 | 20.87 | 21.26 |
| other | 7663 | 18.44 | 40.69 | 18.99 | 11.74 | 12.38 | 12.35 | 12.58 |

### both: all hours (orders/h)

| Metric | Reference (B4) | Attention model | Uniform control | Difference (model − reference) |
|---|---|---|---|---|
| MAE | 29.50 | 29.19 ± 0.30 (28.85, 29.12, 29.58) | 51.05 ± 0.24 | -0.31 (-1.1%) |
| RMSE | 79.81 | 66.67 ± 0.30 (66.56, 66.38, 67.08) | 91.64 ± 0.31 | -13.14 (-16.5%) |

#### both: MAE by A-11 group

| Group | Count | B1 | B2 | B3 | B4 | pred_s0 | pred_s1 | pred_s2 |
|---|---|---|---|---|---|---|---|---|
| all | 8736 | 37.21 | 54.18 | 46.88 | 29.50 | 28.85 | 29.12 | 29.58 |
| inside | 168 | 227.14 | 373.96 | 400.88 | 404.04 | 263.47 | 266.30 | 277.68 |
| after | 905 | 46.66 | 84.39 | 101.77 | 21.29 | 32.74 | 30.44 | 31.25 |
| other | 7663 | 31.92 | 43.60 | 32.64 | 22.26 | 23.24 | 23.77 | 23.95 |

## Change from the control series (MAE, orders/h)

| Forecaster | higher_noise | larger_spikes | both |
|---|---|---|---|
| B1 | +14.17 (+72%) | +2.33 (+12%) | +17.43 (+88%) |
| B2 | +3.14 (+7%) | +6.95 (+16%) | +10.30 (+23%) |
| B3 | +13.31 (+54%) | +8.58 (+35%) | +22.31 (+91%) |
| B4 | +10.35 (+71%) | +4.37 (+30%) | +14.87 (+102%) |
| pred_s0 | +11.22 (+80%) | +3.61 (+26%) | +14.76 (+105%) |
| pred_s1 | +11.58 (+83%) | +3.75 (+27%) | +15.10 (+108%) |
| pred_s2 | +11.43 (+79%) | +3.75 (+26%) | +15.10 (+104%) |

## §19 verdict (PHASE0 Amendment 1)

Per seed: beats B4 on the control series; absolute MAE increase control → both vs B4's +14.87.

- pred_s0: control MAE 14.08 (beats reference: True), increase +14.76 → learned useful structure
- pred_s1: control MAE 14.02 (beats reference: True), increase +15.10 → simply adapted
- pred_s2: control MAE 14.49 (beats reference: True), increase +15.10 → simply adapted

**Verdict: inconclusive.** Original relative rule (reported for transparency): simply adapted to the training distribution.
