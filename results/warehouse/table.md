# Warehouse model vs baselines (FINAL_TEST = True)

Reference baseline, chosen on validation MAE among B1–B4: **B4** (validation MAE: B1 20.74, B2 44.55, B3 26.74, B4 15.57).
Standardisation from training targets: μ = 99.911, s = 54.124.

### val: all hours (orders/h)

| Metric | Reference (B4) | Attention model | Uniform control | Difference (model − reference) |
|---|---|---|---|---|
| MAE | 15.57 | 15.37 ± 0.18 (15.18, 15.33, 15.61) | 43.97 ± 0.01 | -0.20 (-1.3%) |
| RMSE | 30.83 | 25.31 ± 0.70 (24.60, 25.07, 26.25) | 53.61 ± 0.12 | -5.52 (-17.9%) |

#### val: MAE by A-11 group

| Group | Count | B1 | B2 | B3 | B4 | pred_s0 | pred_s1 | pred_s2 | ctrl_s0 | ctrl_s1 | ctrl_s2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 1344 | 20.74 | 44.55 | 26.74 | 15.57 | 15.18 | 15.33 | 15.61 | 43.97 | 43.99 | 43.96 |
| inside | 34 | 61.94 | 125.67 | 156.32 | 150.33 | 81.83 | 90.75 | 107.21 | 131.07 | 130.78 | 133.45 |
| after | 223 | 24.94 | 48.09 | 41.18 | 12.41 | 18.28 | 17.00 | 16.20 | 43.92 | 43.97 | 42.61 |
| other | 1087 | 18.60 | 41.29 | 19.73 | 12.01 | 12.46 | 12.63 | 12.63 | 41.26 | 41.27 | 41.44 |

### test: all hours (orders/h)

| Metric | Reference (B4) | Attention model | Uniform control | Difference (model − reference) |
|---|---|---|---|---|
| MAE | 15.42 | 14.97 ± 0.39 (14.55, 14.88, 15.49) | 43.21 ± 0.08 | -0.44 (-2.9%) |
| RMSE | 30.65 | 25.23 ± 0.75 (24.42, 25.03, 26.22) | 52.82 ± 0.14 | -5.43 (-17.7%) |

#### test: MAE by A-11 group

| Group | Count | B1 | B2 | B3 | B4 | pred_s0 | pred_s1 | pred_s2 | ctrl_s0 | ctrl_s1 | ctrl_s2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 1344 | 20.59 | 43.41 | 24.91 | 15.42 | 14.55 | 14.88 | 15.49 | 43.14 | 43.16 | 43.32 |
| inside | 33 | 67.82 | 134.47 | 142.48 | 151.99 | 77.95 | 91.66 | 108.86 | 139.17 | 139.40 | 143.22 |
| after | 189 | 25.92 | 47.16 | 37.61 | 12.28 | 17.10 | 16.63 | 16.30 | 43.25 | 43.36 | 42.27 |
| other | 1122 | 18.30 | 40.10 | 19.32 | 11.93 | 12.26 | 12.33 | 12.61 | 40.29 | 40.30 | 40.56 |
