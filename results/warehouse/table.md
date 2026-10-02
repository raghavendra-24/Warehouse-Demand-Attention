# Warehouse model vs baselines (FINAL_TEST = False)

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
