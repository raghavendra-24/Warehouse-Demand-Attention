# Ablation: scaled vs unscaled attention (toy task, 3 paired seeds)

## At initialisation (step 0, before any update)

| d_k | Arm | Row entropy ÷ ln n (H2) | Logit SD | Readout max weight |
|---|---|---|---|---|
| 4 | scaled | 0.861, 0.821, 0.888 | 0.975, 1.103, 0.808 | 0.276, 0.341, 0.248 |
| 4 | unscaled | 0.657, 0.592, 0.704 | 1.949, 2.206, 1.615 | 0.437, 0.539, 0.408 |
| 64 | scaled | 0.841, 0.827, 0.832 | 0.967, 1.030, 1.030 | 0.309, 0.356, 0.327 |
| 64 | unscaled | 0.151, 0.145, 0.155 | 7.735, 8.241, 8.237 | 0.868, 0.869, 0.856 |

## Readout-row query gradient at step 0 (H3), per seed

| d_k | Unscaled rows < 1% of scaled median | Scaled rows < 1% | p90/p10 scaled | p90/p10 unscaled | Median unscaled ÷ scaled |
|---|---|---|---|---|---|
| 4 | 0.4%, 0.0%, 0.0% | 0.0%, 0.0%, 0.0% | 4.1, 4.4, 3.8 | 6.25, 8.41, 4.93 | 1.86, 1.72, 1.89 |
| 64 | 12.1%, 14.8%, 11.3% | 0.0%, 0.0%, 0.0% | 3.0, 3.7, 2.9 | 1.27e+03, 4.25e+03, 3.29e+03 | 1.13, 0.74, 1.58 |

## Training (H4)

| d_k | Steps to 95%: scaled | Steps to 95%: unscaled | Median ratio unscaled ÷ scaled | Unscaled seeds not reaching 95% | Held-out: scaled | Held-out: unscaled |
|---|---|---|---|---|---|---|
| 4 | 900, 1050, 950 | 850, 1100, 900 | 0.95 | 0 | 1.000, 1.000, 0.999 | 0.999, 1.000, 0.998 |
| 64 | 350, 350, 400 | 750, 700, — | 2.07 | 1 | 1.000, 1.000, 0.999 | 1.000, 1.000, 0.102 |

Diverged runs: 0
