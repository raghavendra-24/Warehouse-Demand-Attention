# Softmax in float32: naive exp(s)/Σexp(s) vs max-subtracted

| Logits | Naive | Stable | Naive finite | Naive correct (≤ 1e-6) | Stable sums to | Max difference |
|---|---|---|---|---|---|---|
| [80, 79, 0] | [0.7311, 0.2689, 1.319e-35] | [0.7311, 0.2689, 1.319e-35] | yes | yes | 1.0000000 | 3.0e-08 |
| [85, 84, 0] | [0.7311, 0.2689, 8.89e-38] | [0.7311, 0.2689, 8.89e-38] | yes | yes | 1.0000000 | 0.0e+00 |
| [88, 87, 0] | [0.7311, 0.2689, 4.426e-39] | [0.7311, 0.2689, 4.426e-39] | yes | yes | 1.0000000 | 6.0e-08 |
| [88.5, 87.5, 0] | [0, 0, 0] | [0.7311, 0.2689, 2.685e-39] | yes | no | 1.0000000 | 7.3e-01 |
| [89, 88, 0] | [nan, 0, 0] | [0.7311, 0.2689, 1.628e-39] | no | no | 1.0000000 | — |
| [90, 89, 0] | [nan, nan, 0] | [0.7311, 0.2689, 5.99e-40] | no | no | 1.0000000 | — |
| [100, 99, 0] | [nan, nan, 0] | [0.7311, 0.2689, 2.803e-44] | no | no | 1.0000000 | — |
| [1000, 999, 0] | [nan, nan, 0] | [0.7311, 0.2689, 0] | no | no | 1.0000000 | — |

float32 exp overflows above ln(3.4e38) ≈ 88.72; the stable form keeps every exponent ≤ 0.
Finite is not the same as correct: when each exp is finite but their sum overflows, naive softmax
returns finite zeros.
