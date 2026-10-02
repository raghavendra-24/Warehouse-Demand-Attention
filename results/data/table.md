# Warehouse series (seed 101, 8,736 hours)

| Statistic | Value | PHASE0 expectation |
|---|---|---|
| Mean demand | 100.4 orders/h | 100.2 |
| Autocorrelation, lags 1 / 24 / 168 | 0.84 / 0.66 / 0.68 | 0.84 / 0.63 / 0.66 |
| Noise SD at the mean level (formula / measured) | 13.8 / 13.0 orders/h | 13.8 |
| Daily amplitude ÷ noise SD | 5.42 | ≥ 5 (5.42) |
| Spike / drop onsets | 43 / 13 | ≈ 34 / 17 |
| Onset rate per event-free hour, spike / drop | 0.0051 / 0.0015 | 0.004 / 0.002 |
| Share of hours inside an event | 2.5% | 2.3% (≤ 5%) |
| Spike multiplier min / median / max | 2.01 / 2.75 / 3.87 | within 2–4 |
| Drop multiplier min / median / max | 0.30 / 0.41 / 0.60 | within 0.3–0.6 |

Design targets: daily amplitude ≥ 5 × noise SD at the mean level: met; events affect ≤ ~5% of hours: met; spikes 2–4 × the expected level: met
