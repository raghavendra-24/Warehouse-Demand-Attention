# Toy task: associative recall (8 pairs, 16 keys, 16 values)

Chance = 1/16 = 0.0625; guessing among the 8 values present = 0.1250. Held-out = 5,000 unseen sequences.

| Arm | Held-out accuracy, mean ± SD | Per seed | Steps to 95% validation accuracy | Best step |
|---|---|---|---|---|
| attention | 0.9999 ± 0.0001 | 0.9998, 1.0000, 1.0000 | 600, 550, 500 | 700, 650, 600 |
| uniform_control | 0.1249 ± 0.0043 | 0.1196, 0.1250, 0.1302 | not reached, not reached, not reached | 2100, 2100, 2750 |
