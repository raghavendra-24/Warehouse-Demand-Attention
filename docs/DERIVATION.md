# Derivation

This document derives each step of `wda/attention.py`, the project's only attention implementation, which both the toy and the warehouse model call (A-26). Sections 1–6 cover the six §8 topics, section 7 works the tiny example by hand, section 8 covers gradient verification (§9) and section 9 numerical stability (§11).

**Conventions**
- Tokens are rows. x_i is row i of X, a row vector of length d_model. q_i, k_i, v_i and y_i are row i of Q, K, V and Y.
- s_ij is the score of query i against key j. Row i of S, S′ and A belongs to query i.
- Every number is copied from a committed file, cited in brackets. Arithmetic done by hand in this document is marked *hand*.

## The computation at a glance

| # | Operation | Formula | Source in `wda/attention.py` |
|---|---|---|---|
| 1 | Query projection | Q = X W_Q | line 60 |
| 2 | Key projection | K = X W_K | line 61 |
| 3 | Value projection | V = X W_V | line 62 |
| 4 | Similarity scores | S = Q Kᵀ | line 63 |
| 5 | Scaling | S′ = S / √d_k, with d_k read from the last dimension of Q (line 64) | line 65 |
| 6 | Attention weights | A = softmax(S′), row by row | line 69, which calls `stable_softmax` (lines 33–43: row maximum subtracted at line 41, exponential at line 42, row normalisation at line 43) |
| 7 | Output | Y = A V | line 70 |

The function returns every intermediate (Y, A, Q, K, V, S, S′) for inspection, as a named tuple (lines 21–30, returned at line 71). One switch, `scaled` (line 65), changes S′ = S/√d_k to S′ = S and changes nothing else: the §14 ablation (A-19). A second switch, `uniform` (lines 66–67), sets A to 1/n for the controls (A-18, A-27). The softmax is hand-written and no library attention is used (A-02).

## 1. Query, key and value

Each token x_i ∈ ℝ^d_model is projected three times, by three learned matrices:

  q_i = x_i W_Q ∈ ℝ^d_k,  k_i = x_i W_K ∈ ℝ^d_k,  v_i = x_i W_V ∈ ℝ^d_v.

**What each one represents**
- The **query** q_i describes what position i is looking for. It is used only on the row side of S, where it scores position i against every position.
- The **key** k_j describes what position j offers to be matched on. It is used only on the column side of S.
- The **value** v_j is what position j contributes to the output if it is attended to. It never enters the scores. It enters only the weighted sum Y = AV.

Combined, the first two make the score a bilinear form in the raw tokens:

  s_ij = q_i · k_j = x_i W_Q W_Kᵀ x_jᵀ = x_i M x_jᵀ,  where M = W_Q W_Kᵀ is d_model × d_model with rank at most d_k.

So the pair (W_Q, W_K) learns a similarity measure on token space, and W_V separately learns which part of each token's content to pass on.

**Why three projections rather than one shared one**
1. *A shared query and key projection* (W_Q = W_K = W) forces M = W Wᵀ, which is symmetric and positive semidefinite. Then s_ij = s_ji: position i matches j exactly as well as j matches i. And by Cauchy–Schwarz, s_ij ≤ √(s_ii s_jj) ≤ max(s_ii, s_jj), so a token can score another token above itself only when that token has a larger projected norm, not because it is a better match. A forecast often needs one-directional relations: hour t may need hour t−23, the same hour yesterday, without hour t−23 needing hour t. A general M = W_Q W_Kᵀ can be asymmetric and indefinite.
2. *A separate value projection* decouples what is matched from what is carried. In the toy task the match is on the key bits, but the content to return is the value bits (A-18). Tying V to K would force the model to carry the features it matches on. Section 5 shows why this split also matters for the warehouse failure case.

The tiny example (section 7) has W_Q ≠ W_K and gives S = [[4, 1], [4, 5]], which is not symmetric: s_12 = 1 but s_21 = 4 [results/trace/table.md].

## 2. Dot-product similarity

  s_ij = q_i · k_j = Σ_{m=1..d_k} q_im k_jm = ‖q_i‖ ‖k_j‖ cos θ_ij.

**Why q_i · k_j measures how well position j matches what position i is looking for.** Read each of the d_k coordinates as a learned feature. q_im says how much query i wants feature m (negative: it wants its absence; zero: it does not care), and k_jm says how much key j has it. Each product q_im k_jm is positive when the two agree in sign, negative when they disagree, and large when both are strong. The sum adds up the agreement over all features. For fixed norms, the score is largest for a key pointing the same way as the query (cos θ = 1), zero for an orthogonal key and most negative for an opposite key. Softmax compares scores only within row i (section 4), and every score in that row carries the same factor ‖q_i‖. So ‖q_i‖ sets how sharp row i is, while each key's component along the query's direction, ‖k_j‖ cos θ_ij, decides which positions win.

In the toy task the keys are one-hot (A-18). If W_Q and W_K map the same key to aligned vectors and different keys to nearly orthogonal ones, the matching pair token gets the largest score in the readout row. That is what the toy model has to learn.

**Why the matrix QKᵀ.** One matrix product computes all n² dot products: entry (i, j) is row i of Q times column j of Kᵀ, which is row j of K. Rows are queries and columns are keys, so softmax runs along each row. The transpose KQᵀ = Sᵀ swaps the roles with the same n × n shape, so only a value check detects it; hence the tiny example's asymmetric S (section 7).

The dot product has no parameters of its own; its size grows with d_k, which is the subject of the next section.

## 3. Scaling by √d_k

### 3.1 The variance chain

Assume the d_k components of q and of k are independent, with mean zero and variance σ². The initialisation makes σ² ≈ 1 in both models (A-19): toy tokens have ‖x‖² = 2, so W_Q and W_K entries have variance 1/2; warehouse features have E‖f‖² = 3 and the input projection has entries of variance 1/3, which gives unit-variance components of X, and then W_Q and W_K entries have variance 1/16 for d_model = 16 [wda/config.py]. Two tests confirm unit-variance queries (tests/test_attention.py).

1. **With independent zero-mean components of variance σ², Var(q·k) = d_k·σ⁴.** Each term q_m k_m has mean E[q_m]·E[k_m] = 0, and, by independence, variance E[q_m² k_m²] = E[q_m²]·E[k_m²] = σ⁴. The d_k terms are independent, so their variances add: Var(q·k) = d_k σ⁴.
2. **So the logit spread grows like √d_k.** The standard deviation of a score is √d_k·σ². With σ = 1, the scores in a row spread by about 2 at d_k = 4 and by about 8 at d_k = 64.
3. **So the softmax approaches one-hot.** The weight on a row's largest logit is a_max = 1 / (1 + Σ_{j ≠ max} exp(−(s_max − s_j))). The gaps s_max − s_j grow in proportion to the spread. As d_k grows with n fixed, every term in the sum goes to 0 and a_max goes to 1. The row becomes one-hot, attending to a single position that, at initialisation, is chosen by noise.
4. **Dividing by √d_k makes the variance independent of d_k.** Var(q·k/√d_k) = d_k σ⁴ / d_k = σ⁴, which is 1 for σ = 1, whatever d_k is.

### 3.2 What saturation does to gradients

The softmax Jacobian is J = diag(a) − aaᵀ (section 4). For a one-hot row a = e_m, J = diag(e_m) − e_m e_mᵀ = 0. For a nearly one-hot row every entry a_i(δ_ij − a_j) is small, because each a_i is close to 0 or 1. The gradient reaching the scores is J times the gradient reaching the weights, so a saturated row passes almost no gradient back to Q and K: moving its logits barely changes its weights, so the query cannot learn to look elsewhere. Rows saturate to different degrees, so gradients also become uneven.

### 3.3 Scaling changes training, not capacity

Unscaled attention is scaled attention with a larger query matrix:

  softmax(QKᵀ) = softmax((√d_k·Q) Kᵀ / √d_k),  and  √d_k·Q = X (√d_k·W_Q).

Any function the unscaled layer can represent, the scaled layer represents with W_Q multiplied by √d_k, and the reverse holds too. The two arms have the same capacity. They differ in the logit scale at initialisation (section 3.1), and therefore in the gradients the optimiser sees at the start. A test checks this identity numerically (`test_unscaled_equals_scaled_with_rescaled_query_weights` in tests/test_attention.py).

### 3.4 What the ablation measured

On the toy task, at initialisation, three paired seeds per arm. The predicted spread is √d_k·σ² with σ = 1 (*hand*). An entropy fraction of 1 means a uniform row, and 0 means a one-hot row. All four statistics are taken on the fixed diagnostic batch: the logit SD is the standard deviation of all entries of S′, the entropy is averaged over all rows, and the readout max weight is the readout row's largest weight averaged over the batch (`diagnostics` in wda/train.py).

| d_k | Arm | Predicted logit SD | Measured logit SD | Row entropy ÷ ln n | Readout max weight |
|---|---|---|---|---|---|
| 4 | scaled | 1 | 0.975, 1.103, 0.808 | 0.861, 0.821, 0.888 | 0.276, 0.341, 0.248 |
| 4 | unscaled | 2 | 1.949, 2.206, 1.615 | 0.657, 0.592, 0.704 | 0.437, 0.539, 0.408 |
| 64 | scaled | 1 | 0.967, 1.030, 1.030 | 0.841, 0.827, 0.832 | 0.309, 0.356, 0.327 |
| 64 | unscaled | 8 | 7.735, 8.241, 8.237 | 0.151, 0.145, 0.155 | 0.868, 0.869, 0.856 |

[results/ablation/table.md]

Each link of the chain is visible: the unscaled spread tracks √d_k, the unscaled d_k = 64 rows are close to one-hot, and dividing by √d_k brings both back to a spread near 1.

**Gradients (step 0).** The statistic is the norm of the query gradient for the readout row, per sequence of the fixed diagnostic batch of 256 [wda/config.py]. At d_k = 64, 12.1%, 14.8% and 11.3% of the unscaled readout rows have a gradient below 1% of the scaled arm's median, against 0.0% in the scaled arm. The ratio of the 90th to the 10th percentile is 1.27e+03, 4.25e+03 and 3.29e+03 unscaled, against 3.0, 3.7 and 2.9 scaled. The median ratio, unscaled ÷ scaled, is 1.13, 0.74 and 1.58, so saturation does not shrink the typical gradient: it makes gradients very uneven. At d_k = 4 the effect is small: 0.4%, 0.0% and 0.0% of rows, with p90/p10 of 6.25, 8.41 and 4.93 against 4.1, 4.4 and 3.8 [results/ablation/table.md].

**Training.** At d_k = 64, the scaled arm reaches 95% validation accuracy in 350, 350 and 400 steps. The unscaled arm needs 750 and 700 steps, and its third seed never gets there, ending at a held-out accuracy of 0.102. The median ratio of steps, unscaled ÷ scaled, taken over the two seeds where both arms reach 95%, is 2.07. At d_k = 4 the arms are indistinguishable: 900, 1050 and 950 steps against 850, 1100 and 900, a ratio of 0.95 [results/ablation/table.md]. As section 3.3 predicts, wherever both arms train they reach held-out accuracies of 0.998–1.000, so the difference lies in training, not in capacity [results/ablation/table.md]. Adam partly offsets small gradients, so the comparison is judged first on the initial statistics (A-19).

## 4. Softmax

For row i of S′:

  a_ij = exp(s′_ij) / Σ_{l=1..n} exp(s′_il).

**How logits become normalised weights**
- **Exponentiation makes them positive.** exp(z) > 0 for every real z, so every weight is positive, whatever the sign of its logit.
- **Normalising each row makes it sum to 1.** Dividing by the row's sum gives Σ_j a_ij = 1. Row i is a probability distribution over the n positions: it says how query i divides its attention.
- **The order is preserved.** exp is strictly increasing and a row shares one denominator, so s′_ij > s′_il exactly when a_ij > a_il. Softmax is a smooth version of argmax.
- **The scale acts as a temperature.** For softmax(s/τ), the row becomes one-hot on its largest logit as τ → 0 and becomes uniform (1/n) as τ → ∞. Dividing by √d_k applies a temperature τ = √d_k to S. Section 3 chose that value to cancel the growth of the logit spread with d_k.

**Shift invariance.** For any constant c, softmax(s − c·1) = softmax(s), because

  exp(s_j − c) / Σ_l exp(s_l − c) = e^(−c) exp(s_j) / (e^(−c) Σ_l exp(s_l)) = exp(s_j) / Σ_l exp(s_l).

Only the differences between logits matter. With two logits this gives softmax(z_1, z_2) = (σ(z_1 − z_2), σ(z_2 − z_1)), where σ(u) = 1/(1 + e^(−u)) is the logistic sigmoid. Section 7 uses this form.

**Stable form.** Taking c = max_j s_j gives softmax(s) = exp(s − max s) / Σ exp(s − max s). The proof is shift invariance with that choice of c. Section 9 explains why it prevents overflow, and lines 41–43 implement it. The maximum is treated as a constant (detached at line 41). That is exact, not an approximation: the output does not depend on c, so its derivative with respect to c is zero (J·1 = 0 below).

**Jacobian.** Write Z = Σ_l e^(s_l), so that a_i = e^(s_i)/Z.
- For i = j: ∂a_i/∂s_i = (e^(s_i) Z − e^(s_i) e^(s_i)) / Z² = a_i − a_i².
- For i ≠ j: ∂a_i/∂s_j = −e^(s_i) e^(s_j) / Z² = −a_i a_j.

So ∂a_i/∂s_j = a_i(δ_ij − a_j); in matrix form, J = diag(a) − aaᵀ. Its properties are used below:
- J·1 = a − a(aᵀ1) = 0: shifting all logits changes nothing.
- 1ᵀJ = 0: a row's sum cannot change, so sum(A) has zero gradient (section 8).
- J = 0 when a is one-hot: saturation (section 3.2).
- J is largest when a is spread out: its trace, Σ_i a_i(1 − a_i) = 1 − Σ_i a_i², is greatest for a uniform row, where each diagonal entry is (1/n)(1 − 1/n).

## 5. Weighted aggregation

  y_i = Σ_j a_ij v_j,  that is,  Y = A V.

**Each row of AV is a weighted average of the value rows,** with the weights taken from row i of A. Every a_ij is positive and each row of A sums to 1, so y_i is a convex combination of v_1, …, v_n. Three consequences follow.
- **Outputs stay within the range of the values.** For every output component c, min_j v_jc ≤ y_ic ≤ max_j v_jc. Attention can select and blend what the values contain, but it cannot go beyond them.
- **The two extremes are copying and averaging.** A one-hot row copies one value (y_i = v_m). A uniform row gives the mean of the values, which is the mean-pooling control behind the `uniform` switch.
- **The orientation matters.** AᵀV would weight by the columns of A, which are not distributions, so its rows are not averages. The tiny example's asymmetric A catches this bug.

In the tiny example the values are 1 and 4, and the outputs, 1.547277 and 2.867378, both lie inside [1, 4] [results/trace/table.md].

**Why this matters for the failure case.** Both models read their prediction from one row of Y, the last token's, through a linear head (A-06). There is no residual path. With head weights w and bias b, the warehouse forecast is

  ŷ = w·y_t + b = Σ_j a_tj (w·v_j) + b,

so it is a weighted average of per-hour value levels g_j = w·v_j, plus a constant. It can follow a demand spike in only two ways: the values can carry the spike's size, or attention can move onto the tokens with the larger values. The second route is bounded by the largest value level in the window. The failure investigation found that the learned values carry almost no demand magnitude, so the forecast can rise only by moving attention, and it saturates during large spikes. The forecasts stay well below the all-token ceiling, so what limits them is how little the values change with demand, not the ceiling itself [results/failure/table.md].

## 6. Tensor shapes

### 6.1 Symbolic

| Tensor | Formula | One sequence | Batch of B |
|---|---|---|---|
| X | input tokens | n × d_model | B × n × d_model |
| W_Q, W_K | parameters | d_model × d_k | shared across the batch |
| W_V | parameter | d_model × d_v | shared across the batch |
| Q | X W_Q | n × d_k | B × n × d_k |
| K | X W_K | n × d_k | B × n × d_k |
| V | X W_V | n × d_v | B × n × d_v |
| S = QKᵀ | Q Kᵀ | n × n | B × n × n |
| S′ | S / √d_k | n × n | B × n × n |
| A | softmax(S′) by rows | n × n | B × n × n |
| Y | A V | n × d_v | B × n × d_v |

The products conform as (n × d_model)(d_model × d_k) = n × d_k, then (n × d_k)(d_k × n) = n × n, then (n × n)(n × d_v) = n × d_v. With a batch, the transpose acts on the last two axes, and every product is applied to each sequence with the same weights.

### 6.2 At the actual sizes

| Tensor | Tiny example | Gradient check and shape test | Toy model | Warehouse model |
|---|---|---|---|---|
| Raw input per sequence | — | — | 9 tokens × 33 features | 24 tokens × 5 features |
| X | 2 × 3 | 5 × 3 | 9 × 33 | 24 × 16, after the 5 → 16 input projection |
| W_Q, W_K | 3 × 4 | 3 × 4 | 33 × 16 (33 × 4 or 33 × 64 in the ablation) | 16 × 16 |
| W_V | 3 × 1 | 3 × 2 | 33 × 16 | 16 × 16 |
| Q, K | 2 × 4 | 5 × 4 | 9 × 16 (9 × 4 or 9 × 64) | 24 × 16 |
| V | 2 × 1 | 5 × 2 | 9 × 16 | 24 × 16 |
| S = QKᵀ, S′, A | 2 × 2 | 5 × 5 | 9 × 9 | 24 × 24 |
| Y | 2 × 1 | 5 × 2 | 9 × 16 | 24 × 16 |
| Readout | — | — | last row of Y (1 × 16) → linear head → 16 logits, one per value | last row of Y, hour t (1 × 16) → linear head → 1 forecast ẑ(t+1) |
| Training batch B | — | — (shape test also uses B = 3) | 256 | 256 |

[wda/config.py]

- **Toy:** 8 key–value pair tokens plus 1 query token; 33 features = a 16-way one-hot key, a 16-way one-hot value and a query flag [wda/config.py].
- **Warehouse:** the hours y(t−23) … y(t); 5 features = standardised demand plus sine and cosine of hour of day and of day of week (A-05), projected to d_model = 16 [wda/config.py].
- In both models only the readout row of A reaches the loss, which is why section 3.4 measures gradients for that row.

This agrees with the shape tests (FAC-54). tests/test_attention.py, with n = 5, d_model = 3, d_k = 4, d_v = 2, unbatched and with B = 3, asserts Q, K n × d_k; V n × d_v; S, S′, A n × n; Y n × d_v. tests/test_models.py asserts that the toy model gives one logit per value and a 9 × 9 A, and the warehouse model one prediction per window and a 24 × 24 A.

## 7. Worked example

The fixed tiny example, `TINY_EXAMPLE` [wda/config.py], has n = 2, d_model = 3, d_k = 4 and d_v = 1. Because every size differs, a misplaced transpose in a projection gives a shape error. KQᵀ has the right shape, so the example makes S and A asymmetric to catch it by value.

  X = [[1, 0, 1], [0, 2, 1]]
  W_Q = [[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]]
  W_K = [[1, 0, 1, −1], [0, 0, 0, 0], [0, 1, 0, 1]]
  W_V = [[1], [2], [0]]

**Step 1, Q = X W_Q.** Row 1 of X, (1, 0, 1), adds rows 1 and 3 of W_Q: (1, 0, 1, 0) + (1, 1, 0, 0) = (2, 1, 1, 0). Row 2, (0, 2, 1), adds twice row 2 and row 3: (0, 2, 0, 2) + (1, 1, 0, 0) = (1, 3, 0, 2).

**Step 2, K = X W_K.** Row 1: (1, 0, 1, −1) + (0, 1, 0, 1) = (1, 1, 1, 0). Row 2: 2·(0, 0, 0, 0) + (0, 1, 0, 1) = (0, 1, 0, 1).

**Step 3, V = X W_V.** Row 1: 1·1 + 0·2 + 1·0 = 1. Row 2: 0·1 + 2·2 + 1·0 = 4.

**Step 4, S = QKᵀ.**
- s_11 = (2, 1, 1, 0)·(1, 1, 1, 0) = 2 + 1 + 1 + 0 = 4
- s_12 = (2, 1, 1, 0)·(0, 1, 0, 1) = 0 + 1 + 0 + 0 = 1
- s_21 = (1, 3, 0, 2)·(1, 1, 1, 0) = 1 + 3 + 0 + 0 = 4
- s_22 = (1, 3, 0, 2)·(0, 1, 0, 1) = 0 + 3 + 0 + 2 = 5

So S = [[4, 1], [4, 5]]. The bug KQᵀ would give Sᵀ = [[4, 4], [1, 5]].

**Step 5, S′ = S/√d_k.** √d_k = √4 = 2, so S′ = [[2, 0.5], [2, 2.5]].

**Step 6, A = softmax(S′).** With two logits, each weight is the sigmoid of the logit difference (section 4).
- Row 1: the logits are 2 and 0.5, which differ by 1.5. a_11 = σ(1.5) = 1/(1 + e^(−1.5)) = 1/(1 + 0.223130) = 0.817574, and a_12 = 1 − a_11 = 0.182426 (*hand*).
- Row 2: the logits are 2 and 2.5, so token 2 leads by 0.5. a_22 = σ(0.5) = 1/(1 + e^(−0.5)) = 1/(1 + 0.606531) = 0.622459, and a_21 = 0.377541 (*hand*).

Each row sums to 1. A is asymmetric (a_12 ≠ a_21), and its columns do not sum to 1 (0.817574 + 0.377541 ≠ 1), so softmax over the wrong axis would also be caught.

**Step 7, Y = AV.** With values 1 and 4, y_i = a_i1·1 + a_i2·4 = 1 + 3·a_i2.
- y_1 = 1 + 3 × 0.18242552 = 1.54727656, which rounds to 1.547277
- y_2 = 1 + 3 × 0.62245933 = 2.86737799, which rounds to 2.867378

The weights are carried to eight decimals here and Y is rounded to six only at the end. Rounding the weights to six decimals first would give 1.547278 and 2.867377, one unit off in the last place (*hand*).

**Comparison with the program.**

| Quantity | By hand (above) | `results/trace/table.md` |
|---|---|---|
| Q | [[2, 1, 1, 0], [1, 3, 0, 2]] | same |
| K | [[1, 1, 1, 0], [0, 1, 0, 1]] | same |
| V | [[1], [4]] | same |
| S | [[4, 1], [4, 5]] | same |
| S′ | [[2, 0.5], [2, 2.5]] | same |
| A | [[0.817574, 0.182426], [0.377541, 0.622459]] | same; rows sum to [1.0, 1.0] |
| Y | [[1.547277], [2.867378]] | same |

[results/trace/table.md]

The hand-computed forward test in tests/test_attention.py encodes the same values. It checks Q, K, V, S and S′ exactly, A as σ(1.5) and σ(0.5), and Y as a·1 + (1 − a)·4. It also asserts that the unscaled version does not match, so removing 1/√d_k makes the test fail (FAC-14).

Read as attention: query 1 matches key 1 far better (4 against 1), so y_1 stays near value 1; query 2 slightly prefers key 2 (5 against 4), so y_2 leans towards 4. Without scaling, both logit gaps would double and both rows would be sharper, the temperature effect of section 4.

## 8. Gradient verification

**Method: Option A, autograd, cross-checked against float64 central finite differences.** Training uses autograd: Adam receives the gradient of the training loss (cross-entropy on the toy task, MSE on the warehouse model, A-14) with respect to W_Q, W_K, W_V, the readout head and, in the warehouse model, the input projection. Because training relies on it, every entry of autograd's gradient is compared with a central finite difference computed from forward evaluations alone (`wda/gradcheck.py`, A-17).

**The loss that is differentiated.**

  L(X, W_Q, W_K, W_V) = Σ_{i,c} R_ic · Y_ic = Σ R ⊙ Y,

where Y is the scaled attention output and R is a fixed random n × d_v matrix, drawn once from the same generator as X and the three weight matrices, seeded with 7 [wda/config.py]. Then ∂L/∂Y = R, so the check pushes a generic random direction back through every operation.

**The tensors differentiated: X, W_Q, W_K and W_V,** which are the four inputs of the core. X is included because, in the warehouse model, the gradient must also flow through attention into the input projection that produces X.

**Why the loss is non-degenerate.**
- *sum(A) is not used, because each row of A sums to 1, so its gradient is zero.* sum(A) = n for every input, so its gradient is exactly zero (1ᵀJ = 0, section 4), and W_V does not even enter A. A check on sum(A) would compare zero with round-off noise and pass for almost any implementation.
- *sum(Y) is too symmetric.* Its upstream gradient is the all-ones matrix. Then ∂L/∂A_ij = Σ_c v_jc is the same for every query row i, and ∂L/∂W_V = Xᵀ Aᵀ 1 has identical columns, so an error that mixed up value columns would go unseen. A random R gives each output entry its own weight, so every entry of the Jacobian of Y enters the check with a different coefficient.

**Sizes.** n = 5, d_model = 3, d_k = 4, d_v = 2 [wda/config.py]: all different, so a transposition shows up as a shape error rather than as silently wrong numbers. That gives 15 + 12 + 12 + 6 = 45 entries [results/gradcheck/table.md].

**Finite differences and their error.** Each entry θ is perturbed in turn, and g_num = (L(θ + h) − L(θ − h)) / 2h, with h = 1e-6, in float64 [wda/config.py]. By Taylor expansion, L(θ ± h) = L ± hL′ + (h²/2)L″ ± (h³/6)L‴ + O(h⁴). Subtracting cancels the even terms, so g_num = L′ + (h²/6)L‴ + O(h⁴), and **truncation error is proportional to h²**. Each evaluation of L also carries rounding error of order ε|L|, with ε ≈ 1.1e-16 in float64 [wda/gradcheck.py]; dividing by 2h gives **round-off error proportional to ε/h**. The total, E(h) ≈ C₁h² + C₂ε/h, is smallest where 2C₁h = C₂ε/h², that is at **h\* ∝ ε^(1/3)**. In float64, ε^(1/3) ≈ 4.8e-6 (*hand*), so h = 1e-6 sits near the optimum. At that h, truncation is about h²/6 ≈ 1.7e-13 times |L‴| (*hand*), which is negligible. Round-off is ε/h ≈ 1.1e-10 (*hand*), times |L| and the rounding accumulated across the forward pass; the observed differences, up to 8.98e-10 [results/gradcheck/table.md], are of that size. A float32 check would fail on round-off alone, which is why the check runs in float64 (A-17).

**The agreement rule.** An entry agrees if

  |g_auto − g_num| ≤ 1e-8 + 1e-6·|g_num|  (atol = 1e-8, rtol = 1e-6) [wda/config.py].

A purely relative test, |g_auto − g_num| / |g_num| ≤ rtol, fails near zero: if a true gradient is zero, g_num is pure round-off noise, the ratio is of order 1 or undefined, and a correct entry is flagged. The absolute floor of 1e-8 sits about ten times above the largest observed round-off and far below any meaningful error; the relative term lets the tolerance grow with the gradient. The data show the effect: the five largest relative differences, from 2.62e-08 down to 4.10e-09, belong to the five smallest gradients, all of magnitude 0.083 or less [results/gradcheck/table.md]. The smallest gradient, W_K (0, 2) at +0.0200799948, has the largest relative difference, 2.62e-08, with an ordinary absolute difference of 5.27e-10. W_Q (1, 2), at +0.0396528129, has 1.47e-08 [results/gradcheck/table.md]. The same round-off divided by a smaller gradient gives a larger relative figure. The table's relative column divides by the larger of |g_auto| and |g_num|.

**Results.**

| Tensor | Entries | Max abs. difference | Max rel. difference | Disagreeing |
|---|---|---|---|---|
| X | 15 | 8.05e-10 | 3.69e-09 | 0 |
| W_Q | 12 | 8.98e-10 | 1.47e-08 | 0 |
| W_K | 12 | 7.57e-10 | 2.62e-08 | 0 |
| W_V | 6 | 3.37e-10 | 4.10e-09 | 0 |

[results/gradcheck/table.md]

A representative entry is X (0, 0): autograd −0.6625417251, numerical −0.6625417258, absolute difference 7.55e-10, relative difference 1.14e-09 [results/gradcheck/table.md]. The saved table lists all 45 entries with autograd, numerical, absolute and relative columns, and ends with "Unexplained entries: 0". The differences are the round-off analysed above.

**The check has power.** In a negative control, W_Q is detached inside the loss: autograd reports zero, the finite differences still see L change, and the check flags W_Q but not W_K (tests/test_gradients.py). The check also passes for both settings of the scaling switch on batched input.

**What a gradient check cannot catch.** Both g_auto and g_num are derivatives of the same forward code. If the forward pass computed KQᵀ instead of QKᵀ, autograd and finite differences would agree on the derivative of the wrong function, and the check would pass. It verifies that the backward pass matches the forward pass, not that the forward pass matches the PRD. So the forward pass has separate checks (A-17): the hand-computed tiny example (section 7), whose asymmetric S and A make KQᵀ and AᵀV change the result; an entry-by-entry reference s_ij = Σ_m q_im k_jm on random inputs; and a permutation-equivariance test (tests/test_attention.py).

## 9. Numerical stability

### 9.1 Where exp overflows

The largest finite float32 number is about 3.4e38, so exp(z) overflows to inf for z above ln(3.4e38) ≈ 88.72 [results/stability/table.md]. In float64 the threshold is about 709.78 (*hand*). A naive softmax, exp(s)/Σ exp(s), has two ways to fail.
- If one exponential overflows, inf/inf gives NaN.
- If every exponential is finite but their sum overflows, each finite numerator divided by inf gives 0. The result is finite, but wrong.

### 9.2 Why subtracting the maximum prevents overflow

softmax(s − c) = softmax(s) for any constant c (section 4). Choose c = max_j s_j. Then:
- every exponent s_j − c is ≤ 0, so every exp(s_j − c) lies in (0, 1], and no term can overflow;
- the largest entry contributes exactly exp(0) = 1, so the denominator lies between 1 and n. It cannot overflow (n is at most 24 in this project [wda/config.py]), and it cannot underflow to zero, so the division is always defined.

Very small terms can still underflow, but their true weights are below what float32 can represent anyway.

### 9.3 Demonstration in float32

| Logits | Naive | Stable | Naive finite | Naive correct (≤ 1e-6) | Stable sums to |
|---|---|---|---|---|---|
| [80, 79, 0] | [0.7311, 0.2689, 1.319e-35] | [0.7311, 0.2689, 1.319e-35] | yes | yes | 1.0000000 |
| [88, 87, 0] | [0.7311, 0.2689, 4.426e-39] | [0.7311, 0.2689, 4.426e-39] | yes | yes | 1.0000000 |
| [88.5, 87.5, 0] | [0, 0, 0] | [0.7311, 0.2689, 2.685e-39] | yes | no | 1.0000000 |
| [89, 88, 0] | [nan, 0, 0] | [0.7311, 0.2689, 1.628e-39] | no | no | 1.0000000 |
| [90, 89, 0] | [nan, nan, 0] | [0.7311, 0.2689, 5.99e-40] | no | no | 1.0000000 |
| [1000, 999, 0] | [nan, nan, 0] | [0.7311, 0.2689, 0] | no | no | 1.0000000 |

[results/stability/table.md; the saved table also has the rows [85, 84, 0] and [100, 99, 0]]

- **Up to [88, 87, 0]** the naive form is finite and agrees with the stable form to within 6.0e-08 [results/stability/table.md].
- **[88.5, 87.5, 0]: finite is not the same as correct.** Each exponent is below 88.72, so each exponential is finite, but their sum exceeds the float32 maximum and becomes inf. Each finite numerator divided by inf gives 0, so the naive form returns [0, 0, 0], with a maximum difference of 7.3e-01 from the right answer [results/stability/table.md]. No NaN, no error: a finiteness check passes, and only a row-sum check (0 instead of 1) or a comparison with the stable form catches it. This row refuted part of the pre-registered expectation that naive and stable agree wherever the naive form is finite (V2 in RESULTS.md).
- **[89, 88, 0]** overflows e^89: inf/inf = NaN, then finite/inf = 0. From [90, 89, 0] to [1000, 999, 0] both large exponentials overflow: [nan, nan, 0] [results/stability/table.md].
- **The stable form** gives finite weights that sum to 1 on every row, including [1000, 999, 0]. The first two weights, 0.7311 and 0.2689, are the same on every row because the gap between the two large logits is always 1: shift invariance made visible (0.7311 is σ(1), *hand*). The third weight becomes exactly 0 at [1000, 999, 0], where exp(−1000) underflows, which is the correct float32 answer [results/stability/table.md].

A unit test runs the [1000, 999, 0] row: the naive exponentials overflow, and the stable output is finite and equals [1, e^(−1), 0] / (1 + e^(−1)) (tests/test_attention.py).

### 9.4 Saturation: the problem max-subtraction does not fix

Max-subtraction removes overflow, a floating-point problem, exactly. It does nothing about saturation, a property of softmax that occurs in exact arithmetic too: with a large logit spread, rows become nearly one-hot and J = diag(a) − aaᵀ approaches 0 (section 3.2). In float32, tiny weights underflow to exactly 0, as in the last row above, and their gradients are then exactly zero. The remedy is to control the logit scale: divide by √d_k and initialise for unit-variance queries and keys (section 3.1).

The ablation (section 3.4) measures saturation directly. With logits stable but unscaled at d_k = 64, the row entropy at initialisation is 0.151, 0.145 and 0.155 of its maximum, against 0.841, 0.827 and 0.832 scaled; the readout row's largest weight is 0.868, 0.869 and 0.856, against 0.309, 0.356 and 0.327; 12.1%, 14.8% and 11.3% of readout-row query gradients fall below 1% of the scaled median; and one of three seeds never reaches 95% validation accuracy [results/ablation/table.md]. Stable arithmetic did not prevent any of this; the √d_k scaling does.
