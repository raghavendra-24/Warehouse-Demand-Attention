# Debugging journal

Written as the work happened (PRD §24). Each episode follows the same steps: initial implementation, unexpected result, hypothesis, diagnostic, correction, verification. The commits referenced are in the Git history.

## Episode 1 (2026-10-02, T-202): the hand-computed test could not see a transposed score matrix
1. Initial implementation: attention core + tests; all 18 tests passed.
2. Unexpected result: a deliberate mutation check (S = K Qᵀ instead of Q Kᵀ) still passed all 11 attention tests.
3. Hypothesis: the fixed tiny example gives a symmetric S, so Sᵀ = S hides the bug; random-input tests only check invariants (shapes, row sums, equivariance) that KQᵀ also satisfies.
4. Diagnostic: printed S for the tiny example: [[2, 7], [7, 4]], symmetric.
5. Correction: changed W_K in config.TINY_EXAMPLE so S = [[4, 1], [4, 7]] (asymmetric); added an element-by-element reference test on random inputs (s_ij = Σ_d q_i[d] k_j[d]).
6. Verification: five planted bugs (KQᵀ, no scale, wrong softmax axis, /d_k, AᵀV) each now fail at least one test; 19 tests pass on the correct code.
7. Follow-up (commit `3af601f`, from an independent code review): the new example still had a symmetric **A** ([[a, 1−a], [1−a, a]]), so the hand test alone could not tell AV from AᵀV (the element-by-element test could). W_K was changed once more, giving S = [[4, 1], [4, 5]] and an asymmetric A; this is the example in `wda/config.py`, `results/trace/table.md` and DERIVATION.md. The hand test now catches both KQᵀ and AᵀV on its own.

## Episode 2 (T-304): a NumPy boolean broke metrics.json
- Unexpected: generate_data crashed at finish(): "cannot write bool to JSON"; figures existed, metrics did not.
- Hypothesis: the design-target checks return numpy.bool_, which the JSON converter did not handle.
- Correction: convert numpy.bool_ like other NumPy scalars; test added (NumPy scalars, booleans, NaN → null).

## Episode 3 (T-505): results changed after a refactor that changed nothing
- Unexpected: after moving the model loader, shift's metrics.json differed from the committed version.
- Diagnostic: the differences were ~1e-9 relative (a prediction 57.215 → 57.216); two consecutive reruns of the new code were bit-identical.
- Explanation: the committed run happened while other heavy jobs ran; MKL can then split float32 work differently, changing summation order.
- Consequence: reproducibility statement = generated data exact; model outputs within ~1e-8 relative (bit-exact on an idle machine).

## Episode 4 (T-506): my first explanation of the spike failure was wrong
- Observation: under doubled spikes the model predicts ~350 when demand is ~590 (slope of forecast on y(t) 0.21).
- Hypothesis 1 (from F4): bilinear scores push attention AWAY from huge spike tokens.
- Diagnostic: attention mass on spike hours ROSE (0.171 vs 0.070); swapping attention weights between spike sizes showed the forecast follows the weights, not the values. Hypothesis 1 refuted.
- Hypothesis 2: the value path barely encodes demand (dg/dz = 0.058 in seed 0), so the forecast is a convex combination of nearly calendar-only values, capped at max_j g_j.
- Hypothesis 2 refined after the dose-response experiment: the all-token ceiling does not bind (forecasts stay far below it). As spikes grow, seed 0 moves attention onto the spike tokens (mass to 0.999 at k = 16) and its forecast converges to their value level; seed 1 moves attention elsewhere. The common cause is the tiny value gain on demand (0.02-0.06 per standardised unit).
- Verification: results/failure/table.md (dose-response table, value gains, refuted explanations).

