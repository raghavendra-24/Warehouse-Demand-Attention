"""Pre-implementation calculation of initial attention statistics.

A calculation of initial statistics, not an experiment result. It is quoted in
PHASE0 hypotheses H2 and H3 and in ASSUMPTIONS A-19, and was run before any
project code existed. It draws random unit-variance query and key vectors,
computes one attention row with and without the 1/sqrt(d_k) scale, and reports:
- mean row entropy as a fraction of ln n (H2);
- the per-row gradient norm of L = (a V) . r with respect to the query (H3).

Run: python prereg/init_stats.py > prereg/init_stats.txt
"""
import sys

import numpy as np


def rows(n, dk, scaled, trials, rng):
    entropy, grad = [], []
    for _ in range(trials):
        q = rng.standard_normal(dk)
        K = rng.standard_normal((n, dk))
        V = rng.standard_normal((n, 4))
        r = rng.standard_normal(4)
        c = 1 / np.sqrt(dk) if scaled else 1.0
        s = c * K @ q
        a = np.exp(s - s.max())
        a /= a.sum()
        entropy.append(-(a * np.log(a + 1e-300)).sum())
        J = np.diag(a) - np.outer(a, a)            # softmax Jacobian
        grad.append(np.linalg.norm(c * K.T @ (J @ (V @ r))))   # ||dL/dq||
    return np.array(entropy), np.array(grad)


def main():
    rng = np.random.default_rng(0)
    print(f"numpy {np.__version__}, python {sys.version.split()[0]}, 6000 trials per cell")
    print("n   d_k  entropy/ln n (scaled, unscaled)  median ||dL/dq|| (scaled, unscaled)  "
          "unscaled rows < 1% of scaled median  p90/p10 (scaled, unscaled)")
    for n in (9, 17, 24):
        for dk in (4, 64):
            hs, gs = rows(n, dk, True, 6000, rng)
            hu, gu = rows(n, dk, False, 6000, rng)
            tiny = np.mean(gu < 0.01 * np.median(gs))
            print(f"{n:<3} {dk:<4} {hs.mean() / np.log(n):.2f}, {hu.mean() / np.log(n):.2f}"
                  f"{'':22}{np.median(gs):.3f}, {np.median(gu):.3f}"
                  f"{'':24}{tiny:.1%}"
                  f"{'':33}{np.percentile(gs, 90) / np.percentile(gs, 10):.0f}, "
                  f"{np.percentile(gu, 90) / np.percentile(gu, 10):.0f}")


if __name__ == "__main__":
    main()
