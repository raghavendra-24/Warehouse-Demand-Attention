"""Phase 0 design calculation: expected statistics of the warehouse generator.

A calculation, not an experiment: no random numbers are drawn and no project
code is used. Count noise is approximated as Gaussian with the negative-binomial
variance; events are integrated over onset hour, duration and multiplier and
weighted by their expected rate. Quoted in PHASE0 sections 2, 3, 7 and 8.
"""
import math

B, A1, A2 = 100.0, 60.0, 15.0
WDAY = (5, 5, 5, 5, 0, -15, -25)          # Mon..Sun
P_SPIKE, P_DROP = 0.004, 0.002
SPIKE_L, DROP_L = (3, 4, 5, 6), (2, 3, 4)
SPIKE_M, DROP_M = (2.0, 4.0), (0.3, 0.6)
R_BASE, R_SHIFT = 100.0, 14.0
GRID = 8                                   # midpoint grid for uniform multipliers


def daily(h):
    x = 2 * math.pi * (h - 14) / 24
    return A1 * math.cos(x) + A2 * math.cos(2 * x)


LAM0 = [B + daily(t % 24) + WDAY[(t // 24) % 7] for t in range(168 * 4)]


def nb_var(lam, r):
    return lam + lam * lam / r


def e_abs(mu, s):
    """E|e| for e ~ N(mu, s^2)."""
    phi = 0.5 * (1 + math.erf(-mu / s / math.sqrt(2)))
    return s * math.sqrt(2 / math.pi) * math.exp(-mu * mu / (2 * s * s)) + mu * (1 - 2 * phi)


def baseline_errors(lam, t, r):
    """Expected |error| of B1, B2, B3 for target t+1 given expected levels lam."""
    y = lam[t + 1]
    vy = nb_var(y, r)
    b1 = e_abs(y - lam[t], math.sqrt(vy + nb_var(lam[t], r)))
    win = lam[t - 23:t + 1]
    b2 = e_abs(y - sum(win) / 24, math.sqrt(vy + sum(nb_var(l, r) for l in win) / 576))
    b3 = e_abs(y - lam[t - 23], math.sqrt(vy + nb_var(lam[t - 23], r)))
    return b1, b2, b3


def normal_hours(r):
    tot = [0.0, 0.0, 0.0]
    per_day = [[0.0, 0.0] for _ in range(7)]
    for t in range(168, 336):
        e = baseline_errors(LAM0, t, r)
        d = ((t + 1) // 24) % 7
        per_day[d][0] += e[0] / 24
        per_day[d][1] += e[2] / 24
        for i in range(3):
            tot[i] += e[i] / 168
    return tot, per_day


def event_extra(durations, mrange, r, scale=1.0):
    """Expected extra absolute error per event (summed over affected targets),
    and mean MAE over in-event targets and over the first three in-spike targets."""
    ms = [mrange[0] + (mrange[1] - mrange[0]) * (k + 0.5) / GRID for k in range(GRID)]
    extra, inside, first3, n = [0.0] * 3, [0.0] * 3, [0.0] * 3, 0
    for onset in range(168, 336):
        for L in durations:
            for m in ms:
                lam = list(LAM0)
                for t in range(onset, onset + L):
                    lam[t] = LAM0[t] * m * scale
                for t in range(onset - 1, onset + L + 24):
                    ev, no = baseline_errors(lam, t, r), baseline_errors(LAM0, t, r)
                    for i in range(3):
                        extra[i] += ev[i] - no[i]
                        if onset <= t + 1 < onset + L:
                            inside[i] += ev[i]
                            if t + 1 < onset + 3:
                                first3[i] += ev[i]
                n += 1
    Lm = sum(durations) / len(durations)
    return ([x / n for x in extra], [x / n / Lm for x in inside], [x / n / 3 for x in first3])


def event_rates():
    mean_spike, mean_drop = sum(SPIKE_L) / 4, sum(DROP_L) / 3
    p = P_SPIKE + P_DROP
    cycle = 1 / p + (P_SPIKE * mean_spike + P_DROP * mean_drop) / p
    rs, rd = P_SPIKE / p / cycle, P_DROP / p / cycle
    return rs, rd, rs * mean_spike, rd * mean_drop


def acf(lags, r=R_BASE, events=True):
    rs, rd, fs, fd = event_rates() if events else (0, 0, 0, 0)
    ems = sum(SPIKE_M) / 2
    ems2 = ems ** 2 + (SPIKE_M[1] - SPIKE_M[0]) ** 2 / 12
    emd = sum(DROP_M) / 2
    emd2 = emd ** 2 + (DROP_M[1] - DROP_M[0]) ** 2 / 12
    em = 1 + fs * (ems - 1) + fd * (emd - 1)
    em2 = (1 - fs - fd) + fs * ems2 + fd * emd2

    def emm(k):   # E[M_t M_{t+k}], events as a stationary renewal process
        ps = rs * sum(max(L - k, 0) for L in SPIKE_L) / len(SPIKE_L)
        pd = rd * sum(max(L - k, 0) for L in DROP_L) / len(DROP_L)
        p00 = 1 - (fs + fd) - (fs - ps) - (fd - pd)
        return ps * ems2 + pd * emd2 + 2 * (fs - ps) * ems + 2 * (fd - pd) * emd + p00

    m = sum(LAM0[:168]) / 168
    prod = lambda k: sum(LAM0[t] * LAM0[t + k] for t in range(168)) / 168
    mean = m * em
    var = prod(0) * em2 + (m * em + prod(0) * em2 / r) - mean ** 2
    return {k: (prod(k) * emm(k) - mean ** 2) / var for k in lags}, mean, math.sqrt(var)


def main():
    m0 = sum(LAM0[:168]) / 168
    sd = math.sqrt(nb_var(m0, R_BASE))
    print(f"mean expected level {m0:.2f}; noise SD there {sd:.2f}; amplitude 75 / SD = {75 / sd:.2f}")
    print(f"level range {min(LAM0):.0f}..{max(LAM0):.0f}; D(h) = {[round(daily(h), 1) for h in range(24)]}")
    for lam in (30, m0, 100, 180, 720, 1440):
        print(f"  lambda {lam:7.1f}: SD r=100 {math.sqrt(nb_var(lam, R_BASE)):6.1f}, r=14 {math.sqrt(nb_var(lam, R_SHIFT)):6.1f}")
    rs, rd, fs, fd = event_rates()
    print(f"events/h spikes {rs:.5f} drops {rd:.5f}; event-hour share {fs + fd:.4f}; "
          f"per 52 w {rs * 8736:.1f}/{rd * 8736:.1f}, per 36 w {rs * 6048:.1f}/{rd * 6048:.1f}, per 8 w {rs * 1344:.1f}/{rd * 1344:.1f}")
    a, mu, s = acf((1, 12, 24, 168))
    a0, _, _ = acf((1, 12, 24, 168), events=False)
    print("ACF with events", {k: round(v, 2) for k, v in a.items()}, " without", {k: round(v, 2) for k, v in a0.items()})
    print(f"expected target mean {mu:.1f}, SD {s:.1f}")
    oracle = sum(math.sqrt(2 / math.pi) * math.sqrt(nb_var(l, R_BASE)) for l in LAM0[:168]) / 168
    print(f"oracle normal-hour MAE (predicting lambda) {oracle:.1f}")
    for name, r, sc in (("control", R_BASE, 1), ("higher noise", R_SHIFT, 1), ("larger spikes", R_BASE, 2), ("both", R_SHIFT, 2)):
        nh, per_day = normal_hours(r)
        xs, ins, f3 = event_extra(SPIKE_L, SPIKE_M, r, sc)
        xd, _, _ = event_extra(DROP_L, DROP_M, r)
        overall = [nh[i] + rs * xs[i] + rd * xd[i] for i in range(3)]
        print(f"{name:13s} normal B1/B2/B3 {nh[0]:.1f} {nh[1]:.1f} {nh[2]:.1f} | all hours {overall[0]:.1f} {overall[1]:.1f} {overall[2]:.1f}"
              f" | extra per spike {xs[0]:.0f} {xs[1]:.0f} {xs[2]:.0f} | in-spike MAE {ins[0]:.0f} {ins[1]:.0f} {ins[2]:.0f} | B1 first 3 h {f3[0]:.0f}")
        if name == "control":
            print("   normal-hour MAE by target day Mon..Sun  B1", [round(x[0], 1) for x in per_day], " B3", [round(x[1], 1) for x in per_day])


if __name__ == "__main__":
    main()
