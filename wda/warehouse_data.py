"""Synthetic warehouse demand (PRD §4, §6; PHASE0 section 2).

Expected level (orders per hour), for hour t with hour-of-day h and weekday d:

    λ₀(t) = B + D(h) + W(d)
    λ(t)  = max(λ_min, m(t) · λ₀(t))      m(t) = multiplier of the active event, 1 if none
    y(t)  ~ NegBin(mean λ(t), dispersion r),  Var = λ + λ²/r

Events and count noise come from two child streams of one seed. Every event
quantity is drawn up front, one uniform per hour for onset, duration and
multiplier, so the number of draws never depends on r or on the spike scale.
The four shift series therefore share one event timeline (A-20).
"""

import numpy as np

from wda.config import WarehouseDataConfig


def daily_profile(hour: np.ndarray, cfg: WarehouseDataConfig) -> np.ndarray:
    """D(h) = a1 cos(2π(h − peak)/24) + a2 cos(4π(h − peak)/24); sums to zero over a day."""
    phase = 2 * np.pi * (hour - cfg.daily_peak_hour) / 24
    return cfg.daily_a1 * np.cos(phase) + cfg.daily_a2 * np.cos(2 * phase)


def generate_series(cfg: WarehouseDataConfig) -> dict:
    """The hourly series as arrays of length cfg.n_hours (ARCHITECTURE §4.2)."""
    event_seq, noise_seq = np.random.SeedSequence(cfg.seed).spawn(2)
    event_rng, noise_rng = np.random.default_rng(event_seq), np.random.default_rng(noise_seq)

    t = np.arange(cfg.n_hours)
    hour = t % 24
    day = (cfg.start_day + t // 24) % 7
    daily = daily_profile(hour, cfg)
    weekly = np.asarray(cfg.weekday_offsets)[day]
    base_level = cfg.base + daily + weekly                       # λ₀

    multiplier, label, onset = _events(cfg, event_rng)
    level = np.maximum(cfg.floor, multiplier * base_level)       # λ
    p = cfg.dispersion / (cfg.dispersion + level)                # NumPy: n = r, p = r / (r + λ)
    demand = noise_rng.negative_binomial(cfg.dispersion, p)

    return {
        "t": t,
        "hour_of_day": hour,
        "day_of_week": day,
        "daily": daily,
        "weekly": weekly,
        "event": level - np.maximum(cfg.floor, base_level),       # E(t), the event's contribution
        "level": level,
        "demand": demand.astype(np.int64),
        "event_label": label,                                    # +1 spike, −1 drop, 0 none
        "event_onset": onset,                                    # onset hour of the active event, −1 if none
    }


def _events(cfg: WarehouseDataConfig, rng: np.random.Generator):
    """Non-overlapping spikes and drops; one constant multiplier per event."""
    n = cfg.n_hours
    u_onset, u_duration, u_multiplier = rng.random(n), rng.random(n), rng.random(n)
    multiplier = np.ones(n)
    label = np.zeros(n, dtype=np.int64)
    onset = np.full(n, -1, dtype=np.int64)
    t = 0
    while t < n:
        if u_onset[t] < cfg.spike_onset_prob:
            kind, durations, (lo, hi), scale = 1, cfg.spike_durations, cfg.spike_multiplier, cfg.spike_scale
        elif u_onset[t] < cfg.spike_onset_prob + cfg.drop_onset_prob:
            kind, durations, (lo, hi), scale = -1, cfg.drop_durations, cfg.drop_multiplier, 1.0
        else:
            t += 1
            continue
        length = durations[0] + int(u_duration[t] * (durations[1] - durations[0] + 1))
        end = min(n, t + length)
        multiplier[t:end] = scale * (lo + u_multiplier[t] * (hi - lo))
        label[t:end] = kind
        onset[t:end] = t
        t = end
    return multiplier, label, onset
