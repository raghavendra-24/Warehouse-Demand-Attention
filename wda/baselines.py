"""The simple baselines (PRD §17, A-15), on raw windows of shape (N, 24) holding y(t−23) … y(t).

B1 last observation   ŷ(t+1) = y(t)
B2 24-hour mean       ŷ(t+1) = (1/24) Σ_{i=0..23} y(t−i)
B3 seasonal naive     ŷ(t+1) = y(t−23), the same hour yesterday (the oldest value in the window)
B4 hour-of-week mean  ŷ(t+1) = mean of the training targets at the same hour of the week as t+1
                      (PHASE0 Amendment 3; built from training targets only)
"""

import numpy as np


def last_observation(raw: np.ndarray) -> np.ndarray:
    return raw[:, -1]


def moving_average(raw: np.ndarray) -> np.ndarray:
    return raw.mean(axis=1)


def seasonal_naive(raw: np.ndarray) -> np.ndarray:
    return raw[:, 0]


BASELINES = {"B1": last_observation, "B2": moving_average, "B3": seasonal_naive}


def hour_of_week_profile(windows: dict, train_mask: np.ndarray) -> np.ndarray:
    """168 means of the training targets, indexed by day_of_week · 24 + hour_of_day of the target."""
    how = windows["target_day"] * 24 + windows["target_hour"]
    target = windows["target"]
    return np.array([target[train_mask & (how == k)].mean() for k in range(168)])


def hour_of_week_mean(windows: dict, profile: np.ndarray) -> np.ndarray:
    return profile[windows["target_day"] * 24 + windows["target_hour"]]

