"""The simple baselines (PRD §17, A-15), on raw windows of shape (N, 24) holding y(t−23) … y(t).

B1 last observation   ŷ(t+1) = y(t)
B2 24-hour mean       ŷ(t+1) = (1/24) Σ_{i=0..23} y(t−i)
B3 seasonal naive     ŷ(t+1) = y(t−23), the same hour yesterday (the oldest value in the window)
"""

import numpy as np


def last_observation(raw: np.ndarray) -> np.ndarray:
    return raw[:, -1]


def moving_average(raw: np.ndarray) -> np.ndarray:
    return raw.mean(axis=1)


def seasonal_naive(raw: np.ndarray) -> np.ndarray:
    return raw[:, 0]


BASELINES = {"B1": last_observation, "B2": moving_average, "B3": seasonal_naive}
