"""Synthetic data, windows and splits (§4, §18, §23; A-04, A-09, A-11, A-12, A-13, A-20)."""

from dataclasses import replace

import numpy as np

from wda import config
from wda.warehouse_data import generate_series, make_windows, split_masks, training_stats


def test_same_seed_gives_identical_series_and_different_seed_differs():
    a, b = generate_series(config.WAREHOUSE), generate_series(config.WAREHOUSE)
    for key in a:
        assert np.array_equal(a[key], b[key]), key
    other = generate_series(replace(config.WAREHOUSE, seed=config.WAREHOUSE.seed + 1))
    assert not np.array_equal(a["demand"], other["demand"])


def test_demand_is_a_non_negative_integer_count():
    y = generate_series(config.WAREHOUSE)["demand"]
    assert y.dtype.kind == "i" and (y >= 0).all()


def test_window_alignment_catches_off_by_one():
    s = generate_series(config.WAREHOUSE)
    w = make_windows(s)
    assert w["target_index"][0] == config.WINDOW
    for i in (0, 500, len(w["target"]) - 1):
        tau = w["target_index"][i]
        assert np.array_equal(w["raw_window"][i], s["demand"][tau - 24:tau])   # y(τ−24) … y(τ−1)
        assert w["target"][i] == s["demand"][tau]                               # target y(τ)
        assert w["target_hour"][i] == s["hour_of_day"][tau]


def test_splits_are_whole_weeks_in_order_with_no_leakage():
    w = make_windows(generate_series(config.WAREHOUSE))
    m = split_masks(w)
    assert (m["train"].sum(), m["val"].sum(), m["test"].sum()) == (6024, 1344, 1344)
    assert not (m["train"] & m["val"]).any() and not (m["val"] & m["test"]).any()
    tau = w["target_index"]
    assert tau[m["train"]].max() < tau[m["val"]].min() <= tau[m["val"]].max() < tau[m["test"]].min()


def test_standardisation_uses_training_targets_only():
    w = make_windows(generate_series(config.WAREHOUSE))
    m = split_masks(w)
    mean, sd = training_stats(w, m)
    assert np.isclose(mean, w["target"][m["train"]].mean()) and np.isclose(sd, w["target"][m["train"]].std())
    # Changing validation or test targets must not move the statistics.
    changed = dict(w, target=np.where(m["train"], w["target"], 1e6))
    assert training_stats(changed, m) == (mean, sd)


def test_error_groups_follow_the_event_labels():
    s = generate_series(config.WAREHOUSE)
    w = make_windows(s)
    label = s["event_label"]
    for i in np.flatnonzero(w["group"] == "inside")[:20]:
        assert label[w["target_index"][i]] != 0
    for i in np.flatnonzero(w["group"] == "after")[:20]:
        tau = w["target_index"][i]
        assert label[tau] == 0 and (label[tau - 24:tau] != 0).any()
    for i in np.flatnonzero(w["group"] == "other")[:20]:
        tau = w["target_index"][i]
        assert (label[tau - 24:tau + 1] == 0).all()


def test_shift_series_share_one_event_timeline():
    series = {k: generate_series(v) for k, v in config.SHIFT_SERIES.items()}
    control = series["control"]
    for name, s in series.items():
        assert np.array_equal(s["event_label"], control["event_label"]), name
        assert np.array_equal(s["event_onset"], control["event_onset"]), name
    spikes = control["event_label"] == 1
    np.testing.assert_allclose(series["larger_spikes"]["level"][spikes] / control["level"][spikes], 2.0)
    assert len(make_windows(control)["target"]) == 1344 and control["day_of_week"][24] == 0   # Monday


def test_shift_series_use_their_own_seed_not_the_training_data():
    train = generate_series(config.WAREHOUSE)["demand"][:1368]
    assert not np.array_equal(generate_series(config.SHIFT_SERIES["control"])["demand"], train)
