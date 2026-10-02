"""Synthetic data, windows and splits (§4, §18, §23; A-04, A-09, A-11, A-12, A-13, A-20)."""

from dataclasses import replace

import numpy as np
import torch

from wda import config
from wda.baselines import hour_of_week_mean, hour_of_week_profile, last_observation, moving_average, seasonal_naive
from wda.toy_data import make_recall
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
    assert len(make_windows(control)["target"]) == config.SHIFT_WEEKS * 168 and control["day_of_week"][24] == 0


def test_shift_series_use_their_own_seed_not_the_training_data():
    train = generate_series(config.WAREHOUSE)["demand"]
    shift = generate_series(config.SHIFT_SERIES["control"])["demand"]
    assert not np.array_equal(shift[:len(train)], train)


# --- Toy task (A-18) --------------------------------------------------------


def test_toy_tokens_have_the_documented_layout():
    cfg = config.TOY
    tokens, target = make_recall(cfg, 200, seed=0)
    assert tokens.shape == (200, cfg.n_tokens, cfg.token_dim) and target.shape == (200,)
    pairs, query = tokens[:, :cfg.n_pairs], tokens[:, cfg.n_pairs]
    assert (pairs[..., :cfg.n_keys].sum(-1) == 1).all() and (pairs[..., cfg.n_keys:-1].sum(-1) == 1).all()
    assert (pairs[..., -1] == 0).all()
    assert (query[:, :cfg.n_keys].sum(-1) == 1).all() and (query[:, cfg.n_keys:-1] == 0).all()
    assert (query[:, -1] == 1).all()
    assert (tokens.pow(2).sum(-1) == 2).all()        # ‖x‖² = 2 for every token (A-19 initialisation)


def test_toy_target_is_the_value_of_the_queried_key():
    cfg = config.TOY
    tokens, target = make_recall(cfg, 300, seed=1)
    for i in range(300):
        keys = tokens[i, :cfg.n_pairs, :cfg.n_keys].argmax(-1)
        values = tokens[i, :cfg.n_pairs, cfg.n_keys:-1].argmax(-1)
        asked = tokens[i, cfg.n_pairs, :cfg.n_keys].argmax()
        assert len(set(keys.tolist())) == cfg.n_pairs and len(set(values.tolist())) == cfg.n_pairs
        assert target[i] == values[keys == asked].item()


def test_toy_splits_use_different_seeds():
    cfg = config.TOY
    a, _ = make_recall(cfg, 50, cfg.seed_train)
    b, _ = make_recall(cfg, 50, cfg.seed_val)
    assert not torch.equal(a, b)
    assert torch.equal(a, make_recall(cfg, 50, cfg.seed_train)[0])


# --- Baselines (§17, A-15, FAC-34) -----------------------------------------

def test_baselines_on_a_ramp_catch_wrong_lag_or_window():
    """Predicting y(24) from y(0) … y(23) on y(t) = t: B1 = 23, B2 = 11.5, B3 = 0."""
    raw = np.arange(24, dtype=float)[None, :]
    assert last_observation(raw)[0] == 23
    assert moving_average(raw)[0] == 11.5
    assert seasonal_naive(raw)[0] == 0


def test_seasonal_naive_is_the_same_hour_yesterday():
    s = generate_series(config.WAREHOUSE)
    w = make_windows(s)
    i = 1000
    tau = w["target_index"][i]
    assert seasonal_naive(w["raw_window"])[i] == s["demand"][tau - 24]
    assert s["hour_of_day"][tau - 24] == s["hour_of_day"][tau]


def test_hour_of_week_baseline_uses_training_targets_only():
    w = make_windows(generate_series(config.WAREHOUSE))
    m = split_masks(w)
    profile = hour_of_week_profile(w, m["train"])
    assert profile.shape == (168,)
    k = 2 * 24 + 14                                      # Wednesday 14:00
    how = w["target_day"] * 24 + w["target_hour"]
    assert np.isclose(profile[k], w["target"][m["train"] & (how == k)].mean())
    changed = dict(w, target=np.where(m["train"], w["target"], 1e6))
    np.testing.assert_array_equal(hour_of_week_profile(changed, m["train"]), profile)
    pred = hour_of_week_mean(w, profile)
    assert pred[how == k].min() == pred[how == k].max() == profile[k]



def test_select_keeps_rows_aligned():
    from wda.warehouse_data import select
    w = make_windows(generate_series(config.WAREHOUSE))
    val = select(w, split_masks(w)["val"])
    assert len(val["target"]) == len(val["raw_window"]) == len(val["group"]) == 1344
    assert np.array_equal(val["raw_window"][0], w["raw_window"][split_masks(w)["val"]][0])
