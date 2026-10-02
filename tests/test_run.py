"""Run helper: stale outputs are cleared, records are complete, weights round-trip."""

import json

import numpy as np
import pytest
import torch

from wda import config, run


@pytest.fixture
def results_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "RESULTS", tmp_path)
    return tmp_path


def test_start_clears_stale_outputs(results_dir):
    out = run.start("stage")
    (out / "old.json").write_text("{}")
    out = run.start("stage")
    assert list(out.iterdir()) == []


def test_finish_separates_reproducible_metrics_from_run_info(results_dir):
    run.start("stage")
    run.finish("stage", {"mae": torch.tensor(1.5)}, "| x |\n|---|", {"data": config.WAREHOUSE})
    record = json.loads((results_dir / "stage" / "metrics.json").read_text())
    assert record["metrics"]["mae"] == 1.5
    assert record["configs"]["data"]["seed"] == config.WAREHOUSE.seed
    assert "runtime_seconds" not in record and "environment" not in record      # reproducible content only
    info = json.loads((results_dir / "stage" / "run.json").read_text())
    assert {"python", "torch", "numpy", "cpu", "git", "final_test"} <= info["environment"].keys()
    assert info["runtime_seconds"] >= 0
    assert (results_dir / "stage" / "table.md").read_text().startswith("| x |")


def test_metrics_reproduce_exactly_on_rerun(results_dir):
    for _ in range(2):
        run.start("stage")
        run.finish("stage", {"x": 0.1 + 0.2}, "t", {"data": config.WAREHOUSE})
        if _ == 0:
            first = (results_dir / "stage" / "metrics.json").read_text()
    assert (results_dir / "stage" / "metrics.json").read_text() == first


def test_finish_writes_numpy_scalars_and_booleans(results_dir):
    run.start("stage")
    run.finish("stage", {"ok": np.bool_(True), "x": np.float64(0.5), "n": np.int64(3), "v": np.arange(2),
                         "bad": float("nan")}, "t", {})
    record = json.loads((results_dir / "stage" / "metrics.json").read_text())
    assert record["metrics"] == {"ok": True, "x": 0.5, "n": 3, "v": [0, 1], "bad": None}


@pytest.mark.parametrize("name", ["", "../x", "Data", "a/b"])
def test_start_rejects_unsafe_stage_names(results_dir, name):
    with pytest.raises(ValueError):
        run.start(name)


def test_weight_extras_with_numpy_values_load_safely(results_dir):
    out = run.start("stage")
    run.save_weights(out / "w.pt", {"w": torch.ones(1)}, {"mean": np.float64(2.0), "cfg": config.WAREHOUSE_MODEL})
    loaded = run.load_weights(out / "w.pt")
    assert loaded["extra"]["mean"] == 2.0 and loaded["extra"]["cfg"]["d_model"] == 16


def test_weights_round_trip_without_pickle(results_dir):
    out = run.start("stage")
    run.save_weights(out / "w.pt", {"w": torch.arange(3.0)}, {"mean": 2.0, "sd": 0.5})
    loaded = run.load_weights(out / "w.pt")
    assert torch.equal(loaded["state_dict"]["w"], torch.arange(3.0))
    assert loaded["extra"] == {"mean": 2.0, "sd": 0.5}


def test_missing_weights_name_the_problem(results_dir):
    with pytest.raises(FileNotFoundError, match="run the stage that writes it first"):
        run.load_weights(results_dir / "absent.pt")
