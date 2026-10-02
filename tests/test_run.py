"""Run helper: stale outputs are cleared, records are complete, weights round-trip."""

import json

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


def test_finish_records_config_versions_and_runtime(results_dir):
    run.start("stage")
    run.finish("stage", {"mae": torch.tensor(1.5)}, "| x |\n|---|", {"data": config.WAREHOUSE})
    record = json.loads((results_dir / "stage" / "metrics.json").read_text())
    assert record["metrics"]["mae"] == 1.5
    assert record["configs"]["data"]["seed"] == config.WAREHOUSE.seed
    assert {"python", "torch", "numpy", "cpu"} <= record["environment"].keys()
    assert record["runtime_seconds"] >= 0
    assert (results_dir / "stage" / "table.md").read_text().startswith("| x |")


def test_weights_round_trip_without_pickle(results_dir):
    out = run.start("stage")
    run.save_weights(out / "w.pt", {"w": torch.arange(3.0)}, {"mean": 2.0, "sd": 0.5})
    loaded = run.load_weights(out / "w.pt")
    assert torch.equal(loaded["state_dict"]["w"], torch.arange(3.0))
    assert loaded["extra"] == {"mean": 2.0, "sd": 0.5}


def test_missing_weights_name_the_problem(results_dir):
    with pytest.raises(FileNotFoundError, match="run the stage that writes it first"):
        run.load_weights(results_dir / "absent.pt")
