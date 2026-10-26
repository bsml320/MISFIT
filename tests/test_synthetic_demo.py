"""End-to-end checks for the runnable synthetic example."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def invoke(output, *extra):
    env = dict(os.environ, PYTHONPATH=str(ROOT), MISFIT_DATA_DIR=str(output.parent / "absent-data"),
               MISFIT_REPORTING_REFERENCES=str(output.parent / "absent-references.json"))
    return subprocess.run(
        [sys.executable, "-m", "misfit.demo", "--output", str(output),
         "--epochs", "2", "--samples-per-environment", "20", *extra],
        cwd=output.parent, env=env, capture_output=True, text=True, timeout=60)


def test_synthetic_cli_runs_without_study_inputs_and_repeats(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    a, b = invoke(first), invoke(second)
    assert a.returncode == 0, a.stderr
    assert b.returncode == 0, b.stderr
    for name in ("synthetic_scores.csv", "summary.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    summary = json.loads((first / "summary.json").read_text())
    assert summary["synthetic"] is True
    assert summary["settings"]["device"] == "cpu"
    assert summary["n_samples"] == 60
    assert summary["n_tested"] == 60
    with (first / "synthetic_scores.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len({r["sample_id"] for r in rows}) == 60
    assert all(r["sample_id"].startswith("synthetic_") for r in rows)
    scores = np.array([float(r["misfit"]) for r in rows])
    errors = np.array([int(r["held_out_error"]) for r in rows])
    assert np.isfinite(scores).all() and (scores >= 0).all()
    assert all(int(r["held_out_error"]) == (int(r["prediction"]) != int(r["label"])) for r in rows)
    assert summary["held_out_accuracy"] == pytest.approx(1-errors.mean())
    if len(np.unique(errors)) == 2:
        assert summary["failure_detection_auroc"] == pytest.approx(roc_auc_score(errors, scores))
    else:
        assert summary["failure_detection_auroc"] is None


def test_synthetic_cli_does_not_overwrite_an_existing_run(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "summary.json"
    marker.write_text("preserve this run")
    result = invoke(output)
    assert result.returncode != 0
    assert "empty" in result.stderr.lower()
    assert marker.read_text() == "preserve this run"


@pytest.mark.parametrize("args", [
    ["--epochs", "0"], ["--samples-per-environment", "5"], ["--seed", "-1"]
])
def test_synthetic_cli_rejects_invalid_settings_before_writing(tmp_path, args):
    output = tmp_path / "invalid"
    result = invoke(output, *args)
    assert result.returncode == 2
    assert "error:" in result.stderr.lower()
    assert not output.exists()
