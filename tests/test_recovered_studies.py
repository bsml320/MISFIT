"""Integration checks for the recovered study modules; no cohort inputs."""
import json
from pathlib import Path

import numpy as np
import pytest
from misfit.paths import resolve_path
from misfit.crt_study.code import analysis, driver
from misfit.experiments import run_oasis3_cnn18_study as cnn18
from misfit.experiments import run_oasis3_cnn_upweight_study as upweight


def test_recovered_modules_use_the_same_backbone():
    from misfit.experiments.run_oasis3_cnn_backbone import CNN3D
    assert cnn18.CNN3D is CNN3D
    assert upweight.CNN3D is CNN3D
    assert np.isclose(analysis.pair_mse(np.array([1., 3., 5.])), 8.)


def test_upweight_selection_stays_within_source_pool_and_preserves_ties():
    score = np.array([100., 9., 9., 8.])
    source = np.array([3, 2, 1])
    assert upweight.topk(score, source, 2).tolist() == [2, 1]


def test_atomic_json_creates_its_output_directory(tmp_path):
    out = tmp_path / "dev" / "nested" / "record.json"
    driver.json_atomic(str(out), {"value": 7})
    assert json.loads(out.read_text()) == {"value": 7}
    assert not Path(str(out) + ".tmp").exists()


def test_protocol_gate_requires_local_freeze_and_rejects_changed_code(monkeypatch, tmp_path):
    monkeypatch.setattr(driver, "STUDY", str(tmp_path))
    monkeypatch.delenv("CRT_ALLOW_CODE_DRIFT", raising=False)
    with pytest.raises(RuntimeError, match="not frozen"):
        driver.gate_frozen()
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "example.py").write_text("value = 1")
    driver.json_atomic(str(tmp_path / "protocol" / "PROTOCOL_FROZEN.txt"),
                       {"files": {"code/example.py": "wrong-hash"}})
    with pytest.raises(RuntimeError, match="drifted"):
        driver.gate_frozen()


def test_nilearn_cache_override(monkeypatch, tmp_path):
    monkeypatch.setenv("MISFIT_NILEARN_DATA", str(tmp_path / "cache with spaces"))
    assert resolve_path("<NILEARN_DATA>/oasis1") == str(tmp_path / "cache with spaces" / "oasis1")
