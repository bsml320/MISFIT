"""Reporting consumes external aggregates; source code has no result fallback."""
import json
import pytest
import misfit.paths


def test_missing_reporting_reference_is_explicit(tmp_path, monkeypatch):
    monkeypatch.setenv("MISFIT_REPORTING_REFERENCES", str(tmp_path / "absent.json"))
    with pytest.raises(FileNotFoundError, match="MISFIT_REPORTING_REFERENCES"):
        misfit.paths.reporting_reference("synthetic")


def test_external_reference_is_read_without_an_embedded_fallback(tmp_path, monkeypatch):
    source = tmp_path / "references.json"
    source.write_text(json.dumps({"synthetic": {"value": 7}}))
    monkeypatch.setenv("MISFIT_REPORTING_REFERENCES", str(source))
    assert misfit.paths.reporting_reference("synthetic") == {"value": 7}
    with pytest.raises(KeyError, match="missing"):
        misfit.paths.reporting_reference("missing")
