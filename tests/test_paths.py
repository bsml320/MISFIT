from pathlib import Path
import pytest
from misfit.paths import ROOT, resolve_path

def test_repository_root_is_independent_of_working_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert resolve_path("<REPO_ROOT>/misfit/core/types.py") == str(ROOT / "misfit/core/types.py")

def test_data_root_supports_spaces(monkeypatch, tmp_path):
    p = tmp_path / "authorized data"
    monkeypatch.setenv("MISFIT_DATA_DIR", str(p))
    assert resolve_path("<REPO_ROOT>/data/oasis3/features.npz") == str(p / "oasis3/features.npz")

@pytest.mark.parametrize("token,variable", [("<DATA_SHARE>", "MISFIT_DATA_SHARE"), ("<TOOLS>", "MISFIT_TOOLS")])
def test_external_paths_require_explicit_configuration(monkeypatch, tmp_path, token, variable):
    monkeypatch.delenv(variable, raising=False)
    with pytest.raises(RuntimeError, match=variable):
        resolve_path(token + "/input")
    monkeypatch.setenv(variable, str(tmp_path))
    assert resolve_path(token + "/input") == str(tmp_path / "input")

def test_ordinary_strings_are_not_changed():
    assert resolve_path("relative/path") == "relative/path"
