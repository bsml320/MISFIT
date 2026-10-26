"""The paper release is self-contained under the misfit package."""
import ast
import os
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
OLD_PROJECTS = {"core", "data", "phase1a_fire_tensor", "phase1b_fire_manifold", "misfit_paths", "misfit_demo", "phase1b_mia"}


def test_only_the_paper_package_is_distributed():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert config["tool"]["setuptools"]["packages"]["find"]["include"] == ["misfit*"]
    assert not config["tool"]["setuptools"].get("py-modules")
    for name in OLD_PROJECTS:
        # An ignored data directory may be created by a local cohort workflow.
        assert not (ROOT / name / "__init__.py").exists()
        assert not (ROOT / (name + ".py")).exists()


def test_paper_source_has_no_old_project_imports():
    violations = []
    for path in (ROOT / "misfit").rglob("*.py"):
        if path.name.startswith("._"):
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                names = [node.module or ""]
            for name in names:
                if name.split(".")[0] in OLD_PROJECTS:
                    violations.append((str(path.relative_to(ROOT)), name))
    assert not violations, violations


def test_demo_works_when_old_project_imports_are_blocked(tmp_path):
    script = """
import importlib.abc
import sys
class BlockOldProjects(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"core", "data", "phase1a_fire_tensor", "phase1b_fire_manifold", "misfit_paths", "misfit_demo", "phase1b_mia"}:
            raise AssertionError("Old project imported: " + fullname)
        return None
sys.meta_path.insert(0, BlockOldProjects())
from misfit.demo import main
main(["--output", sys.argv[1], "--epochs", "2", "--samples-per-environment", "20"])
"""
    run = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "demo")],
        cwd=tmp_path, env=dict(os.environ, PYTHONPATH=str(ROOT)),
        capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    assert (tmp_path / "demo" / "synthetic_scores.csv").is_file()


def test_paths_and_collector_binding_follow_the_paper_package():
    from misfit import paths
    from misfit.manuscript_analysis import analyze_cdr_stage
    assert paths.ROOT == ROOT
    assert analyze_cdr_stage.METHOD_SOURCE == ROOT / "misfit/experiments/run_clinical_stratified_media.py"
    assert analyze_cdr_stage.METHOD_SOURCE.is_file()


def test_abide_download_cache_stays_outside_the_source_package():
    from misfit.data.abide_loader import DEFAULT_DATA_DIR
    assert DEFAULT_DATA_DIR == ROOT / "data/abide"
