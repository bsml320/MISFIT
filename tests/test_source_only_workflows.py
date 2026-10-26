"""Exercise reporting in a source-only checkout using synthetic inputs."""
import importlib
import json
import runpy
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pytest
import misfit.paths

SOURCE = Path(__file__).resolve().parents[1]
PREFIXES = ["misfit"]


@pytest.fixture
def isolated_root(tmp_path, monkeypatch):
    monkeypatch.setattr(misfit.paths, "ROOT", tmp_path)
    monkeypatch.delenv("MISFIT_DATA_DIR", raising=False)
    monkeypatch.delenv("MISFIT_REPORTING_REFERENCES", raising=False)
    yield tmp_path
    plt.close("all")


def test_optional_reference_can_be_absent(isolated_root):
    assert misfit.paths.reporting_reference("misfit_anchor", required=False) is None


def test_explicit_broken_reference_is_not_silently_ignored(isolated_root, monkeypatch):
    monkeypatch.setenv("MISFIT_REPORTING_REFERENCES", str(isolated_root / "missing.json"))
    with pytest.raises(FileNotFoundError):
        misfit.paths.reporting_reference("misfit_anchor", required=False)


@pytest.mark.parametrize("prefix", PREFIXES)
def test_enrichment_report_needs_only_current_outputs(prefix, isolated_root, monkeypatch):
    mod = importlib.import_module(prefix + ".experiments.run_enrichment_media")
    monkeypatch.setattr(mod, "OUT", isolated_root)
    panel = {"adni": {"structural_misfit_train": {"mean": 0.4, "std": 0.05}}}
    result = {
        "shift_severity": {"per_env": {}, "analysis": {}, "verdict": "synthetic"},
        "stronger_panel": {"per_dataset": panel, "verdict": mod.verdict_panel(panel)},
        "clinical_utility": {"per_dataset": {}, "verdict": "synthetic"},
    }
    mod.write_md(result)
    assert "+0.400" in (isolated_root / "enrichment.md").read_text()
    verdict = mod.verdict_panel({"adni": {"other": {"mean": 0.2}}})
    assert "not assessed" in verdict.lower()


@pytest.mark.parametrize("prefix", PREFIXES)
def test_operator_summary_does_not_require_historical_results(prefix, isolated_root, monkeypatch):
    mod = importlib.import_module(prefix + ".experiments.run_operator_constraints_media")
    monkeypatch.setattr(mod, "N_SEEDS", 1)
    monkeypatch.setattr(mod, "VARIANTS", ["V0_frozen_random"])
    monkeypatch.setattr(mod, "collect_variant", lambda *args: (0.25, np.arange(3)))
    data = (np.ones((4, 2)), np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1]))
    result = mod.run_dataset("adni", lambda: data)
    assert result["anchor"] is None
    assert result["variants"]["V0_frozen_random"]["misfit_rho_mean"] == 0.25


@pytest.mark.parametrize("prefix", PREFIXES)
def test_reliability_report_does_not_abort_after_fitting(prefix, isolated_root, monkeypatch):
    mod = importlib.import_module(prefix + ".experiments.run_ml_reliability_media")
    monkeypatch.setattr(mod, "N_SEEDS", 1)
    record = {"tested": np.ones(4, dtype=bool), "misfit": np.arange(4), "err": np.array([0, 1, 0, 1])}
    monkeypatch.setattr(mod, "collect_seed", lambda *args: record)
    monkeypatch.setattr(mod, "block_a_conformal", lambda *args: {})
    monkeypatch.setattr(mod, "block_b_auroc", lambda *args: ({"misfit": {"auroc": 0.6}}, {}))
    monkeypatch.setattr(mod, "block_c_attribution", lambda *args: {"stability_spearman": None})
    monkeypatch.setattr(mod, "block_d_programs", lambda *args: {"stability_spearman": None, "top_program_consistency": None})
    data = (np.ones((4, 2)), np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1]))
    assert mod.run_dataset("adni", lambda: data)["n"] == 4


def test_forest_creates_output_directory(isolated_root, monkeypatch):
    ref = isolated_root / "references.json"
    ref.write_text(json.dumps({"cross_cohort_forest": [["Synthetic", "test only", 0.2, None, 0.7, 0.1, True, False]]}))
    monkeypatch.setenv("MISFIT_REPORTING_REFERENCES", str(ref))
    runpy.run_path(str(SOURCE / "misfit/experiments/fancy_figs/fig_cross_cohort_forest.py"))
    assert (isolated_root / "misfit/paper/figures/cross_cohort_forest.pdf").read_bytes().startswith(b"%PDF")


def test_dashboard_creates_output_directory(isolated_root, monkeypatch):
    ff = isolated_root / "data/oasis3/_fancy"
    res = isolated_root / "misfit/results/oasis3"
    ff.mkdir(parents=True)
    res.mkdir(parents=True)
    rng = np.random.default_rng(7)
    nib.save(nib.Nifti1Image(rng.uniform(-0.2, 0.2, (12, 12, 12)), np.diag([4, 4, 4, 1])), ff / "misfit_gm_spearman.nii.gz")
    np.savez(ff / "per_subject.npz", tested=np.ones(20, dtype=bool), misfit=np.arange(20), err=np.tile([0, 1], 10))
    (res / "oasis3_per_scanner.json").write_text(json.dumps([{"scanner": "Synthetic", "rho_mean": 0.2, "rho_std": 0.05}]))
    ref = isolated_root / "references.json"
    ref.write_text(json.dumps({"dashboard_rows": [["OASIS-3", 0.2, 0.05, 0.7, 0.05, "#336699", "D"]]}))
    monkeypatch.setenv("MISFIT_REPORTING_REFERENCES", str(ref))
    runpy.run_path(str(SOURCE / "misfit/experiments/fancy_figs/fig_dashboard.py"))
    assert (res / "fancy_figs/fig_dashboard.pdf").read_bytes().startswith(b"%PDF")


@pytest.mark.parametrize("provide_previous_mean", [False, True])
def test_cdr_export_from_new_fits_needs_no_prior_result(provide_previous_mean, isolated_root, monkeypatch):
    import pandas as pd
    collector = importlib.import_module("misfit.experiments.run_clinical_stratified_media")
    data = isolated_root / "data/oasis3"
    data.mkdir(parents=True)
    sessions = np.array([f"Synthetic{i}_d0001" for i in range(12)])
    labels = np.tile([0, 1, 1], 4)
    scores = np.arange(12, dtype=float) / 12
    np.savez(data / "oasis3_features.npz", features=np.ones((12, 2)), subjects=np.arange(12),
             sessions=sessions, label=labels, age=np.arange(12)+60,
             sex=np.tile(["M", "F"], 6), scanner=np.repeat(["scannerA", "scannerB"], 6))
    pd.DataFrame({"session": sessions, "CDRTOT": np.tile([0, 0.5, 1], 4)}).to_csv(data / "oasis3_mr_labels.csv", index=False)
    if provide_previous_mean:
        old = isolated_root / "misfit/results/oasis3"
        old.mkdir(parents=True)
        np.save(old / "oasis3_misfit_leakagefree_5seed.npy", scores)
    def collect(*args):
        pred = np.tile([0, 0, 1], 4)
        return {"misfit": scores, "err": pred != labels, "pred": pred, "tested": np.ones(12, dtype=bool)}
    monkeypatch.setattr(collector, "collect", collect)
    runpy.run_path(str(SOURCE / "misfit/experiments/export_oasis3_cdr_records.py"))
    destination = isolated_root / "misfit/results/provenance"
    assert pd.read_csv(destination / "oasis3_cdr_participant_records.csv").shape == (12, 30)
    meta = json.loads((destination / "oasis3_cdr_participant_records.json").read_text())
    assert meta["verification_from_csv"]["mean_matches_saved_npy"] is (True if provide_previous_mean else None)


def test_corrected_figure_cli_creates_output_directory(isolated_root, monkeypatch):
    import sys
    results = isolated_root / "misfit/correction_20261018/results"
    results.mkdir(parents=True)
    (results / "corrected_v2_summaries.json").write_text(json.dumps({
        "within_env_canonical": {"mean_within_env_rho": 0.2}, "pooled_auroc": {"mean": 0.7, "std": 0.05}}))
    ref = isolated_root / "references.json"
    ref.write_text(json.dumps({"cross_cohort_forest": [["Synthetic", "test", 0.3, None, 0.6, 0.1, True, False]]}))
    monkeypatch.setenv("MISFIT_REPORTING_REFERENCES", str(ref))
    script = SOURCE / "misfit/correction_20261018/integration/code/regen_figures_corrected.py"
    monkeypatch.setattr(sys, "argv", [str(script), "forest"])
    runpy.run_path(str(script), run_name="__main__")
    assert (isolated_root / "misfit/paper/figures/cross_cohort_forest.pdf").read_bytes().startswith(b"%PDF")
