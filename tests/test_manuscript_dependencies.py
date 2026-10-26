"""Current manuscript routes must not require optional legacy modules."""
import builtins
import importlib

import numpy as np
import torch


def test_current_training_and_loaders_do_not_import_legacy_modules(monkeypatch):
    blocked = (
        "phase1a_fire_tensor.experiments.baselines",
        "phase1a_fire_tensor.experiments.experiment_utils",
        "data.tcga_loader",
    )
    original = builtins.__import__

    def checked(name, *args, **kwargs):
        if any(name == item or name.startswith(item + ".") for item in blocked):
            raise AssertionError("Current manuscript route imported a legacy dependency")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked)
    adni = importlib.import_module("misfit.data.adni_loader")
    purged = importlib.import_module("misfit.correction_20261018.code.purged_harness")
    dynamics = importlib.import_module("misfit.experiments.run_training_dynamics")
    for module in (
        "misfit.correction_20261018.code.rerun_adni_purged",
        "misfit.experiments.run_oasis3_validation",
        "misfit.experiments.run_oasis3_cnn18_study",
        "misfit.manuscript_analysis.analyze_cnn_crossseed",
        "misfit.rank_validation.real_validation.run_validation",
    ):
        importlib.import_module(module)
    assert callable(adni.load_adni_smri)
    train, test = purged.purge_masks(
        torch.tensor([0, 1, 1, 2]), np.array(["p0", "p0", "p1", "p2"]), 0)
    assert train.tolist() == [False, False, True, True]
    assert test.tolist() == [True, False, False, False]
    monkeypatch.setattr(dynamics, "DEVICE", torch.device("cpu"))
    monkeypatch.setattr(dynamics, "N_EPOCHS", 2)
    generator = torch.Generator().manual_seed(117)
    _, scores = dynamics.train_log_dynamics(
        torch.randn(24, 8, generator=generator), torch.arange(24) % 2, 7)
    assert set(scores) == {"misfit", "aum_neg", "forgetting", "el2n"}
    assert all(values.shape == (24,) and np.isfinite(values).all()
               for values in scores.values())


def test_direct_clinical_input_keeps_the_original_hash_gate(tmp_path, monkeypatch):
    from misfit.manuscript_analysis import analyze_cnn_cdr_scanner as analysis
    cnn, clinical = tmp_path / "cnn.npz", tmp_path / "clinical.npz"
    cnn.write_bytes(b"synthetic CNN input")
    clinical.write_bytes(b"incorrect clinical archive")
    import hashlib
    monkeypatch.setattr(analysis, "CNN_SHA256", hashlib.sha256(cnn.read_bytes()).hexdigest())
    import pytest
    with pytest.raises(ValueError, match="SHA-256"):
        analysis.load_verified_inputs(cnn, clinical_npz=clinical)
