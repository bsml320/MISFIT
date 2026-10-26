"""Synthetic checks of the released implementation, not new cohort evidence."""
import importlib
from unittest.mock import Mock
import numpy as np
import torch
from misfit.experiments import run_honest_estimator_table as H
from misfit.experiments import run_clinical_stratified_media as C
from misfit.correction_20261018.code.purged_harness import purge_masks, random_removal_masks
from misfit.correction_20261018.code.purged_loader import fold_pca

def test_adni_loader_does_not_require_unreleased_benchmark_modules():
    m = importlib.import_module("misfit.data.adni_loader")
    assert callable(m.load_adni_smri)

def test_transfer_helpers_import_from_the_supplied_module():
    m = importlib.import_module("misfit.experiments.run_transfer_rescue")
    assert callable(m._recover_basis)

def test_purge_removes_all_acquisitions_of_held_out_participants():
    sites = torch.tensor([0, 0, 1, 1, 2, 2])
    participants = np.array(["a", "b", "a", "c", "b", "d"])
    tr, te = purge_masks(sites, participants, 0)
    assert np.where(tr.numpy())[0].tolist() == [3, 5]
    assert np.where(te.numpy())[0].tolist() == [0, 1]
    for held in range(3):
        tr, te = purge_masks(sites, participants, held)
        assert not set(participants[tr.numpy()]) & set(participants[te.numpy()])
        assert torch.equal(te, sites == held)

def test_size_matched_control_matches_purge_size_and_repeats():
    sites = torch.tensor([0, 0, 1, 1, 2, 2])
    participants = np.array(["a", "b", "a", "c", "b", "d"])
    purged, test = purge_masks(sites, participants, 0)
    a, at = random_removal_masks(sites, participants, 0, 42)
    b, bt = random_removal_masks(sites, participants, 0, 42)
    assert a.sum() == purged.sum()
    assert torch.equal(a, b) and torch.equal(at, test) and torch.equal(at, bt)
    assert not torch.any(a & at)

def test_fold_pca_is_invariant_to_held_out_data():
    raw = np.random.default_rng(7).normal(size=(20, 8))
    train = np.arange(14)
    changed = raw.copy()
    changed[14:] += 1000
    a = fold_pca(raw, train, pca_dim=4)(train)
    b = fold_pca(changed, train, pca_dim=4)(train)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(a.mean(0), torch.zeros(4), atol=1e-6, rtol=0)
    torch.testing.assert_close(a.std(0), torch.ones(4), atol=1e-6, rtol=0)

def test_training_retains_frozen_programs_and_fixed_loadings(monkeypatch):
    monkeypatch.setattr(H, "N_EPOCHS", 2)
    monkeypatch.setattr(H, "BATCH_SIZE", 8)
    recovery = Mock(side_effect=AssertionError("Inactive recovery branch executed"))
    monkeypatch.setattr(H, "recover_programs", recovery)
    original = H.TwoStageTrainer.stage2_train
    records = []
    def observed(self, data, *args, **kwargs):
        before = {k: v.clone() for k, v in self.model.program_dict.state_dict().items()}
        loadings = data["loadings"].clone()
        result = original(self, data, *args, **kwargs)
        records.append((before, self.model.program_dict.state_dict(), loadings, data["loadings"]))
        return result
    monkeypatch.setattr(H.TwoStageTrainer, "stage2_train", observed)
    x = torch.tensor(np.random.default_rng(9).normal(size=(24, 6)), dtype=torch.float32)
    y = (x[:, 0] > 0).long()
    sites = torch.arange(24) % 2
    a, la = H._train_fire(x, y, sites, 2, 6, 42)
    b, lb = H._train_fire(x, y, sites, 2, 6, 42)
    recovery.assert_not_called()
    for before, after, old_l, new_l in records:
        for key in before:
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
        torch.testing.assert_close(old_l, new_l, rtol=0, atol=0)
    for key, tensor in a.state_dict().items():
        torch.testing.assert_close(tensor, b.state_dict()[key], rtol=0, atol=0)
    torch.testing.assert_close(la, lb, rtol=0, atol=0)
    test_loadings = H._predict_loadings(x, la, x[:5])
    a.eval()
    with torch.no_grad():
        logits = a(x[:5], test_loadings, torch.zeros(5, dtype=torch.long))
    assert logits.shape == (5, 2)
    assert torch.isfinite(logits).all()

def test_collector_averages_only_refits_containing_each_subject(monkeypatch):
    sites = torch.arange(3).repeat_interleave(10)
    x = torch.arange(30, dtype=torch.float32).reshape(-1, 1)
    y = torch.zeros(30, dtype=torch.long)
    calls = []
    class ConstantModel:
        def __init__(self, value): self.value = value
        def __call__(self, x, loadings, env):
            return torch.tensor([0., self.value]).repeat(len(x), 1)
    def train(x, y, env, n_envs, dim, seed):
        held = next(e for e in range(3) if not ((x[:, 0] // 10) == e).any())
        calls.append(held)
        return ConstantModel(float(held)), torch.zeros(len(x), H.N_PROGRAMS)
    monkeypatch.setattr(C, "_train_fire", train)
    monkeypatch.setattr(C, "_predict_loadings", lambda x, l, t: torch.zeros(len(t), H.N_PROGRAMS))
    result = C.collect(x, y, sites, 42)
    assert calls == [0, 1, 2] and result["tested"].all()
    for site in range(3):
        expected = np.mean([np.logaddexp(0., e) for e in range(3) if e != site])
        np.testing.assert_allclose(result["misfit"][sites.numpy() == site], expected, rtol=1e-6)
