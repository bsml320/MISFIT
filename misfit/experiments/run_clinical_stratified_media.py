"""Clinical, class-stratified, and OOD-diagnostic analyses for the MedIA revision.

Reviewer feedback asked for: class-balanced and protocol-stratified reliability
metrics; clinically meaningful selective-prediction reporting (sensitivity /
specificity / false-negative / false-positive among accepted subjects, and
whether abstention is class-disproportionate); and OOD-assumption diagnostics
(per-fold cross-moment condition numbers and recovered-eigenvalue spectra).

Reuses the validated IOP harness (deployable training-side misfit, binary
per-subject-error target, leave-one-environment-out, 5 seeds). float64 covariance,
no torch.inverse. Output: runs/external_robustness/clinical_stratified.{json,md}
"""
from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path
import json, sys, warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as sp_stats
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

sys.path.insert(0, _resolve_path("<REPO_ROOT>"))
from misfit.experiments.run_honest_estimator_table import (  # noqa: E402
    N_PROGRAMS, N_SEEDS, SEED_BASE, _predict_loadings, _rho, _train_fire,
    load_adni, load_oasis_ds,
)
from misfit.tensor_spectral.recovery import recover_programs  # noqa: E402
from misfit.core.types import TensorConfig  # noqa: E402
from misfit.core.utils import seed_everything  # noqa: E402
from misfit.data.mri_multisite import MRIData, mri_to_multienv_format  # noqa: E402
warnings.filterwarnings("ignore")
OUT = Path(_resolve_path("<REPO_ROOT>/misfit/results"))
COVERAGES = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]


def collect(features, targets, site_ids, seed):
    """Leave-one-environment-out harness: per-subject misfit and held-out predictions.

    For each held-out site, trains the model on the remaining sites and
    accumulates each training subject's per-sample training CE; a subject's
    misfit is that loss averaged over the folds in which it was a training
    subject. Held-out subjects receive predictions via transferred loadings.

    Args:
        features: (N, dim) float tensor.
        targets: (N,) class labels.
        site_ids: (N,) environment ids.
        seed: Seed passed to seed_everything before the fold loop.

    Returns:
        Dict with per-subject arrays: misfit, pred, true, err (binary), site,
        and tested (subjects that were ever held out).
    """
    n_sites = int(site_ids.max().item()) + 1
    dim = features.shape[1]; N = features.shape[0]
    seed_everything(seed)
    mis_sum = np.zeros(N); mis_cnt = np.zeros(N)
    pred = np.full(N, -1); tested = np.zeros(N, dtype=bool)
    for held in range(n_sites):
        tr = site_ids != held; te = site_ids == held
        if te.sum() < 5 or tr.sum() < 20:
            continue
        trf, trt, trs = features[tr], targets[tr], site_ids[tr]
        tef, tet = features[te], targets[te]
        uniq = trs.unique(); smap = {int(z): i for i, z in enumerate(uniq)}
        rem = torch.tensor([smap[int(z)] for z in trs]); nts = len(uniq)
        ti_tr = torch.where(tr)[0].numpy(); ti_te = torch.where(te)[0].numpy()
        model, loadings = _train_fire(trf, trt, rem, nts, dim, seed)
        with torch.no_grad():
            tr_loss = F.cross_entropy(model(trf, loadings, rem), trt.long(),
                                      reduction="none").numpy()
        mis_sum[ti_tr] += tr_loss; mis_cnt[ti_tr] += 1.0
        tl = _predict_loadings(trf, loadings, tef)
        with torch.no_grad():
            pl = model(tef, tl, torch.zeros(len(tef), dtype=torch.long)).argmax(-1).numpy()
        pred[ti_te] = pl; tested[ti_te] = True
    misfit = mis_sum / np.maximum(mis_cnt, 1.0)
    true = targets.numpy().astype(int)
    err = (pred != true).astype(float)
    return {"misfit": misfit, "pred": pred, "true": true, "err": err,
            "site": site_ids.numpy(), "tested": tested}


def _sens_spec(true, pred, mask):
    t = true[mask]; p = pred[mask]
    pos = t == 1; neg = t == 0
    sens = float((p[pos] == 1).mean()) if pos.sum() else float("nan")
    spec = float((p[neg] == 0).mean()) if neg.sum() else float("nan")
    return sens, spec


def block_stratified(data):
    """Class- and protocol-stratified reliability metrics, pooled over seeds.

    Reports overall and per-class failure-detection AUROC and rho, balanced
    accuracy, sensitivity/specificity, and the same associations after
    excluding single-class protocols or keeping only near-balanced ones.
    """
    # pool over seeds (each seed's per-subject arrays are aligned)
    mis = np.concatenate([d["misfit"][d["tested"]] for d in data])
    err = np.concatenate([d["err"][d["tested"]] for d in data])
    true = np.concatenate([d["true"][d["tested"]] for d in data])
    pred = np.concatenate([d["pred"][d["tested"]] for d in data])
    site = np.concatenate([d["site"][d["tested"]] for d in data])
    out = {}
    # overall + per-class failure-detection AUROC and within-class misfit->err rho
    def auroc(m):
        e = err[m]
        return float(roc_auc_score(e, mis[m])) if 0 < e.sum() < len(e) else None
    out["auroc_overall"] = auroc(np.ones(len(mis), bool))
    out["auroc_impaired"] = auroc(true == 1)
    out["auroc_control"] = auroc(true == 0)
    out["rho_overall"] = float(_rho(mis, err))
    out["rho_impaired"] = float(_rho(mis[true == 1], err[true == 1]))
    out["rho_control"] = float(_rho(mis[true == 0], err[true == 0]))
    out["balanced_accuracy"] = float(balanced_accuracy_score(true, pred))
    s, sp = _sens_spec(true, pred, np.ones(len(true), bool))
    out["sensitivity"] = s; out["specificity"] = sp
    out["prevalence"] = float(true.mean())
    # per-protocol prevalence + identify single-class protocols
    prev = {int(e_): float(true[site == e_].mean()) for e_ in np.unique(site)}
    out["protocol_prevalence"] = prev
    single = [e_ for e_, p in prev.items() if p in (0.0, 1.0) or p < 0.05 or p > 0.95]
    balanced = [e_ for e_, p in prev.items() if 0.35 <= p <= 0.65]
    # exclude single-class protocols
    keep = ~np.isin(site, single)
    out["single_class_protocols"] = single
    out["auroc_excl_singleclass"] = auroc(keep) if keep.sum() else None
    out["rho_excl_singleclass"] = float(_rho(mis[keep], err[keep])) if keep.sum() else None
    # near-balanced protocols only
    bal = np.isin(site, balanced)
    out["balanced_protocols"] = balanced
    out["rho_balanced_only"] = float(_rho(mis[bal], err[bal])) if bal.sum() > 5 else None
    return out


def block_selective(data):
    """Clinical selective-prediction table over COVERAGES, pooled over seeds.

    Subjects are accepted in ascending misfit order; each row reports accepted
    accuracy, sensitivity, specificity, FN/FP rates among accepted subjects,
    and the impaired fraction of the abstained set.
    """
    mis = np.concatenate([d["misfit"][d["tested"]] for d in data])
    err = np.concatenate([d["err"][d["tested"]] for d in data])
    true = np.concatenate([d["true"][d["tested"]] for d in data])
    pred = np.concatenate([d["pred"][d["tested"]] for d in data])
    order = np.argsort(mis)
    rows = []
    for cov in COVERAGES:
        n = max(1, int(round(cov * len(mis))))
        acc_idx = order[:n]; m = np.zeros(len(mis), bool); m[acc_idx] = True
        sens, spec = _sens_spec(true, pred, m)
        e = err[m]
        # FN among accepted = predicted negative but truly positive; FP = predicted pos truly neg
        fn = float(((pred[m] == 0) & (true[m] == 1)).mean())
        fp = float(((pred[m] == 1) & (true[m] == 0)).mean())
        # abstention rate by class among the abstained (1-cov) set
        ab = ~m
        ab_pos = float((true[ab] == 1).mean()) if ab.sum() else float("nan")
        rows.append({"coverage": cov, "accuracy": float(1 - e.mean()),
                     "sensitivity": sens, "specificity": spec,
                     "fn_among_accepted": fn, "fp_among_accepted": fp,
                     "frac_abstained_impaired": ab_pos})
    return rows


def block_ood_diag(features, targets, site_ids, seed):
    """OOD-assumption diagnostics per leave-one-site-out fold.

    Runs tensor program recovery on the training sites of each fold and
    records recovered-eigenvalue condition numbers and top spectra, plus the
    effective rank of the float64 feature covariance. Failing folds skipped.
    """
    n_sites = int(site_ids.max().item()) + 1
    dim = features.shape[1]
    conds, eff_ranks, top_eigs = [], [], []
    for held in range(n_sites):
        tr = site_ids != held
        if tr.sum() < 20:
            continue
        feat, sites = features[tr], site_ids[tr]
        uniq = sites.unique(); smap = {int(s): i for i, s in enumerate(uniq)}
        rem = torch.tensor([smap[int(s)] for s in sites]); nts = len(uniq)
        n_sub = len(feat); ssm = torch.zeros(n_sub, nts)
        for i in range(n_sub):
            ssm[i, int(rem[i])] = 1.0
        mri = MRIData(features=feat, targets=torch.zeros(n_sub),
                      subject_ids=torch.arange(n_sub), site_ids=rem, n_sites=nts,
                      n_subjects=n_sub, feature_names=[f"f{i}" for i in range(dim)],
                      site_names=[str(i) for i in range(nts)], subject_site_matrix=ssm)
        try:
            obs, _ = mri_to_multienv_format(mri)
            tc = TensorConfig(n_programs=N_PROGRAMS, n_envs=nts, dim=dim)
            rec = recover_programs(obs, n_programs=N_PROGRAMS, tensor_config=tc)
            ev = rec.eigenvalues.abs().to(torch.float64).numpy()
            ev = ev[ev > 0]
            if len(ev):
                conds.append(float(ev.max() / ev.min()))
                top_eigs.append([float(x) for x in np.sort(ev)[::-1][:N_PROGRAMS]])
            # whitening condition number from the pooled second moment
            X = feat.to(torch.float64)
            cov = (X.T @ X) / len(X)
            sv = torch.linalg.svdvals(cov).numpy()
            eff_ranks.append(float((sv > 1e-6 * sv.max()).sum()))
        except Exception:
            continue
    return {"recovery_condnum_mean": float(np.mean(conds)) if conds else None,
            "recovery_condnum_max": float(np.max(conds)) if conds else None,
            "feature_effective_rank_mean": float(np.mean(eff_ranks)) if eff_ranks else None,
            "n_folds": len(conds),
            "top_eigenvalue_spectrum_fold0": top_eigs[0] if top_eigs else None}


def run_dataset(name, loader):
    """Collect all seeds for one cohort and run the three analysis blocks."""
    features, targets, site_ids = loader()
    print(f"[{name}] N={features.shape[0]} E={int(site_ids.max())+1}", flush=True)
    data = [collect(features, targets, site_ids, SEED_BASE + s) for s in range(N_SEEDS)]
    strat = block_stratified(data)
    sel = block_selective(data)
    ood = block_ood_diag(features, targets, site_ids, SEED_BASE)
    print(f"  [{name}] AUROC overall {strat['auroc_overall']:.3f} "
          f"impaired {strat['auroc_impaired']} control {strat['auroc_control']}; "
          f"rho excl single-class {strat['rho_excl_singleclass']}; "
          f"recovery cond {ood['recovery_condnum_mean']}", flush=True)
    return {"stratified": strat, "selective_clinical": sel, "ood_diagnostics": ood}


def main():
    """Run both cohorts and write clinical_stratified.json plus the Markdown report."""
    res = {"adni": run_dataset("adni", load_adni), "oasis": run_dataset("oasis", load_oasis_ds)}
    (OUT / "clinical_stratified.json").write_text(json.dumps(res, indent=2))
    L = ["# Clinical / class-stratified / OOD-diagnostic analyses (MedIA revision)", ""]
    for n in ["adni", "oasis"]:
        r = res[n]; s = r["stratified"]; o = r["ood_diagnostics"]
        L += [f"## {n.upper()}", "", "### Class-stratified reliability",
              f"- prevalence {s['prevalence']:.3f}; balanced accuracy {s['balanced_accuracy']:.3f}; "
              f"sensitivity {s['sensitivity']:.3f}; specificity {s['specificity']:.3f}",
              f"- failure-detection AUROC: overall {s['auroc_overall']:.3f}, "
              f"impaired {s['auroc_impaired']}, control {s['auroc_control']}",
              f"- misfit->err rho: overall {s['rho_overall']:.3f}, impaired {s['rho_impaired']:.3f}, "
              f"control {s['rho_control']:.3f}",
              f"- single-class protocols {s['single_class_protocols']}; "
              f"AUROC excl single-class {s['auroc_excl_singleclass']}, "
              f"rho excl single-class {s['rho_excl_singleclass']}; "
              f"rho balanced-protocols-only {s['rho_balanced_only']}", "",
              "### Clinical selective prediction",
              "| coverage | accuracy | sensitivity | specificity | FN(acc) | FP(acc) | frac abstained impaired |",
              "|---|---|---|---|---|---|---|"]
        for x in r["selective_clinical"]:
            def f(v): return f"{v:.3f}" if v is not None and not (isinstance(v, float) and np.isnan(v)) else "--"
            L.append(f"| {int(x['coverage']*100)}% | {f(x['accuracy'])} | {f(x['sensitivity'])} | "
                     f"{f(x['specificity'])} | {f(x['fn_among_accepted'])} | {f(x['fp_among_accepted'])} | "
                     f"{f(x['frac_abstained_impaired'])} |")
        L += ["", "### OOD-assumption diagnostics",
              f"- per-fold tensor-recovery condition number: mean {o['recovery_condnum_mean']}, "
              f"max {o['recovery_condnum_max']} ({o['n_folds']} folds)",
              f"- feature-covariance effective rank (mean): {o['feature_effective_rank_mean']}",
              f"- top-{N_PROGRAMS} recovered eigenvalue spectrum (fold 0): {o['top_eigenvalue_spectrum_fold0']}", ""]
    (OUT / "clinical_stratified.md").write_text("\n".join(L))
    print("saved clinical_stratified.{json,md}", flush=True)


if __name__ == "__main__":
    main()
