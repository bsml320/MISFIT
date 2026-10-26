"""Re-run the clinical / class-stratified / OOD-diagnostic analyses with PER-SEED
aggregation so every reported number carries a 5-seed mean and standard deviation,
replacing the earlier seed-pooled point estimates (which gave no error bars and
mixed subjects across seeds in a single misfit ranking). Same validated IOP
collector: leave-one-environment-out, deployable training-side misfit, binary
per-subject error target. GPU 2. Output: misfit/results/clinical_stratified.{json,md}.
"""
from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

import json
import os
import sys
import warnings
from pathlib import Path

# CUDA visibility is selected by the caller.
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, roc_auc_score  # noqa: E402

sys.path.insert(0, _resolve_path("<REPO_ROOT>"))
from misfit.experiments.run_honest_estimator_table import (  # noqa: E402
    N_PROGRAMS, N_SEEDS, SEED_BASE, _rho, load_adni, load_oasis_ds,
)
from misfit.experiments.run_clinical_stratified_media import (  # noqa: E402
    COVERAGES, _sens_spec, collect,
)
from misfit.tensor_spectral.recovery import recover_programs  # noqa: E402
from misfit.core.types import TensorConfig  # noqa: E402
from misfit.core.utils import seed_everything  # noqa: E402
from misfit.data.mri_multisite import MRIData, mri_to_multienv_format  # noqa: E402

warnings.filterwarnings("ignore")
RES = Path(_resolve_path("<REPO_ROOT>/misfit/results"))


def _agg(vals):
    """Mean/std across seeds, dropping None/nan entries."""
    a = np.array([v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))], float)
    if len(a) == 0:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(a.mean()), "std": float(a.std()), "n": int(len(a))}


def stratified_perseed(data):
    """Class-stratified reliability metrics aggregated per seed.

    Args:
        data: List of per-seed collect() dicts.

    Returns:
        Dict of mean/std aggregates (AUROC and rho overall and per class,
        balanced accuracy, sensitivity, specificity), plus the variants that
        exclude single-class or keep only label-balanced protocols, and the
        prevalence bookkeeping used to define those subsets.
    """
    def auroc(mis, err, m):
        e = err[m]
        return float(roc_auc_score(e, mis[m])) if 0 < e.sum() < len(e) else None

    keys = ["auroc_overall", "auroc_impaired", "auroc_control", "rho_overall",
            "rho_impaired", "rho_control", "balanced_accuracy", "sensitivity",
            "specificity", "auroc_excl_singleclass", "rho_excl_singleclass", "rho_balanced_only"]
    per = {k: [] for k in keys}
    prev = single = balanced = None
    for d in data:
        t = d["tested"]
        mis, err, true, pred, site = d["misfit"][t], d["err"][t], d["true"][t], d["pred"][t], d["site"][t]
        per["auroc_overall"].append(auroc(mis, err, np.ones(len(mis), bool)))
        per["auroc_impaired"].append(auroc(mis, err, true == 1))
        per["auroc_control"].append(auroc(mis, err, true == 0))
        per["rho_overall"].append(float(_rho(mis, err)))
        per["rho_impaired"].append(float(_rho(mis[true == 1], err[true == 1])))
        per["rho_control"].append(float(_rho(mis[true == 0], err[true == 0])))
        per["balanced_accuracy"].append(float(balanced_accuracy_score(true, pred)))
        s, sp = _sens_spec(true, pred, np.ones(len(true), bool))
        per["sensitivity"].append(s)
        per["specificity"].append(sp)
        prev = {int(e_): float(true[site == e_].mean()) for e_ in np.unique(site)}
        single = [e_ for e_, p in prev.items() if p in (0.0, 1.0) or p < 0.05 or p > 0.95]
        balanced = [e_ for e_, p in prev.items() if 0.35 <= p <= 0.65]
        keep = ~np.isin(site, single)
        bal = np.isin(site, balanced)
        per["auroc_excl_singleclass"].append(auroc(mis, err, keep) if keep.sum() else None)
        per["rho_excl_singleclass"].append(float(_rho(mis[keep], err[keep])) if keep.sum() else None)
        per["rho_balanced_only"].append(float(_rho(mis[bal], err[bal])) if bal.sum() > 5 else None)
    out = {k: _agg(v) for k, v in per.items()}
    t0 = data[0]["tested"]
    out["prevalence"] = float(data[0]["true"][t0].mean())
    out["protocol_prevalence"] = {str(k): v for k, v in prev.items()}
    out["single_class_protocols"] = [int(x) for x in single]
    out["balanced_protocols"] = [int(x) for x in balanced]
    return out


def selective_perseed(data):
    """Clinical selective-prediction metrics per coverage, aggregated per seed.

    Subjects are accepted in ascending misfit order; each COVERAGES level
    yields accuracy, sensitivity, specificity, false negatives and false
    positives among accepted subjects, and the impaired fraction of the
    abstained set.
    """
    metrics = ["accuracy", "sensitivity", "specificity", "fn_among_accepted",
               "fp_among_accepted", "frac_abstained_impaired"]
    per = {cov: {mk: [] for mk in metrics} for cov in COVERAGES}
    for d in data:
        t = d["tested"]
        mis, err, true, pred = d["misfit"][t], d["err"][t], d["true"][t], d["pred"][t]
        order = np.argsort(mis)
        for cov in COVERAGES:
            n = max(1, int(round(cov * len(mis))))
            m = np.zeros(len(mis), bool)
            m[order[:n]] = True
            sens, spec = _sens_spec(true, pred, m)
            e = err[m]
            vals = {
                "accuracy": float(1 - e.mean()), "sensitivity": sens, "specificity": spec,
                "fn_among_accepted": float(((pred[m] == 0) & (true[m] == 1)).mean()),
                "fp_among_accepted": float(((pred[m] == 1) & (true[m] == 0)).mean()),
                "frac_abstained_impaired": float((true[~m] == 1).mean()) if (~m).sum() else float("nan"),
            }
            for mk in metrics:
                per[cov][mk].append(vals[mk])
    return [dict(coverage=cov, **{mk: _agg(per[cov][mk]) for mk in metrics}) for cov in COVERAGES]


def ood_perseed(features, targets, site_ids, seeds):
    """Spectral OOD diagnostics per leave-one-site-out fold and seed.

    For each fold, runs tensor program recovery on the training sites and
    records the recovery eigenvalue condition number and top spectrum, plus
    the effective rank of the float64 feature covariance. Folds where
    recovery raises are skipped.
    """
    n_sites = int(site_ids.max().item()) + 1
    dim = features.shape[1]
    conds, eff_ranks, spectra = [], [], []
    for seed in seeds:
        seed_everything(seed)
        for held in range(n_sites):
            tr = site_ids != held
            if tr.sum() < 20:
                continue
            feat, sites = features[tr], site_ids[tr]
            uniq = sites.unique()
            smap = {int(s): i for i, s in enumerate(uniq)}
            rem = torch.tensor([smap[int(s)] for s in sites])
            nts = len(uniq)
            n_sub = len(feat)
            ssm = torch.zeros(n_sub, nts)
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
                    spectra.append(np.sort(ev)[::-1][:N_PROGRAMS])
                X = feat.to(torch.float64)
                cov = (X.T @ X) / len(X)
                sv = torch.linalg.svdvals(cov).numpy()
                eff_ranks.append(float((sv > 1e-6 * sv.max()).sum()))
            except Exception:
                continue
    spec_arr = np.array([s for s in spectra if len(s) == N_PROGRAMS]) if spectra else np.zeros((0, N_PROGRAMS))
    return {
        "recovery_condnum": _agg(conds),
        "recovery_condnum_max": float(np.max(conds)) if conds else None,
        "feature_effective_rank": _agg(eff_ranks),
        "n_folds_x_seeds": len(conds),
        "top_eigenvalue_spectrum_mean": [float(x) for x in spec_arr.mean(0)] if len(spec_arr) else None,
        "top_eigenvalue_spectrum_std": [float(x) for x in spec_arr.std(0)] if len(spec_arr) else None,
    }


def run_dataset(name, loader):
    """Collect all seeds for one cohort and run the three analyses."""
    features, targets, site_ids = loader()
    seeds = [SEED_BASE + s for s in range(N_SEEDS)]
    print(f"[{name}] N={features.shape[0]} dim={features.shape[1]} E={int(site_ids.max())+1}, {N_SEEDS} seeds",
          flush=True)
    data = [collect(features, targets, site_ids, sd) for sd in seeds]
    strat = stratified_perseed(data)
    sel = selective_perseed(data)
    ood = ood_perseed(features, targets, site_ids, seeds)
    a, c = strat["auroc_overall"], ood["recovery_condnum"]
    print(f"  [{name}] AUROC overall {a['mean']:.3f}+/-{a['std']:.3f}; "
          f"recovery cond {c['mean']:.0f}+/-{c['std']:.0f} (max {ood['recovery_condnum_max']:.0f})", flush=True)
    return {"stratified": strat, "selective_clinical": sel, "ood_diagnostics": ood}


def fmt(d):
    return "--" if d is None or d.get("mean") is None else f"{d['mean']:.3f}$\\pm${d['std']:.3f}"


def main():
    """Run both cohorts and write clinical_stratified.json plus the Markdown table."""
    res = {"adni": run_dataset("adni", load_adni), "oasis": run_dataset("oasis", load_oasis_ds)}
    res["_meta"] = {"n_seeds": N_SEEDS, "seed_base": SEED_BASE,
                    "aggregation": "per-seed mean+/-std (replaces earlier seed-pooled point estimates)"}
    (RES / "clinical_stratified.json").write_text(json.dumps(res, indent=2))
    L = ["# Clinical / class-stratified / OOD diagnostics (per-seed mean$\\pm$std, N=5 seeds)", ""]
    for n in ["adni", "oasis"]:
        s = res[n]["stratified"]
        o = res[n]["ood_diagnostics"]
        L += [f"## {n.upper()}", "",
              f"- AUROC overall {fmt(s['auroc_overall'])}, impaired {fmt(s['auroc_impaired'])}, "
              f"control {fmt(s['auroc_control'])}",
              f"- within-class rho: overall {fmt(s['rho_overall'])}, impaired {fmt(s['rho_impaired'])}, "
              f"control {fmt(s['rho_control'])}",
              f"- recovery condition number {fmt(o['recovery_condnum'])} (max {o['recovery_condnum_max']:.0f}), "
              f"effective rank {fmt(o['feature_effective_rank'])} over {o['n_folds_x_seeds']} fold$\\times$seed", "",
              "| coverage | accuracy | sensitivity | specificity | FN(acc) | frac abstained impaired |",
              "|---|---|---|---|---|---|"]
        for x in res[n]["selective_clinical"]:
            L.append(f"| {int(x['coverage']*100)}% | {fmt(x['accuracy'])} | {fmt(x['sensitivity'])} | "
                     f"{fmt(x['specificity'])} | {fmt(x['fn_among_accepted'])} | {fmt(x['frac_abstained_impaired'])} |")
        L.append("")
    (RES / "clinical_stratified.md").write_text("\n".join(L))
    print("saved per-seed clinical_stratified.{json,md}", flush=True)


if __name__ == "__main__":
    main()
