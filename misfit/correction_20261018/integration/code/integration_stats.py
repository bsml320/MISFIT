"""Integration statistics for the manuscript/figure update (2026-10-19).

Derives, from the VERIFIED corrected V2 per-scan arrays
(correction_20261018/reruns/purged_v2_perscan_local.npz) plus the audit linkage
(audit/adni_linkage_local.csv), the per-scan summary statistics the corrected
manuscript and figures need and that the frozen correction package did not
precompute:

  - diagnostic stratification (misfit percentile by class; CN/impaired mean
    ratio; KS statistic), per seed and seed-averaged
  - per-protocol difficulty (mean/std misfit vs class balance)
  - joint class-by-protocol strata table (per-stratum n, expected errors,
    estimable seeds, Spearman rho, AUROC)
  - demographic subgroup failure-detection AUROCs (sex, acquisition-age median
    split) with participant-blocked bootstrap 95% CIs
  - reliability-decile bin-level Spearman
  - the Gaussian-copula heuristic reading at 50% coverage
  - example-case picks (lowest-misfit always-correct / highest-misfit
    always-wrong acquisition across the 5 seeds)

Everything here is a descriptive summary of the already-verified corrected
outputs; no model is refit and no selection depends on the outcome. Statistics
convention matches the correction package (np.std ddof=0; seeds 42-46 as s0-s4;
participant-blocked uncertainty; RandomState(0) for bootstrap draws).

Output: correction_20261018/integration/results/integration_stats.json
"""
from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp, norm, rankdata, spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(_resolve_path("<REPO_ROOT>/misfit"))
COR = ROOT / "correction_20261018"
OUT = COR / "integration" / "results"
OUT.mkdir(parents=True, exist_ok=True)

N_SEEDS = 5


def load_arrays():
    """Load per-seed misfit/error stacks plus aligned labels, protocols, PTIDs, sex, age."""
    z = np.load(COR / "reruns" / "purged_v2_perscan_local.npz", allow_pickle=True)
    mis = np.stack([z[f"misfit_s{s}"] for s in range(N_SEEDS)])        # (S, N)
    err = np.stack([z[f"err_s{s}"].astype(int) for s in range(N_SEEDS)])
    rows = list(csv.DictReader(open(COR / "audit" / "adni_linkage_local.csv")))
    sex = np.array([r["sex"] for r in rows])
    age = np.array([float(r["age"]) for r in rows])
    # linkage CSV rows and npz arrays must be the same scans in the same order
    assert (np.array([r["ptid"] for r in rows]) == z["ptids"]).all()
    return mis, err, z["label"].astype(int), z["site"].astype(int), z["ptids"], sex, age


def stratification(mis, err, label):
    """Mirror run_biology_composite panel (a): seed-averaged misfit percentile."""
    ma = mis.mean(0)
    pca = 100.0 * rankdata(ma) / len(ma)
    ks = ks_2samp(ma[label == 0], ma[label == 1])
    per_seed_ratio, per_seed_D, per_seed_p = [], [], []
    for s in range(N_SEEDS):
        m = mis[s]
        per_seed_ratio.append(float(m[label == 0].mean() / m[label == 1].mean()))
        k = ks_2samp(m[label == 0], m[label == 1])
        per_seed_D.append(float(k.statistic))
        per_seed_p.append(float(k.pvalue))
    return {
        "median_pct_cn": float(np.median(pca[label == 0])),
        "median_pct_imp": float(np.median(pca[label == 1])),
        "mean_misfit_ratio_cn_over_imp": float(ma[label == 0].mean() / ma[label == 1].mean()),
        "ks_D": float(ks.statistic), "ks_p": float(ks.pvalue),
        "per_seed_ratio": per_seed_ratio, "per_seed_ks_D": per_seed_D,
        "per_seed_ks_p": per_seed_p,
        "n_cn": int((label == 0).sum()), "n_imp": int((label == 1).sum()),
    }


def protocol_difficulty(mis, label, site):
    """Per-protocol mean misfit vs class balance (Spearman of |pct_disease - 50| against mean misfit)."""
    ma = mis.mean(0)
    rows = []
    for e in sorted(set(site.tolist())):
        m = site == e
        rows.append({
            "protocol": int(e), "n": int(m.sum()),
            "pct_disease": float(100.0 * label[m].mean()),
            "mean_misfit": float(ma[m].mean()), "std_misfit": float(ma[m].std()),
        })
    bal = np.array([abs(r["pct_disease"] - 50.0) for r in rows])
    mm = np.array([r["mean_misfit"] for r in rows])
    rho, p = spearmanr(bal, mm)
    return {"rows": rows, "spearman_absbal_vs_misfit": float(rho), "p": float(p)}


def strata_table(mis, err, label, site):
    """Class-by-protocol strata table.

    A (stratum, seed) cell is estimable with n >= 8 scans and both error
    outcomes present; the aggregate is an equal-weight mean over estimable
    cells.
    """
    out = []
    for c in (0, 1):
        for e in sorted(set(site.tolist())):
            m = (label == c) & (site == e)
            n = int(m.sum())
            exp_err = float(err[:, m].sum(1).mean()) if n else 0.0
            cells_rho, cells_auc = [], []
            for s in range(N_SEEDS):
                es = err[s, m]
                if n >= 8 and len(np.unique(es)) == 2:
                    cells_rho.append(float(spearmanr(mis[s, m], es).statistic))
                    cells_auc.append(float(roc_auc_score(es, mis[s, m])))
            out.append({
                "class": "CN" if c == 0 else "imp", "env": int(e), "n": n,
                "expected_errors": exp_err, "estimable_seeds": len(cells_rho),
                "rho_mean": float(np.mean(cells_rho)) if cells_rho else None,
                "rho_std": float(np.std(cells_rho)) if cells_rho else None,
                "auroc_mean": float(np.mean(cells_auc)) if cells_auc else None,
                "n_positive": int(sum(r > 0 for r in cells_rho)),
            })
    est = [r for r in out if r["estimable_seeds"] > 0]
    # aggregate over estimable cells, equal weight per cell
    cell_rhos, cell_aucs, pos = [], [], 0
    for c in (0, 1):
        for e in sorted(set(site.tolist())):
            m = (label == c) & (site == e)
            n = int(m.sum())
            for s in range(N_SEEDS):
                es = err[s, m]
                if n >= 8 and len(np.unique(es)) == 2:
                    r_ = float(spearmanr(mis[s, m], es).statistic)
                    cell_rhos.append(r_)
                    cell_aucs.append(float(roc_auc_score(es, mis[s, m])))
                    pos += r_ > 0
    return {
        "rows": out,
        "n_strata_estimable": len(est), "n_strata_total": len(out),
        "n_cells": len(cell_rhos),
        "rho_mean": float(np.mean(cell_rhos)), "rho_std": float(np.std(cell_rhos)),
        "auroc_mean": float(np.mean(cell_aucs)),
        "pct_positive": float(100.0 * pos / len(cell_rhos)),
    }


def subgroup_aurocs(mis, err, ptids, sex, age, n_boot=2000):
    """Per-seed pooled AUROC averaged over seeds (correction-package convention),
    participant-blocked bootstrap percentile CIs (RandomState(0), 2000 draws)."""
    med = float(np.median(age))
    defs = [("All", np.ones(len(sex), bool)),
            ("Female", sex == "F"), ("Male", sex == "M"),
            ("Younger", age < med), ("Older", age >= med)]
    rng = np.random.RandomState(0)
    res = []
    for name, m in defs:
        aucs = [roc_auc_score(err[s, m], mis[s, m]) for s in range(N_SEEDS)]
        point = float(np.mean(aucs))
        pts = np.array(sorted(set(ptids[m].tolist())))
        scans_by_pt = {p: np.where(m & (ptids == p))[0] for p in pts}
        boots = []
        for _ in range(n_boot):
            draw = pts[rng.randint(0, len(pts), len(pts))]
            idx = np.concatenate([scans_by_pt[p] for p in draw])
            vals = []
            for s in range(N_SEEDS):
                es = err[s, idx]
                if len(np.unique(es)) == 2:
                    vals.append(roc_auc_score(es, mis[s, idx]))
            if vals:
                boots.append(np.mean(vals))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        res.append({"group": name, "n_acq": int(m.sum()),
                    "n_participants": int(len(pts)),
                    "auroc": point, "lo": float(lo), "hi": float(hi)})
        print(f"  {name:8s} acq={m.sum():4d} AUROC={point:.3f} [{lo:.3f},{hi:.3f}]",
              flush=True)
    return {"age_median_acquisitions": med, "groups": res}


def copula_heuristic(rho_s, err_rate, coverage):
    """Gaussian-copula accepted-set accuracy heuristic.

    Converts Spearman rho to the Gaussian-copula Pearson correlation via
    2*sin(pi*rho/6), then evaluates the implied accepted-set accuracy when
    accepting the lowest-score coverage fraction.
    """
    rho_p = 2.0 * np.sin(np.pi * rho_s / 6.0)
    p, c = err_rate, coverage
    arg = (norm.ppf(c) - rho_p * norm.ppf(1 - p)) / np.sqrt(1 - rho_p**2)
    return float(1 - (p / c) * norm.cdf(arg))


def examples(mis, err, label, site):
    """Example picks: lowest-misfit always-correct and highest-misfit always-wrong acquisitions across seeds."""
    ma = mis.mean(0)
    always_correct = (err == 0).all(0)
    always_wrong = (err == 1).all(0)
    rel_pool = np.where(always_correct)[0]
    unr_pool = np.where(always_wrong)[0]
    rel = int(rel_pool[ma[rel_pool].argmin()])
    unr = int(unr_pool[ma[unr_pool].argmax()])
    def rec(i):
        return {"index": int(i), "label": int(label[i]), "protocol": int(site[i]),
                "misfit_percentile": float(100.0 * (ma < ma[i]).mean())}
    return {"n_always_correct": int(always_correct.sum()),
            "n_always_wrong": int(always_wrong.sum()),
            "most_reliable": rec(rel), "least_reliable": rec(unr)}


def main():
    """Compute every block and write integration_stats.json."""
    mis, err, label, site, ptids, sex, age = load_arrays()
    v2 = json.load(open(COR / "results" / "corrected_v2_summaries.json"))

    res = {
        "source": "purged_v2_perscan_local.npz (verified corrected V2) + adni_linkage_local.csv",
        "convention": "seeds 42-46; np.std ddof=0; participant-blocked bootstrap RandomState(0)",
        "stratification": stratification(mis, err, label),
        "protocol_difficulty": protocol_difficulty(mis, label, site),
        "strata": strata_table(mis, err, label, site),
        "subgroup": subgroup_aurocs(mis, err, ptids, sex, age),
        "examples": examples(mis, err, label, site),
    }
    bins = v2["reliability_deciles"]["bins"]
    rho, p = spearmanr([b["bin"] for b in bins], [b["err_rate"] for b in bins])
    res["decile_bin_spearman"] = {"rho": float(rho), "p": float(p)}
    err_rate = 1.0 - v2["selective"]["1.0"]["accuracy"]["mean"]
    res["copula_heuristic_50pct"] = {
        "rho_pooled": v2["pooled_rho"]["mean"], "err_rate": err_rate,
        "bound": copula_heuristic(v2["pooled_rho"]["mean"], err_rate, 0.5),
        "observed": v2["selective"]["0.5"]["accuracy"]["mean"],
    }
    res["balanced_frontier"] = {
        "full": float((v2["selective"]["1.0"]["sensitivity"]["mean"]
                       + v2["selective"]["1.0"]["specificity"]["mean"]) / 2),
        "at50": float((v2["selective"]["0.5"]["sensitivity"]["mean"]
                       + v2["selective"]["0.5"]["specificity"]["mean"]) / 2),
    }
    (OUT / "integration_stats.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items()
                      if k in ("stratification", "decile_bin_spearman",
                               "copula_heuristic_50pct", "balanced_frontier",
                               "examples")}, indent=2))
    print("protocol difficulty:", json.dumps(res["protocol_difficulty"]["rows"], indent=1))
    print("strata agg:", {k: v for k, v in res["strata"].items() if k != "rows"})
    print("wrote", OUT / "integration_stats.json")


if __name__ == "__main__":
    main()
