"""TASK 3: Headline (v2) re-derivation from the per-scan records.

From reruns/purged_v2_perscan_local.npz, independently compute:
  - per-seed pooled Spearman rho(misfit, err) and its mean/std,
  - per-seed pooled AUROC (sklearn) and its mean/std,
  - mean within-protocol rho over the 40 (protocol x seed) cells + n positive,
  - selective accuracy at 50% coverage (sort by misfit ascending, accept
    first half; n_accept = int(round(0.5*n))),
  - per-seed error rate.
Compare against results/purged_v2_estimators.json,
results/corrected_v2_summaries.json and results/purged_v2_mlrel.json.
Spot-check v1 (pooled rho + AUROC only).
"""

from misfit.paths import resolve_path as _resolve_path
import json
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

COR = _resolve_path("<REPO_ROOT>/misfit/correction_20261018")
out = {"task": 3}


def derive(tag):
    """Per-seed pooled rho, AUROC, error rate, 50%-coverage selective accuracy,
    and per-(protocol, seed) within-cell rhos from one per-scan archive."""
    f = np.load(f"{COR}/reruns/{tag}_perscan_local.npz", allow_pickle=False)
    site = f["site"]
    d = {"rho": [], "auroc": [], "err_rate": [], "sel50": []}
    within = {}  # (protocol, seed) -> rho
    for s in range(5):
        t = f[f"tested_s{s}"].astype(bool)
        mis = f[f"misfit_s{s}"][t]
        err = f[f"err_s{s}"][t].astype(np.float64)
        d["rho"].append(float(spearmanr(mis, err).statistic))
        d["auroc"].append(float(roc_auc_score(err, mis)))
        d["err_rate"].append(float(err.mean()))
        # selective accuracy @ 50% coverage
        order = np.argsort(mis, kind="stable")
        n_acc = int(round(0.5 * len(mis)))
        acc = 1.0 - float(err[order[:n_acc]].mean())
        d["sel50"].append(acc)
        st = site[t]
        for p in range(8):
            m = st == p
            within[(p, s)] = float(spearmanr(mis[m], err[m]).statistic)
    d["within_cells"] = within
    return d


v2 = derive("purged_v2")

# --- package references
est = json.load(open(f"{COR}/results/purged_v2_estimators.json"))
summ = json.load(open(f"{COR}/results/corrected_v2_summaries.json"))
mlrel = json.load(open(f"{COR}/results/purged_v2_mlrel.json"))

ref_rho_per_seed = est["estimators"]["structural_misfit"]["rho_per_seed"]
ref_rho_mean = summ["pooled_rho"]["mean"]
ref_auroc_mean = summ["pooled_auroc"]["mean"]
ref_auroc_mlrel = mlrel["block_b_auroc"]["misfit"]["auroc"]
ref_auroc_per_seed = mlrel["block_b_auroc"]["misfit"]["auroc_per_seed"]
ref_within_mean = summ["within_env_canonical"]["mean_within_env_rho"]
ref_within_npos = summ["within_env_canonical"]["n_positive"]
ref_sel50_mean = summ["selective"]["0.5"]["accuracy"]["mean"]
ref_sel50_per_seed = [x["50"] for x in est["selective_misfit_per_seed"]]
ref_err = est["err_rate_per_seed"]

cells = list(v2["within_cells"].values())
my = {
    "rho_per_seed": v2["rho"],
    "rho_mean": float(np.mean(v2["rho"])),
    "rho_std_ddof0": float(np.std(v2["rho"])),
    "auroc_per_seed": v2["auroc"],
    "auroc_mean": float(np.mean(v2["auroc"])),
    "within_mean_40cells": float(np.mean(cells)),
    "within_n_positive": int(sum(c > 0 for c in cells)),
    "within_n_cells": len(cells),
    "sel50_per_seed": v2["sel50"],
    "sel50_mean": float(np.mean(v2["sel50"])),
    "err_rate_per_seed": v2["err_rate"],
    "err_rate_mean": float(np.mean(v2["err_rate"])),
}
out["v2_my"] = my
out["v2_ref"] = {
    "rho_per_seed": ref_rho_per_seed, "rho_mean": ref_rho_mean,
    "auroc_mean": ref_auroc_mean, "auroc_mlrel": ref_auroc_mlrel,
    "auroc_per_seed": ref_auroc_per_seed,
    "within_mean": ref_within_mean, "within_n_positive": ref_within_npos,
    "sel50_mean": ref_sel50_mean, "sel50_per_seed": ref_sel50_per_seed,
    "err_rate_per_seed": ref_err,
}
diffs = {
    "rho_per_seed": max(abs(a - b) for a, b in zip(my["rho_per_seed"], ref_rho_per_seed)),
    "rho_mean": abs(my["rho_mean"] - ref_rho_mean),
    "auroc_mean_vs_summ": abs(my["auroc_mean"] - ref_auroc_mean),
    "auroc_mean_vs_mlrel": abs(my["auroc_mean"] - ref_auroc_mlrel),
    "auroc_per_seed": max(abs(a - b) for a, b in zip(my["auroc_per_seed"], ref_auroc_per_seed)),
    "within_mean": abs(my["within_mean_40cells"] - ref_within_mean),
    "within_n_positive": abs(my["within_n_positive"] - ref_within_npos),
    "sel50_mean": abs(my["sel50_mean"] - ref_sel50_mean),
    "sel50_per_seed": max(abs(a - b) for a, b in zip(my["sel50_per_seed"], ref_sel50_per_seed)),
    "err_rate": max(abs(a - b) for a, b in zip(my["err_rate_per_seed"], ref_err)),
}
out["v2_abs_diffs"] = diffs
# tolerance: selective per-seed / err-rate values are serialized via float32
# (1e-7); everything else float64 (1e-12)
out["v2_pass"] = bool(
    diffs["rho_per_seed"] < 1e-12 and diffs["rho_mean"] < 1e-12
    and diffs["auroc_mean_vs_summ"] < 1e-12 and diffs["auroc_per_seed"] < 1e-12
    and diffs["within_mean"] < 1e-12 and diffs["within_n_positive"] == 0
    and diffs["sel50_mean"] < 1e-7 and diffs["sel50_per_seed"] < 1e-7
    and diffs["err_rate"] < 1e-7)

# --- v1 spot check (pooled rho + auroc only)
v1 = derive("purged_v1")
est1 = json.load(open(f"{COR}/results/purged_v1_estimators.json"))
mlrel1 = json.load(open(f"{COR}/results/purged_v1_mlrel.json"))
ref1_rho = est1["estimators"]["structural_misfit"]["rho_per_seed"]
ref1_auroc = mlrel1["block_b_auroc"]["misfit"]["auroc_per_seed"]
out["v1_my"] = {"rho_per_seed": v1["rho"], "rho_mean": float(np.mean(v1["rho"])),
                "auroc_per_seed": v1["auroc"],
                "auroc_mean": float(np.mean(v1["auroc"]))}
out["v1_ref"] = {"rho_per_seed": ref1_rho,
                 "auroc_per_seed": ref1_auroc,
                 "auroc_mean": mlrel1["block_b_auroc"]["misfit"]["auroc"]}
out["v1_abs_diffs"] = {
    "rho_per_seed": max(abs(a - b) for a, b in zip(v1["rho"], ref1_rho)),
    "auroc_per_seed": max(abs(a - b) for a, b in zip(v1["auroc"], ref1_auroc)),
}
out["v1_pass"] = bool(out["v1_abs_diffs"]["rho_per_seed"] < 1e-12
                      and out["v1_abs_diffs"]["auroc_per_seed"] < 1e-12)

out["PASS"] = bool(out["v2_pass"] and out["v1_pass"])
print(json.dumps(out, indent=2, default=str))
json.dump(out, open(f"{COR}/verify/out_task3.json", "w"), indent=2, default=str)
