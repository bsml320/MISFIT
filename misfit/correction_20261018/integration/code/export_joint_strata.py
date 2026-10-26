"""Export the corrected ADNI joint protocol-by-class stratum table (V2).

Requested export: per-stratum (8 protocols x 2 diagnostic classes) size,
unique-participant count, per-seed error count, Spearman correlation and
AUROC, with the aggregation and SD conventions stated, plus the aggregate
within-stratum AUROC. Derived entirely from the saved corrected V2 per-scan
arrays (reruns/purged_v2_perscan_local.npz); no new model fits.

Conventions (identical to the frozen correction package, corrected_summaries.py
strata16):
  - stratum = held-out protocol x diagnostic class (CN vs impaired);
  - cell = stratum x seed (5 seeds, 42-46);
  - a cell is estimable when it holds >= 5 tested acquisitions and both error
    outcomes are present (0 < n_errors < n_tested); under V2 every acquisition
    is tested every seed, so n_tested equals the stratum size;
  - within a cell, Spearman rho is computed between per-acquisition misfit and
    binary error, and AUROC treats error as the positive class with misfit as
    the score (acquisition-level point estimates);
  - per-stratum summaries average over that stratum's estimable seeds;
  - the aggregate averages over all estimable cells with equal weight;
  - every SD is np.std with ddof=0, matching the package convention;
  - participant counts are unique-participant tallies only (no identifiers).

Binding gate: the aggregate must reproduce the verified package strata16 block
and the integration aggregate within-stratum AUROC; the export refuses to
write if the externally supplied references disagree.

Writes results/adni_joint_strata_v2.json and results/adni_joint_strata_v2.csv
(one row per stratum x seed cell). Both outputs are DUA-safe aggregates.
"""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "reruns" / "purged_v2_perscan_local.npz"
CACHE = ROOT / "reruns" / "adni_cache_local.npz"
SUMMARIES = ROOT / "results" / "corrected_v2_summaries.json"
INTEGRATION = ROOT / "integration" / "results" / "integration_stats.json"
DST_JSON = ROOT / "results" / "adni_joint_strata_v2.json"
DST_CSV = ROOT / "results" / "adni_joint_strata_v2.csv"
N_SEEDS = 5
SEEDS = [42, 43, 44, 45, 46]


def main() -> None:
    """Build the per-stratum table, enforce the binding gate, write JSON and CSV."""
    z = np.load(SRC, allow_pickle=True)
    site, label, ptids = z["site"], z["label"], z["ptids"]
    site_names = [str(s) for s in np.load(CACHE, allow_pickle=True)["site_names"]]

    strata, cells = [], []
    for e in sorted(set(site.tolist())):
        for cls in (0, 1):
            m = (site == e) & (label == cls)
            n_acq = int(m.sum())
            n_part = int(len(set(ptids[m].tolist())))
            per_seed = []
            for s in range(N_SEEDS):
                tested = z[f"tested_s{s}"] & m
                err = z[f"err_s{s}"][tested]
                mis = z[f"misfit_s{s}"][tested]
                n_tested = int(tested.sum())
                n_err = int(err.sum())
                estimable = n_tested >= 5 and 0 < n_err < n_tested
                rho = auc = None
                if estimable:
                    r = float(spearmanr(mis, err).statistic)
                    if np.isnan(r):
                        estimable = False
                    else:
                        rho = r
                        auc = float(roc_auc_score((err > 0.5).astype(int), mis))
                per_seed.append({"seed": SEEDS[s], "n_tested": n_tested,
                                 "n_errors": n_err, "estimable": estimable,
                                 "rho": rho, "auroc": auc})
                if estimable:
                    cells.append((rho, auc))
            est = [c for c in per_seed if c["estimable"]]
            strata.append({
                "protocol_index": int(e), "protocol": site_names[int(e)],
                "class": "CN" if cls == 0 else "impaired",
                "n_acquisitions": n_acq, "n_participants_unique": n_part,
                "estimable_seeds": len(est),
                "rho_mean": float(np.mean([c["rho"] for c in est])) if est else None,
                "rho_std": float(np.std([c["rho"] for c in est])) if est else None,
                "auroc_mean": float(np.mean([c["auroc"] for c in est])) if est else None,
                "auroc_std": float(np.std([c["auroc"] for c in est])) if est else None,
                "n_seeds_rho_positive": int(sum(c["rho"] > 0 for c in est)),
                "per_seed": per_seed,
            })

    rhos = np.array([r for r, _ in cells])
    aucs = np.array([a for _, a in cells])
    aggregate = {
        "n_strata_total": len(strata),
        "n_strata_estimable": int(sum(s["estimable_seeds"] > 0 for s in strata)),
        "n_cells": len(cells),
        "rho_mean": float(rhos.mean()), "rho_std": float(rhos.std()),
        "auroc_mean": float(aucs.mean()), "auroc_std": float(aucs.std()),
        "pct_cells_rho_positive": float(100.0 * (rhos > 0).mean()),
    }

    # Binding gate against the verified package and integration values.
    pkg = json.load(open(SUMMARIES))["strata16"]
    integ = json.load(open(INTEGRATION))["strata"]
    checks = {
        "binds_pkg_rho_mean": bool(np.isclose(aggregate["rho_mean"], pkg["rho_mean"], atol=1e-12)),
        "binds_pkg_rho_std": bool(np.isclose(aggregate["rho_std"], pkg["rho_std"], atol=1e-12)),
        "binds_pkg_counts": aggregate["n_strata_estimable"] == pkg["n_strata_estimable"]
        and aggregate["n_cells"] == pkg["n_cells"]
        and aggregate["pct_cells_rho_positive"] == pkg["pct_positive"],
        "binds_integration_auroc": bool(np.isclose(aggregate["auroc_mean"],
                                                   integ["auroc_mean"], atol=1e-12)),
    }
    print(json.dumps({"aggregate": aggregate, "binding": checks}, indent=2))
    if not all(checks.values()):
        raise SystemExit("BINDING FAILED: export does not reproduce the verified aggregates")

    out = {
        "source": "corrected V2 (participant-purged + per-fold source-only "
                  "preprocessing), seeds 42-46, reruns/purged_v2_perscan_local.npz",
        "conventions": {
            "stratum": "held-out protocol x diagnostic class (CN vs impaired)",
            "cell": "stratum x seed",
            "estimable_cell": ">= 5 tested acquisitions and both error outcomes present",
            "rho": "Spearman(misfit, binary error) over tested acquisitions in the cell",
            "auroc": "error as positive class, misfit as score, within the cell",
            "per_stratum_summary": "mean/SD over that stratum's estimable seeds",
            "aggregate": "unweighted mean/SD over all estimable cells",
            "sd": "np.std ddof=0 (package convention)",
            "unit": "point estimates acquisition-level; participant counts are "
                    "unique-participant tallies only (no identifiers)",
        },
        "aggregate": aggregate, "binding": checks, "strata": strata,
    }
    DST_JSON.write_text(json.dumps(out, indent=1))
    with open(DST_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["protocol_index", "protocol", "class", "n_acquisitions",
                    "n_participants_unique", "seed", "n_tested", "n_errors",
                    "estimable", "rho", "auroc"])
        for s in strata:
            for c in s["per_seed"]:
                w.writerow([s["protocol_index"], s["protocol"], s["class"],
                            s["n_acquisitions"], s["n_participants_unique"],
                            c["seed"], c["n_tested"], c["n_errors"],
                            c["estimable"], c["rho"], c["auroc"]])
    print(f"wrote {DST_JSON}\nwrote {DST_CSV}")


if __name__ == "__main__":
    main()
