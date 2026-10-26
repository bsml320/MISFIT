"""Addendum A2 S-b: LABELED one-acquisition-per-participant sensitivity (V3).

NOT a replacement for the participant purge. Selection rule (fixed before any
corrected result was inspected): keep each participant's acquisition from the
earliest PHASE by the ordering ADNIGO < ADNI2 < ADNI3 (deterministic
protocol-name tie-break; this is a phase ordering, NOT chronological
selection -- acquisition dates are not available on this server). Environments
are the historical protocol groups restricted to retained acquisitions, with
the historical min-30 eligibility rule re-applied (dropped protocols are
reported). Features: per-fold source-only PCA + z-norm (V2 construction);
the purge is structurally trivial here (one acquisition per participant).

Outputs: results/dedup_v3_estimators.json, results/dedup_v3_dynamics.json,
         reruns/dedup_v3_perscan_local.npz
"""

from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

import purged_loader  # noqa: E402
from purged_harness import collect_dynamics_purged, purge_masks, run_seed_purged  # noqa: E402

sys.path.insert(0, _resolve_path("<REPO_ROOT>"))
sys.path.insert(0, _resolve_path("<REPO_ROOT>/misfit/experiments"))
from misfit.experiments.run_honest_estimator_table import (  # noqa: E402
    N_SEEDS, SEED_BASE,
)
from run_training_dynamics import SCORES  # noqa: E402

COR = CODE.parent
PHASE_ORDER = {"ADNIGO": 0, "ADNI2": 1, "ADNI3": 2}


def main() -> None:
    """Build the dedup cohort, run the 5-seed estimator and dynamics passes, write outputs."""
    d = purged_loader.load()
    raw, targets = d["raw"], d["targets"]
    site_ids, ptids, site_names = d["site_ids"].numpy(), d["ptids"], d["site_names"]

    # outcome-independent phase-order earliest selection
    # (np.lexsort sorts by the LAST key first: phase order, then protocol name)
    order = np.lexsort((np.array([site_names[s] for s in site_ids]),
                        np.array([PHASE_ORDER[site_names[s].split("-")[0]]
                                  for s in site_ids])))
    seen, keep = set(), np.zeros(len(ptids), bool)
    for i in order:
        if ptids[i] not in seen:
            seen.add(ptids[i])
            keep[i] = True
    kept_counts = {site_names[s]: int(((site_ids == s) & keep).sum())
                   for s in range(len(site_names))}
    eligible_sites = [s for s in range(len(site_names))
                      if ((site_ids == s) & keep).sum() >= 30]
    dropped = {site_names[s]: kept_counts[site_names[s]]
               for s in range(len(site_names)) if s not in eligible_sites}
    sel = keep & np.isin(site_ids, eligible_sites)
    idx = np.where(sel)[0]
    remap = {s: i for i, s in enumerate(eligible_sites)}
    sub_sites = torch.tensor([remap[s] for s in site_ids[idx]])
    sub_targets = targets[idx]
    sub_ptids = ptids[idx]
    sub_raw = raw[idx]
    assert len(set(sub_ptids)) == len(sub_ptids), "dedup violated"
    meta = {"n_retained": int(len(idx)),
            "per_protocol_after_dedup": kept_counts,
            "protocols_dropped_lt30": dropped,
            "selection_rule": "earliest phase ADNIGO<ADNI2<ADNI3 "
                              "(phase ordering, NOT chronological)"}
    print(json.dumps(meta, indent=1), flush=True)

    # fold feature bank: purged train pools are seed-independent, so one
    # source-only PCA per held-out protocol serves all seeds
    bank = {}
    for held in range(len(eligible_sites)):
        tr, _ = purge_masks(sub_sites, sub_ptids, held)
        project = purged_loader.fold_pca(sub_raw, np.where(tr.numpy())[0])
        full = torch.empty(len(idx), 50)
        for chunk in np.array_split(np.arange(len(idx)), 8):
            full[chunk] = project(chunk)
        bank[held] = full

    def fold_features(train_mask, test_mask):
        held = int(sub_sites[test_mask][0])
        return bank[held][train_mask], bank[held][test_mask]

    per_seed_scores: dict[str, list[float]] = {}
    extras_all, perscan_all = [], []
    for s_idx in range(N_SEEDS):
        seed = SEED_BASE + s_idx
        t0 = time.time()
        out, extras, perscan = run_seed_purged(
            fold_features, sub_targets, sub_sites, sub_ptids, seed,
            want_selective=True)
        for k, v in out.items():
            per_seed_scores.setdefault(k, []).append(float(v))
        extras_all.append(extras)
        perscan_all.append(perscan)
        print(f"[v3] seed {seed} misfit_rho={out['structural_misfit']:.3f} "
              f"err={extras['err_rate']:.3f} ({time.time()-t0:.0f}s)", flush=True)

    dyn = {k: {"rho": [], "auroc": []} for k in SCORES}
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    for s_idx in range(N_SEEDS):
        scores, err, m, _ = collect_dynamics_purged(
            fold_features, sub_targets, sub_sites, sub_ptids, SEED_BASE + s_idx)
        e = err[m]
        for k in SCORES:
            dyn[k]["rho"].append(float(spearmanr(scores[k][m], e)[0]))
            if 0 < e.sum() < len(e):
                dyn[k]["auroc"].append(float(roc_auc_score(e, scores[k][m])))

    (COR / "results/dedup_v3_estimators.json").write_text(json.dumps({
        "variant": "v3_dedup_sensitivity (LABELED; not a purge replacement)",
        "meta": meta,
        "estimand": "acquisition-level, one acquisition per participant",
        "sd_convention": "np.std ddof=0 over 5 seeds",
        "estimators": {k: {"rho_mean": float(np.mean(v)),
                           "rho_std": float(np.std(v)), "rho_per_seed": v}
                       for k, v in per_seed_scores.items()},
        "err_rate_per_seed": [e["err_rate"] for e in extras_all],
        "selective_misfit_per_seed": [e.get("selective_misfit") for e in extras_all],
    }, indent=2))
    (COR / "results/dedup_v3_dynamics.json").write_text(json.dumps({
        k: {"rho_mean": float(np.mean(v["rho"])), "rho_std": float(np.std(v["rho"])),
            "auroc_mean": float(np.mean(v["auroc"])),
            "auroc_std": float(np.std(v["auroc"]))}
        for k, v in dyn.items()}, indent=2))
    np.savez_compressed(
        COR / "reruns/dedup_v3_perscan_local.npz",
        ptids=sub_ptids, site=sub_sites.numpy(),
        label=sub_targets.numpy().astype(int),
        **{f"{k}_s{j}": perscan_all[j][k]
           for k in ["misfit", "erm_loss", "err", "tested"]
           for j in range(N_SEEDS)})
    print("[v3] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
