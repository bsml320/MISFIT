"""Driver for the participant-purged ADNI reruns (plan section 2).

Usage: CUDA_VISIBLE_DEVICES=2 python3 rerun_adni_purged.py v1
       CUDA_VISIBLE_DEVICES=3 python3 rerun_adni_purged.py v2

V1: historical global-PCA features unchanged; purge only.
V2: per-fold PCA + z-norm fit on the purged training pool (fold feature bank
    precomputed once per held-out protocol; purged pools are seed-independent).

Outputs (aggregate -> results/, participant-level -> reruns/*_local.npz):
  results/purged_<v>_estimators.json   all estimators, per-seed rho + extras
  results/purged_<v>_withinfold.json   per-protocol within-fold table
  results/purged_<v>_mlrel.json        conformal block A + AUROC block B (+C/D)
  results/purged_<v>_dynamics.json     AUM / forgetting / EL2N / final-loss
  reruns/purged_<v>_perscan_local.npz  per-seed per-scan scores (DUA local)
Every stage appends to reruns/execution_log.jsonl. Purge is verified at every
fold (zero shared participants) and the run aborts on violation.
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
from purged_harness import (  # noqa: E402
    collect_dynamics_purged, purge_masks, random_removal_masks,
    run_seed_purged, within_fold_purged,
)

sys.path.insert(0, _resolve_path("<REPO_ROOT>"))
sys.path.insert(0, _resolve_path("<REPO_ROOT>/misfit/experiments"))
from misfit.experiments.run_honest_estimator_table import (  # noqa: E402
    N_SEEDS, SEED_BASE,
)
from run_ml_reliability_media import (  # noqa: E402
    block_a_conformal, block_b_auroc, block_c_attribution, block_d_programs,
)
from run_training_dynamics import SCORES  # noqa: E402

COR = CODE.parent
RES = COR / "results"
RER = COR / "reruns"
LOG = RER / "execution_log.jsonl"


def log(rec: dict) -> None:
    rec["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")


def verify_purge(site_ids, ptids):
    """Hard check: under purge_masks, no fold shares a participant between
    train and test; returns per-fold purged-train sizes."""
    sizes = {}
    for held in range(int(site_ids.max()) + 1):
        tr, te = purge_masks(site_ids, ptids, held)
        shared = set(ptids[tr.numpy()]) & set(ptids[te.numpy()])
        assert not shared, f"purge violated for fold {held}: {len(shared)} shared"
        sizes[held] = {"train": int(tr.sum()), "test": int(te.sum())}
    return sizes


def hist_masks(site_ids, ptids, held_out, seed=None):
    """v0: the historical (contaminated) masks, for reproduction/binding."""
    return site_ids != held_out, site_ids == held_out


def main() -> None:
    """Run the requested variant end to end.

    Stages: GPU guard, purge verification (v1/v2), fold feature construction,
    the 5-seed estimator loop with per-scan archives, ml_reliability blocks on
    the same fits, the within-fold table, and training-dynamics scores.
    """
    import os
    # Select CUDA_VISIBLE_DEVICES externally.
    RES.mkdir(parents=True, exist_ok=True)
    RER.mkdir(parents=True, exist_ok=True)
    variant = sys.argv[1]
    assert variant in {"v0", "v1", "v2", "v1r"}
    d = purged_loader.load()
    raw, feats = d["raw"], d["features"]
    targets, site_ids, ptids = d["targets"], d["site_ids"], d["ptids"]
    site_names = d["site_names"]

    mask_fn = {"v0": hist_masks, "v1": purge_masks, "v2": purge_masks,
               "v1r": random_removal_masks}[variant]
    if variant in {"v1", "v2"}:
        sizes = verify_purge(site_ids, ptids)
        log({"stage": "verify_purge", "variant": variant, "sizes": sizes})
    else:
        log({"stage": f"control_{variant}", "note": "v1r = size-matched random "
             "removal (addendum A8); v0 = historical masks for reproduction"})

    if variant in {"v0", "v1", "v1r"}:
        def fold_features(train_mask, test_mask):
            return feats[train_mask], feats[test_mask]
    else:
        bank = {}
        for held in range(int(site_ids.max()) + 1):
            tr, _ = purge_masks(site_ids, ptids, held)
            idx_tr = np.where(tr.numpy())[0]
            project = purged_loader.fold_pca(raw, idx_tr)
            full = torch.empty(len(ptids), 50)
            all_idx = np.arange(len(ptids))
            for chunk in np.array_split(all_idx, 8):
                full[chunk] = project(chunk)
            bank[held] = full
            log({"stage": "fold_pca", "held_out": held, "n_train": len(idx_tr)})

        def fold_features(train_mask, test_mask):
            held = int(site_ids[test_mask][0])
            return bank[held][train_mask], bank[held][test_mask]

    # ---- full estimator table + per-scan records, 5 seeds ----
    per_seed_scores: dict[str, list[float]] = {}
    extras_all, perscan_all = [], []
    for s_idx in range(N_SEEDS):
        seed = SEED_BASE + s_idx
        t0 = time.time()
        out, extras, perscan = run_seed_purged(
            fold_features, targets, site_ids, ptids, seed,
            want_selective=True, mask_fn=mask_fn)
        for k, v in out.items():
            per_seed_scores.setdefault(k, []).append(float(v))
        extras_all.append(extras)
        perscan_all.append(perscan)
        log({"stage": "run_seed", "variant": variant, "seed": seed,
             "wall_s": round(time.time() - t0, 1),
             "err_rate": extras["err_rate"],
             "misfit_rho": float(out["structural_misfit"])})
        print(f"[{variant}] seed {seed} done "
              f"misfit_rho={out['structural_misfit']:.3f} "
              f"err_rate={extras['err_rate']:.3f}", flush=True)

    # corrected overlap diagnostics: error rate by historical-overlap status
    copies = {}
    for p in ptids:
        copies[p] = copies.get(p, 0) + 1
    in_multi = np.array([copies[p] >= 2 for p in ptids])
    diag = {"per_seed_err_overlap": [], "per_seed_err_single": []}
    for ps in perscan_all:
        e = ps["err"]
        diag["per_seed_err_overlap"].append(float(np.nanmean(e[in_multi])))
        diag["per_seed_err_single"].append(float(np.nanmean(e[~in_multi])))

    est = {
        "variant": variant, "seeds": list(range(SEED_BASE, SEED_BASE + N_SEEDS)),
        "estimand": "acquisition-level (731 acquisitions from 547 participants)",
        "sd_convention": "np.std ddof=0 over 5 seeds",
        "estimators": {k: {"rho_mean": float(np.mean(v)),
                           "rho_std": float(np.std(v)),
                           "rho_per_seed": v}
                       for k, v in per_seed_scores.items()},
        "err_rate_per_seed": [e["err_rate"] for e in extras_all],
        "selective_misfit_per_seed": [e.get("selective_misfit") for e in extras_all],
        "n_tested": extras_all[0]["n_tested"],
        "overlap_diagnostic_after_purge": diag,
        "n_contributing_refits_hist": {
            str(k): int(v) for k, v in zip(
                *np.unique(perscan_all[0]["n_contributing_refits"],
                           return_counts=True))},
    }
    (RES / f"purged_{variant}_estimators.json").write_text(json.dumps(est, indent=2))
    log({"stage": "estimators_saved", "variant": variant})

    np.savez_compressed(
        RER / f"purged_{variant}_perscan_local.npz",
        ptids=ptids, site=site_ids.numpy(), label=targets.numpy().astype(int),
        n_refits=perscan_all[0]["n_contributing_refits"],
        **{f"{k}_s{j}": perscan_all[j][k]
           for k in ["misfit", "erm_loss", "err", "tested", "conformal_q",
                     "max_softmax", "energy", "mc_dropout", "test_ce",
                     "maha_test", "deep_ensemble5"]
           for j in range(N_SEEDS)},
    )

    # ---- ml_reliability blocks on the SAME purged fits ----
    data_by_seed = [
        {k: ps[k] for k in ["tested", "err", "site", "misfit", "erm_loss",
                            "conformal_q", "max_softmax", "energy",
                            "mc_dropout", "attr", "prog"]}
        for ps in perscan_all
    ]
    a = block_a_conformal(data_by_seed)
    b, roc = block_b_auroc(data_by_seed)
    c = block_c_attribution(data_by_seed, None)
    dd = block_d_programs(data_by_seed)
    (RES / f"purged_{variant}_mlrel.json").write_text(json.dumps(
        {"variant": variant, "block_a_conformal": a, "block_b_auroc": b,
         "roc_points": roc, "block_c_attribution": c,
         "block_d_programs": dd}, indent=2))
    log({"stage": "mlrel_saved", "variant": variant,
         "misfit_auroc": b["misfit"]["auroc"]})

    # ---- within-fold per-protocol table ----
    wf = within_fold_purged(fold_features, targets, site_ids, ptids,
                            mask_fn=mask_fn)
    wf["site_names"] = site_names
    (RES / f"purged_{variant}_withinfold.json").write_text(json.dumps(wf, indent=2))
    log({"stage": "withinfold_saved", "variant": variant,
         "n_positive": wf["all"]["n_positive"], "n_total": wf["all"]["n_total"]})

    # ---- training-dynamics scores ----
    dyn_per = {k: {"rho": [], "auroc": []} for k in SCORES}
    dyn_scores_by_seed = []
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    for s_idx in range(N_SEEDS):
        seed = SEED_BASE + s_idx
        scores, err, m, nref = collect_dynamics_purged(
            fold_features, targets, site_ids, ptids, seed, mask_fn=mask_fn)
        e = err[m]
        for k in SCORES:
            sc = scores[k][m]
            dyn_per[k]["rho"].append(float(spearmanr(sc, e)[0]))
            if 0 < e.sum() < len(e):
                dyn_per[k]["auroc"].append(float(roc_auc_score(e, sc)))
        dyn_scores_by_seed.append({"scores": {k: scores[k] for k in SCORES},
                                   "err": err, "tested": m})
        log({"stage": "dynamics_seed", "variant": variant, "seed": seed})
    (RES / f"purged_{variant}_dynamics.json").write_text(json.dumps(
        {"variant": variant,
         **{k: {"rho_mean": float(np.mean(v["rho"])),
                "rho_std": float(np.std(v["rho"])),
                "auroc_mean": float(np.mean(v["auroc"])),
                "auroc_std": float(np.std(v["auroc"])),
                "rho_per_seed": v["rho"], "auroc_per_seed": v["auroc"]}
            for k, v in dyn_per.items()}}, indent=2))
    np.savez_compressed(
        RER / f"purged_{variant}_dynamics_local.npz",
        ptids=ptids,
        **{f"{k}_s{j}": dyn_scores_by_seed[j]["scores"][k]
           for k in SCORES for j in range(N_SEEDS)},
        **{f"err_s{j}": dyn_scores_by_seed[j]["err"] for j in range(N_SEEDS)},
        **{f"tested_s{j}": dyn_scores_by_seed[j]["tested"] for j in range(N_SEEDS)},
    )
    log({"stage": "done", "variant": variant})
    print(f"[{variant}] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
