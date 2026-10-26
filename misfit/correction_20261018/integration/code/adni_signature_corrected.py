"""Corrected ADNI anatomical signature (2026-10-19 manuscript integration).

Recomputes the ADNI gray-matter-vs-misfit signature of
experiments/run_image_signature_media.py (and the covariate-adjusted map of the
clinical panel) with the VERIFIED corrected V2 misfit
(correction_20261018/reruns/purged_v2_perscan_local.npz) in place of the
contaminated historical score. The voxel extraction, mask, estimators
(voxelwise Spearman; within-class map pooled by class size; split-half
stability with RandomState(0); Schaefer-100 parcel summary; cross-cohort
agreement against the unchanged OASIS-1 parcel signature) replicate the
historical script exactly. Descriptive only: no model refit, no selection.

Additions mirroring the clinical panel: rank-partial (age, sex) adjusted map,
its split-half stability, its correlation with the unadjusted map, and the
top-decile sign agreement across the five per-seed adjusted maps.

Outputs (integration/results/):
  adni_misfit_signature_corrected.nii.gz
  adni_image_signature_corrected.json
Historical results/ files are NOT touched.
"""
from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

import csv
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, _resolve_path("<REPO_ROOT>"))
from misfit.experiments.run_image_signature_media import (  # noqa: E402
    adni_raw_voxels,
)

ROOT = Path(_resolve_path("<REPO_ROOT>/misfit"))
COR = ROOT / "correction_20261018"
OUT = COR / "integration" / "results"
OUT.mkdir(parents=True, exist_ok=True)
N_SEEDS = 5
CHUNK = 40000


def norm_ranks(v):
    r = rankdata(v).astype(np.float64)
    return (r - r.mean()) / (r.std() + 1e-12)


def chunked_maps(X, y_list, Z=None):
    """Per-voxel Spearman of X (N,V) against each y in y_list; if Z (N,k) is
    given, rank-partial: residualize the ranks of both sides on Z first."""
    N, V = X.shape
    ys = [norm_ranks(y) for y in y_list]
    if Z is not None:
        P = Z @ np.linalg.pinv(Z)
        ys = [y - P @ y for y in ys]
        ys = [y / (y.std() + 1e-12) for y in ys]
    outs = [np.empty(V, np.float32) for _ in ys]
    for a in range(0, V, CHUNK):
        b = min(a + CHUNK, V)
        R = rankdata(X[:, a:b], axis=0).astype(np.float64)
        R -= R.mean(0)
        R /= R.std(0) + 1e-12
        if Z is not None:
            R -= P @ R
            s = R.std(0) + 1e-12
            R /= s
        for o, y in zip(outs, ys):
            o[a:b] = (R.T @ y / N).astype(np.float32)
    return outs


def main():
    """Compute the corrected voxelwise maps (overall, within-class, split-half,
    age/sex-adjusted), the parcel summary and cross-cohort agreement; write the
    NIfTI map and the JSON summary."""
    z = np.load(COR / "reruns" / "purged_v2_perscan_local.npz", allow_pickle=True)
    mis_seed = np.stack([z[f"misfit_s{s}"] for s in range(N_SEEDS)])
    misfit = mis_seed.mean(0)
    rows = list(csv.DictReader(open(COR / "audit" / "adni_linkage_local.csv")))
    sex = np.array([1.0 if r["sex"] == "M" else 0.0 for r in rows])
    age = np.array([float(r["age"]) for r in rows])

    print("loading raw voxels ...", flush=True)
    Vx, labs, mask, affine = adni_raw_voxels()
    assert Vx.shape[0] == len(misfit) == 731
    assert np.array_equal(labs, z["label"].astype(int)), "label alignment broken"
    print(f"aligned: N={Vx.shape[0]}, voxels={Vx.shape[1]}", flush=True)

    # unadjusted overall map + per-seed unadjusted (not needed) ---------------
    (overall,) = chunked_maps(Vx, [misfit])
    print("overall map done", flush=True)

    # within-class map (pooled by class size), replicating within_class_map --
    wc = np.zeros(Vx.shape[1], np.float64)
    n = 0
    for c in (0, 1):
        m = labs == c
        (mc,) = chunked_maps(Vx[m], [misfit[m]])
        wc += m.sum() * mc.astype(np.float64)
        n += m.sum()
    within = (wc / n).astype(np.float32)
    print("within-class map done", flush=True)

    # split-half stability (RandomState(0) permutation, as historical) -------
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(misfit))
    h = len(misfit) // 2
    (a_map,) = chunked_maps(Vx[idx[:h]], [misfit[idx[:h]]])
    (b_map,) = chunked_maps(Vx[idx[h:]], [misfit[idx[h:]]])
    stab = float(np.corrcoef(a_map, b_map)[0, 1])
    print(f"split-half stability = {stab:.3f}", flush=True)

    # adjusted (age, sex) rank-partial maps ----------------------------------
    Z = np.column_stack([np.ones(len(misfit)), norm_ranks(age), norm_ranks(sex)])
    adj_list = chunked_maps(Vx, [misfit] + [mis_seed[s] for s in range(N_SEEDS)], Z=Z)
    adj_mean, adj_seeds = adj_list[0], adj_list[1:]
    (adj_a,) = chunked_maps(Vx[idx[:h]], [misfit[idx[:h]]], Z=Z[idx[:h]])
    (adj_b,) = chunked_maps(Vx[idx[h:]], [misfit[idx[h:]]], Z=Z[idx[h:]])
    adj_stab = float(np.corrcoef(adj_a, adj_b)[0, 1])
    top = np.argsort(-np.abs(adj_mean))[: len(adj_mean) // 10]
    sign_agree = float(np.mean([
        (np.sign(s_[top]) == np.sign(adj_mean[top])).mean() for s_ in adj_seeds]))
    all_seed_agree = float(np.mean(np.all(
        [np.sign(s_[top]) == np.sign(adj_mean[top]) for s_ in adj_seeds], axis=0)))
    print(f"adjusted: {100*(adj_mean<0).mean():.1f}% neg, split-half {adj_stab:.3f}, "
          f"corr w/ unadjusted {np.corrcoef(adj_mean, overall)[0,1]:.3f}", flush=True)

    # save map + parcel summary ----------------------------------------------
    vol = np.zeros(mask.shape, np.float32)
    vol[mask] = overall
    img = nib.Nifti1Image(vol, affine)
    nib.save(img, OUT / "adni_misfit_signature_corrected.nii.gz")

    from misfit.experiments.run_image_signature_media import adni_parcel_summary
    parcel = adni_parcel_summary(img)
    sig_hist = json.load(open(ROOT / "results" / "image_signature.json"))
    o = np.array(sig_hist["oasis"]["roi_signature"], float)
    a = np.array(parcel, float)
    ok = ~np.isnan(a) & ~np.isnan(o)
    rho_cc, p_cc = spearmanr(a[ok], o[ok])

    res = {
        "source": "corrected V2 misfit (purged_v2_perscan_local.npz), seeds 42-46 mean",
        "n": 731, "n_voxels": int(Vx.shape[1]),
        "pct_voxels_negative": float(100.0 * (overall < 0).mean()),
        "stability_split_half": stab,
        "overall_within_agreement": float(np.corrcoef(overall, within)[0, 1]),
        "pct_within_negative": float(100.0 * (within < 0).mean()),
        "parcel_signature": parcel,
        "n_parcels_negative": int((a[ok] < 0).sum()), "n_parcels": int(ok.sum()),
        "cross_cohort_schaefer100_spearman": float(rho_cc), "cross_cohort_p": float(p_cc),
        "adjusted_age_sex": {
            "pct_voxels_negative": float(100.0 * (adj_mean < 0).mean()),
            "stability_split_half": adj_stab,
            "corr_with_unadjusted": float(np.corrcoef(adj_mean, overall)[0, 1]),
            "top_decile_sign_agreement_mean": sign_agree,
            "top_decile_all5_sign_agreement": all_seed_agree,
        },
    }
    (OUT / "adni_image_signature_corrected.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "parcel_signature"}, indent=2))
    print("wrote corrected signature map + json", flush=True)


if __name__ == "__main__":
    main()
