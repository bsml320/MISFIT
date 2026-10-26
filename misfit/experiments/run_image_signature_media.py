"""Image-space results for the MIA paper: the ANATOMICAL SIGNATURE of the
structural-misfit reliability score.

Honest framing: the misfit is the per-subject training cross-entropy (the
operator is near-inert), so this asks a data-level question: which brain regions'
gray matter is associated with the model being unreliable on a subject. RAW
gray-matter density (re-extracted, unnormalized) is correlated against the
seed-averaged per-subject misfit, voxelwise on ADNI (VBM in MNI space) and per
Schaefer-100 ROI on OASIS (the model's own feature parcels).

To show the map is not merely the CN-vs-impaired contrast, both the OVERALL
signature and the WITHIN-CLASS signature (computed within each label, then pooled)
are reported. A split-half stability coefficient gates whether the map is
reportable, and both cohorts are summarized at the Schaefer-100 parcel level so
the ADNI voxel map and the OASIS ROI map can be compared on a common atlas.

Outputs (misfit):
  paper/figures/adni_misfit_signature.pdf, oasis_misfit_signature.pdf
  results/adni_misfit_signature.nii.gz, oasis_misfit_signature.nii.gz
  results/image_signature.json   (stabilities, cross-cohort agreement, top regions)

GPU 2.
"""
from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

import json
import os
import sys
from pathlib import Path

# CUDA visibility is selected by the caller.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

sys.path.insert(0, _resolve_path("<REPO_ROOT>"))
from misfit.experiments.run_clinical_stratified_media import collect  # noqa: E402
from misfit.experiments.run_honest_estimator_table import (  # noqa: E402
    N_SEEDS,
    SEED_BASE,
    load_adni,
    load_oasis_ds,
)

OUT_FIG = Path(_resolve_path("<REPO_ROOT>/misfit/paper/figures"))
OUT_RES = Path(_resolve_path("<REPO_ROOT>/misfit/results"))
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_RES.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})


# ----------------------------------------------------------------------------- misfit
def seed_avg_misfit(features, targets, site_ids):
    """Per-subject misfit averaged over seeds (aligned to loader order)."""
    per_seed = []
    last = None
    for s in range(N_SEEDS):
        d = collect(features, targets, site_ids, SEED_BASE + s)
        per_seed.append(d["misfit"])
        last = d
    M = np.array(per_seed)  # (S, N)
    return M.mean(0), M, last


# ----------------------------------------------------------------------------- spearman
def _z_rank_cols(X):
    """Column-wise ranks, centered and scaled to unit variance."""
    R = rankdata(X, axis=0).astype(np.float64)
    R -= R.mean(0)
    R /= R.std(0) + 1e-12
    return R


def spearman_map(X, y):
    """Per-column Spearman corr of (N,V) gray-matter X against (N,) misfit y."""
    ry = rankdata(y).astype(np.float64)
    ry = (ry - ry.mean()) / (ry.std() + 1e-12)
    Rx = _z_rank_cols(X)
    return (Rx.T @ ry / len(y)).astype(np.float32)


def within_class_map(X, y, labels):
    """Spearman map computed within each class, pooled by class size."""
    out = np.zeros(X.shape[1], np.float64)
    n = 0
    for c in np.unique(labels):
        m = labels == c
        if m.sum() < 10:
            continue
        out += m.sum() * spearman_map(X[m], y[m]).astype(np.float64)
        n += m.sum()
    return (out / max(n, 1)).astype(np.float32)


def split_half_stability(X, y, seed=0):
    """Pearson correlation between Spearman maps from two random subject halves."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    h = len(y) // 2
    a = spearman_map(X[idx[:h]], y[idx[:h]])
    b = spearman_map(X[idx[h:]], y[idx[h:]])
    return float(np.corrcoef(a, b)[0, 1])


# ----------------------------------------------------------------------------- ADNI
def adni_raw_voxels():
    """Replicate the load_adni_smri iteration to recover raw masked GM voxels,
    the mask, and the affine, in the SAME subject order as the model features."""
    import pandas as pd
    from misfit.data.adni_loader import (
        IMPAIRED_GROUPS,
        LABELS_CSV,
        MIN_SUBJECTS_PER_SITE,
        NII_DIR,
        PROTOCOLS,
        _create_brain_mask,
    )

    labels_df = pd.read_csv(LABELS_CSV)
    label_lookup: dict[tuple, str] = {}
    for _, row in labels_df.iterrows():
        key = (str(row["dataset_id"]).strip(), str(row["file_name"]).strip())
        if key not in label_lookup:
            label_lookup[key] = str(row["Group"]).strip()

    mask = _create_brain_mask(NII_DIR, PROTOCOLS)
    affine = None
    rows, labs = [], []
    for protocol in PROTOCOLS:
        proto_dir = NII_DIR / protocol
        if not proto_dir.exists():
            continue
        proto_rows, proto_labs = [], []
        for subj_dir in sorted(proto_dir.iterdir()):
            smwc1 = subj_dir / "smwc1pT1.nii"
            if not smwc1.exists():
                continue
            key = (protocol, subj_dir.name)
            if key not in label_lookup:
                continue
            group = label_lookup[key]
            if group not in {"CN"} | IMPAIRED_GROUPS:
                continue
            img = nib.load(smwc1)
            if affine is None:
                affine = img.affine
            data = img.get_fdata(dtype=np.float32)
            proto_rows.append(data[mask])
            proto_labs.append(0 if group == "CN" else 1)
        if len(proto_rows) < MIN_SUBJECTS_PER_SITE:
            continue
        rows.extend(proto_rows)
        labs.extend(proto_labs)
    return np.asarray(rows, np.float32), np.asarray(labs), mask, affine


def run_adni():
    """ADNI voxelwise signature: maps, stability, figure, parcel summary."""
    print("ADNI: model features + misfit ...", flush=True)
    feats, targets, site = load_adni()
    misfit, M, last = seed_avg_misfit(feats, targets, site)
    print("ADNI: raw voxels ...", flush=True)
    Vx, labs, mask, affine = adni_raw_voxels()
    assert Vx.shape[0] == len(targets), f"voxel/model N mismatch {Vx.shape[0]} vs {len(targets)}"
    assert np.array_equal(labs, targets.numpy().astype(int)), "label order mismatch (alignment broken)"
    print(f"ADNI aligned: N={Vx.shape[0]}, voxels={Vx.shape[1]}", flush=True)

    overall = spearman_map(Vx, misfit)
    within = within_class_map(Vx, misfit, labs)
    stab = split_half_stability(Vx, misfit)
    print(f"ADNI stability(split-half)={stab:.3f}", flush=True)

    def to_img(vec):
        vol = np.zeros(mask.shape, np.float32)
        vol[mask] = vec
        return nib.Nifti1Image(vol, affine)

    img_overall = to_img(overall)
    img_within = to_img(within)
    nib.save(img_overall, OUT_RES / "adni_misfit_signature.nii.gz")

    # figure: glass brain, overall + within-class
    from nilearn import plotting

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.2))
    vmax = float(np.percentile(np.abs(overall), 99))
    plotting.plot_glass_brain(
        img_overall, axes=axes[0], colorbar=True, plot_abs=False,
        vmax=vmax, cmap="cold_hot",
        title="ADNI: corr(gray matter, misfit), all subjects")
    plotting.plot_glass_brain(
        img_within, axes=axes[1], colorbar=True, plot_abs=False,
        vmax=vmax, cmap="cold_hot",
        title="ADNI: within-class (controls for CN vs impaired)")
    fig.savefig(OUT_FIG / "adni_misfit_signature.pdf", bbox_inches="tight")
    plt.close(fig)

    # parcel-level summary on Schaefer-100 for cross-cohort comparison
    parcel = adni_parcel_summary(img_overall)
    return {
        "n": int(Vx.shape[0]), "n_voxels": int(Vx.shape[1]),
        "stability_split_half": stab,
        "overall_within_agreement": float(np.corrcoef(overall, within)[0, 1]),
        "parcel_signature": parcel,
    }


def adni_parcel_summary(img_overall):
    """Average the ADNI voxel signature within Schaefer-100 parcels."""
    from nilearn.datasets import fetch_atlas_schaefer_2018
    from nilearn.image import resample_to_img

    atlas = fetch_atlas_schaefer_2018(n_rois=100)
    lab_img = resample_to_img(atlas.maps, img_overall, interpolation="nearest",
                              force_resample=True, copy_header=True)
    labvol = np.asarray(lab_img.dataobj).astype(int)
    sig = np.asarray(img_overall.dataobj)
    out = np.full(100, np.nan)
    for j in range(1, 101):
        m = labvol == j
        if m.sum() > 0:
            out[j - 1] = float(sig[m].mean())
    return out.tolist()


# ----------------------------------------------------------------------------- OASIS
def oasis_raw_rois():
    """Re-extract raw (unnormalized) Schaefer-100 ROI GM for OASIS in the same
    subject order/filtering as load_oasis (CDR-valid, age-quartile envs kept)."""
    from nilearn.datasets import fetch_oasis_vbm

    from misfit.data.oasis_loader import _extract_vbm_features

    oasis = fetch_oasis_vbm()
    gm_maps = np.asarray(oasis.gray_matter_maps)
    cdr = oasis.ext_vars["cdr"].values.astype(float)
    age = oasis.ext_vars["age"].values.astype(float)
    valid = ~np.isnan(cdr)

    raw = _extract_vbm_features(list(gm_maps), 100)  # (n_total, 100), raw GM
    raw = raw[valid]
    gm_valid = gm_maps[valid]
    age_v = age[valid]
    # replicate age-quartile env + keep filter (n_envs=4, min 10/env)
    q = np.quantile(age_v, np.linspace(0, 1, 5))
    env = np.digitize(age_v, q[1:-1])
    counts = {e: int((env == e).sum()) for e in sorted(set(env))}
    keep_envs = [e for e, c in counts.items() if c >= 10]
    keep = np.isin(env, keep_envs)
    return raw[keep], gm_valid[keep], age_v[keep]


def run_oasis():
    """OASIS-1 ROI signature: maps, stability, figure, top-region lists."""
    print("OASIS: model features + misfit ...", flush=True)
    feats, targets, site = load_oasis_ds()
    misfit, M, last = seed_avg_misfit(feats, targets, site)
    print("OASIS: raw ROIs ...", flush=True)
    raw, gm_valid, age_v = oasis_raw_rois()
    assert raw.shape[0] == len(targets), f"ROI/model N mismatch {raw.shape[0]} vs {len(targets)}"
    print(f"OASIS aligned: N={raw.shape[0]}, ROIs={raw.shape[1]}", flush=True)
    labs = targets.numpy().astype(int)

    overall = spearman_map(raw, misfit)
    within = within_class_map(raw, misfit, labs)
    stab = split_half_stability(raw, misfit)
    print(f"OASIS stability(split-half)={stab:.3f}", flush=True)

    # map ROI corr onto Schaefer-100 atlas
    from nilearn.datasets import fetch_atlas_schaefer_2018
    from nilearn import plotting

    atlas = fetch_atlas_schaefer_2018(n_rois=100)
    lab_img = nib.load(atlas.maps)
    labvol = np.asarray(lab_img.dataobj).astype(int)
    statvol = np.zeros(labvol.shape, np.float32)
    for j in range(1, 101):
        statvol[labvol == j] = overall[j - 1]
    stat_img = nib.Nifti1Image(statvol, lab_img.affine)
    nib.save(stat_img, OUT_RES / "oasis_misfit_signature.nii.gz")

    fig = plt.figure(figsize=(7.2, 3.0))
    vmax = float(np.percentile(np.abs(overall), 95))
    plotting.plot_stat_map(
        stat_img, figure=fig, colorbar=True, display_mode="z",
        cut_coords=6, vmax=vmax, cmap="cold_hot",
        title="OASIS-1: corr(Schaefer-100 gray matter, misfit)")
    fig.savefig(OUT_FIG / "oasis_misfit_signature.pdf", bbox_inches="tight")
    plt.close(fig)

    labels = [s.replace("7Networks_", "") for s in atlas.labels[1:]]
    order = np.argsort(overall)
    top_neg = [(labels[i], float(overall[i])) for i in order[:8]]
    top_pos = [(labels[i], float(overall[i])) for i in order[::-1][:8]]
    return {
        "n": int(raw.shape[0]), "n_rois": 100,
        "stability_split_half": stab,
        "overall_within_agreement": float(np.corrcoef(overall, within)[0, 1]),
        "roi_signature": overall.tolist(),
        "top_negative_rois": top_neg,
        "top_positive_rois": top_pos,
        "labels": labels,
    }


# ----------------------------------------------------------------------------- main
def main():
    """Run both cohorts, compute cross-cohort parcel agreement, write the JSON."""
    res = {}
    res["adni"] = run_adni()
    res["oasis"] = run_oasis()

    # cross-cohort agreement on the common Schaefer-100 parcellation
    a = np.array(res["adni"]["parcel_signature"], float)
    o = np.array(res["oasis"]["roi_signature"], float)
    ok = ~np.isnan(a) & ~np.isnan(o)
    from scipy.stats import spearmanr

    rho, p = spearmanr(a[ok], o[ok])
    res["cross_cohort"] = {
        "n_parcels": int(ok.sum()),
        "schaefer100_spearman": float(rho),
        "p": float(p),
    }
    print(f"\nCross-cohort Schaefer-100 agreement: rho={rho:.3f} p={p:.3g} "
          f"(n={int(ok.sum())} parcels)", flush=True)

    (OUT_RES / "image_signature.json").write_text(json.dumps(res, indent=2))
    print("wrote results/image_signature.json and 2 signature PDFs", flush=True)


if __name__ == "__main__":
    main()
