"""Orthogonal-view supplementary artwork for the three anatomical association maps.

External visualization request (2026-10-20): the journal asks that 3D-MRI findings be
shown in at least two perpendicular planes. This script creates NEW supplementary
artwork (axial + coronal multi-slice views) for
  (1) the OASIS-1 Schaefer-100 gray-matter/misfit association map
      (main-text biology composite, panel c, right; supp fig oasis_misfit_signature),
  (2) the OASIS-3 participant-level voxelwise association map
      (supp fig oasis3_signature_participant), and
  (3) in a second part (step E), the corrected ADNI voxelwise association map
      (main-text biology composite, panel c, left; the participant-purged map from
      correction_20261018/integration/).
No existing figure is modified. No new statistics are computed: both maps are loaded
from their saved artifacts and the display conventions of the published panels are
reproduced exactly (z-scored display values, |z|>0.8 threshold, symmetric limit 2.6,
RdBu_r, default MNI152 background).

Steps:
  A. Load saved maps; verify bindings (parcel projection == image_signature.json
     coefficients; OASIS-3 nonzero support == recorded mask size; stats == caption).
  B. Reproduce the published axial displays from the saved maps (verification panels).
  C. Render the new two-plane figure with a systematic slice rule: eight slices per
     plane, equally spaced between the 5th and 95th percentile of the analyzed-mask
     voxel coordinates along that axis (computed in mm, in each map's own grid).
  D. Write PDF + lossless PNG, a provenance manifest (sources, sha256, transforms,
     slice coordinates, color limits, environment), and stdout verification lines.
  E. Repeat the binding verification, published-panel reproduction, and two-plane
     rendering for the corrected ADNI map (ortho_views_adni.{pdf,png}) and extend
     the manifest.

Run: python misfit/ortho_views/make_ortho_views.py   (CPU only)
"""
from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import gridspec  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from nilearn import plotting  # noqa: E402
from nilearn.datasets import fetch_atlas_schaefer_2018  # noqa: E402

ROOT = Path(_resolve_path("<REPO_ROOT>/misfit"))
RES = ROOT / "results"
OUT = ROOT / "ortho_views_20261020" / "out"
OUT.mkdir(parents=True, exist_ok=True)

THRESHOLD, VMAX, CMAP = 0.8, 2.6, "RdBu_r"
N_SLICES, PCT_LO, PCT_HI = 8, 5, 95
plt.rcParams.update({"pdf.fonttype": 42, "font.size": 8.0})


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def zscore_over(values: np.ndarray) -> np.ndarray:
    return (values - values.mean()) / (values.std() + 1e-12)


def mask_world_percentiles(mask: np.ndarray, affine: np.ndarray, axis: int,
                           lo: float, hi: float) -> tuple[float, float]:
    """Percentile range (mm) of analyzed-mask voxel centers along a world axis."""
    ijk = np.argwhere(mask)
    world = nib.affines.apply_affine(affine, ijk)[:, axis]
    return float(np.percentile(world, lo)), float(np.percentile(world, hi))


def slice_coords(mask: np.ndarray, affine: np.ndarray, axis: int) -> list[float]:
    a, b = mask_world_percentiles(mask, affine, axis, PCT_LO, PCT_HI)
    return [float(round(c)) for c in np.linspace(a, b, N_SLICES)]


def row(fig, sub, zimg, mask_img, mode, coords, title):
    ax = fig.add_subplot(sub)
    disp = plotting.plot_stat_map(
        zimg, axes=ax, display_mode=mode, cut_coords=coords, threshold=THRESHOLD,
        vmax=VMAX, cmap=CMAP, colorbar=False, annotate=True, black_bg=False)
    disp.add_contours(mask_img, levels=[0.5], colors="k", linewidths=0.3)
    ax.set_title(title, fontsize=8.0, pad=2)
    return disp


def cohort_colorbar(fig, sub, label):
    ax = fig.add_subplot(sub)
    norm = Normalize(vmin=-VMAX, vmax=VMAX)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=CMAP), cax=ax,
                      orientation="vertical")
    cb.set_ticks([-VMAX, -1.3, -THRESHOLD, 0, THRESHOLD, 1.3, VMAX])
    cb.set_label(label, fontsize=7.5)
    # gray out the sub-threshold band exactly as nilearn's thresholded colorbar does
    ax.axhspan(-THRESHOLD, THRESHOLD, color="#b8b8b8", zorder=5)
    cb.ax.tick_params(labelsize=6.5)


def main() -> None:
    manifest: dict = {"date": "2026-10-20", "purpose":
                      "orthogonal-view supplementary artwork; no new statistics",
                      "display": {"threshold_z": THRESHOLD, "vmax_z": VMAX,
                                  "cmap": CMAP, "background":
                                  "nilearn default MNI152 template",
                                  "zero_vs_unanalyzed": "thin black contour marks the "
                                  "analyzed mask; uncolored inside = |z|<=0.8"},
                      "slice_rule": f"{N_SLICES} slices equally spaced between the "
                      f"{PCT_LO}th and {PCT_HI}th percentile of analyzed-mask voxel "
                      "center coordinates (mm) along the axis, rounded to 1 mm",
                      "environment": {
                          "python": sys.version.split()[0],
                          "numpy": np.__version__,
                          "nibabel": nib.__version__,
                          "matplotlib": matplotlib.__version__,
                          "nilearn": __import__("nilearn").__version__,
                          "scipy": __import__("scipy").__version__}}

    # ---------------- A. load + verify bindings ----------------
    sig_p = RES / "image_signature.json"
    sig = json.load(open(sig_p))["oasis"]
    roi = np.array(sig["roi_signature"], float)
    o1_p = RES / "oasis_misfit_signature.nii.gz"
    o1_img = nib.load(o1_p)
    o1_vol = np.asarray(o1_img.dataobj).astype(np.float32)
    atlas = fetch_atlas_schaefer_2018(n_rois=100)
    lab_img = nib.load(atlas.maps)
    labvol = np.asarray(lab_img.dataobj).astype(int)
    assert o1_vol.shape == labvol.shape and np.allclose(o1_img.affine, lab_img.affine)
    for j in range(1, 101):
        assert np.allclose(o1_vol[labvol == j], roi[j - 1], atol=1e-6), f"parcel {j}"
    assert np.allclose(o1_vol[labvol == 0], 0.0)
    assert (roi < 0).all() and len(roi) == 100 and sig["n"] == 226
    print("OASIS-1 binding: saved NIfTI == image_signature.json projection "
          f"(100/100 parcels); n={sig['n']}, split-half={sig['stability_split_half']:.3f}")

    o3_p = RES / "oasis3" / "oasis3_signature_participant.nii.gz"
    o3_js = json.load(open(RES / "oasis3" / "oasis3_signature_participant.json"))
    o3_img = nib.load(o3_p)
    o3_vol = np.asarray(o3_img.dataobj).astype(np.float32)
    o3_mask = o3_vol != 0
    assert int(o3_mask.sum()) == o3_js["n_voxels"] == 417294
    assert o3_js["n"] == 1360
    frac_neg = float((o3_vol[o3_mask] < 0).mean())
    assert abs(frac_neg - o3_js["frac_negative"]) < 1e-6
    print(f"OASIS-3 binding: mask voxels={int(o3_mask.sum())} == recorded n_voxels; "
          f"n={o3_js['n']}, frac_negative={frac_neg:.4f}, "
          f"split-half={o3_js['split_half']:.3f}")

    # display volumes, exactly the published conventions:
    # OASIS-1: z across the 100 parcel VALUES, then projected (biology panel c).
    roi_z = zscore_over(roi)
    o1_zvol = np.zeros_like(o1_vol)
    for j in range(1, 101):
        o1_zvol[labvol == j] = roi_z[j - 1]
    o1_zimg = nib.Nifti1Image(o1_zvol, lab_img.affine)
    o1_maskimg = nib.Nifti1Image((labvol > 0).astype(np.float32), lab_img.affine)
    # OASIS-3: z across the in-mask VOXELS (supp fig).
    o3_zvol = np.zeros_like(o3_vol)
    o3_zvol[o3_mask] = zscore_over(o3_vol[o3_mask])
    o3_zimg = nib.Nifti1Image(o3_zvol, o3_img.affine)
    o3_maskimg = nib.Nifti1Image(o3_mask.astype(np.float32), o3_img.affine)

    # ---------------- B. reproduce the published axial displays ----------------
    fig = plt.figure(figsize=(7.4, 5.6))
    gs = gridspec.GridSpec(2, 1, hspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    plotting.plot_stat_map(o1_zimg, axes=ax1, colorbar=True, display_mode="z",
                           cut_coords=[-20, 10, 45], threshold=THRESHOLD, vmax=VMAX,
                           cmap=CMAP, annotate=False)
    ax1.set_title("reproduction of main Fig (biology, c-right): OASIS-1 "
                  "Schaefer-100 GM association ($z$), cuts z=-20,10,45", fontsize=8)
    ax2 = fig.add_subplot(gs[1])
    d3 = plotting.plot_stat_map(o3_zimg, axes=ax2, colorbar=True, display_mode="z",
                                cut_coords=6, threshold=THRESHOLD, vmax=VMAX,
                                cmap=CMAP)
    auto_cuts = [float(c) for c in d3.cut_coords]
    ax2.set_title("reproduction of supp fig: OASIS-3 participant-level association "
                  f"($z$), auto cuts z={[round(c) for c in auto_cuts]}", fontsize=8)
    fig.savefig(OUT / "verification_axial_reproduction.png", dpi=200,
                bbox_inches="tight")
    fig.savefig(OUT / "verification_axial_reproduction.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"reproduced OASIS-3 auto axial cuts: {auto_cuts}")

    # ---------------- C. the new two-plane figure ----------------
    o1_ax = slice_coords(labvol > 0, lab_img.affine, 2)
    o1_co = slice_coords(labvol > 0, lab_img.affine, 1)
    o3_ax = slice_coords(o3_mask, o3_img.affine, 2)
    o3_co = slice_coords(o3_mask, o3_img.affine, 1)
    manifest["slices_mm"] = {"oasis1_axial_z": o1_ax, "oasis1_coronal_y": o1_co,
                             "oasis3_axial_z": o3_ax, "oasis3_coronal_y": o3_co}
    print("slice coordinates (mm):", manifest["slices_mm"])

    fig = plt.figure(figsize=(7.4, 7.6))
    gsm = gridspec.GridSpec(4, 2, width_ratios=[1.0, 0.018], hspace=0.16,
                            wspace=0.02)
    row(fig, gsm[0, 0], o1_zimg, o1_maskimg, "z", o1_ax,
        "(a) OASIS-1 ($N{=}226$): Schaefer-100 GM association ($z$), axial")
    row(fig, gsm[1, 0], o1_zimg, o1_maskimg, "y", o1_co,
        "OASIS-1, coronal")
    row(fig, gsm[2, 0], o3_zimg, o3_maskimg, "z", o3_ax,
        "(b) OASIS-3 ($N{=}1360$, participant-level): voxelwise GM association "
        "($z$), axial")
    row(fig, gsm[3, 0], o3_zimg, o3_maskimg, "y", o3_co,
        "OASIS-3, coronal")
    cb1 = gridspec.GridSpecFromSubplotSpec(1, 1, gsm[0:2, 1])
    cohort_colorbar(fig, cb1[0], "OASIS-1 display $z$ (per-parcel)")
    cb2 = gridspec.GridSpecFromSubplotSpec(1, 1, gsm[2:4, 1])
    cohort_colorbar(fig, cb2[0], "OASIS-3 display $z$ (per-voxel)")
    fig.savefig(OUT / "ortho_views_oasis1_oasis3.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT / "ortho_views_oasis1_oasis3.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote ortho_views_oasis1_oasis3.{pdf,png}")

    # ---------------- D. manifest ----------------
    manifest["sources"] = {
        "oasis1_map_nii": {"path": str(o1_p), "sha256": sha256(o1_p),
                           "content": "raw per-parcel Spearman(GM density, "
                           "seed-averaged misfit) projected on Schaefer-100"},
        "oasis1_coefficients": {"path": str(sig_p), "sha256": sha256(sig_p),
                                "field": "oasis.roi_signature (100 raw values)"},
        "oasis3_map_nii": {"path": str(o3_p), "sha256": sha256(o3_p),
                           "content": "raw voxelwise Spearman(GM density, canonical "
                           "5-seed misfit), leakage-free cohort, zeros outside mask"},
        "atlas": {"name": "Schaefer-2018 100-parcel, 7 networks, 1 mm MNI152",
                  "fetched_via": "nilearn.datasets.fetch_atlas_schaefer_2018",
                  "maps_sha256": sha256(Path(atlas.maps))},
        "oasis3_grid": {"shape": list(o3_img.shape),
                        "voxel_mm": [float(x) for x in
                                     nib.affines.voxel_sizes(o3_img.affine)],
                        "space": "SPM DARTEL population template normalized to MNI "
                        "(route-B smwc1 grid)"}}
    manifest["transformations"] = {
        "oasis1": "z-score across the 100 parcel values, project constant per parcel "
                  "(identical to published panel); NO per-slice or per-plane scaling",
        "oasis3": "z-score across the 417,294 in-mask voxels (identical to published "
                  "figure); NO per-slice or per-plane scaling"}
    manifest["reproduction_check"] = {
        "oasis1_parcel_binding": "100/100 parcels equal saved NIfTI (atol 1e-6)",
        "oasis3_mask_binding": "nonzero support == recorded 417,294 voxels; "
                               f"frac_negative {frac_neg:.6f} == JSON",
        "published_axial_cuts": {"oasis1_biology_panel": [-20, 10, 45],
                                 "oasis3_supp_auto": auto_cuts}}

    # ---------------- E. corrected-ADNI extension (2026-10-20, second round) ----------
    # Closes the whole-submission orthogonal-plane gap: the third anatomical
    # association map, the corrected ADNI voxelwise map of the main-text composite
    # (panel c left), gets the same axial + coronal treatment as its own figure.
    ad_p = ROOT / "correction_20261018" / "integration" / "results" / \
        "adni_misfit_signature_corrected.nii.gz"
    ad_js = json.load(open(ROOT / "correction_20261018" / "integration" / "results" /
                           "adni_image_signature_corrected.json"))
    ad_img = nib.load(ad_p)
    ad_vol = np.asarray(ad_img.dataobj).astype(np.float32)
    ad_mask = ad_vol != 0
    assert int(ad_mask.sum()) == ad_js["n_voxels"] == 435026
    assert ad_js["n"] == 731
    ad_frac_neg = float((ad_vol[ad_mask] < 0).mean())
    assert abs(100.0 * ad_frac_neg - ad_js["pct_voxels_negative"]) < 1e-6
    print(f"ADNI binding: mask voxels={int(ad_mask.sum())} == recorded n_voxels; "
          f"n={ad_js['n']} acquisitions, frac_negative={ad_frac_neg:.4f}, "
          f"split-half={ad_js['stability_split_half']:.3f}")

    ad_zvol = np.zeros_like(ad_vol)
    ad_zvol[ad_mask] = zscore_over(ad_vol[ad_mask])
    ad_zimg = nib.Nifti1Image(ad_zvol, ad_img.affine)
    ad_maskimg = nib.Nifti1Image(ad_mask.astype(np.float32), ad_img.affine)

    # reproduce the published main-composite ADNI panel (cuts z=-20,10,45)
    fig = plt.figure(figsize=(7.4, 2.8))
    ax = fig.add_subplot(111)
    plotting.plot_stat_map(ad_zimg, axes=ax, colorbar=True, display_mode="z",
                           cut_coords=[-20, 10, 45], threshold=THRESHOLD, vmax=VMAX,
                           cmap=CMAP, annotate=False)
    ax.set_title("reproduction of main Fig (biology, c-left): ADNI (corrected) "
                 "voxelwise GM association ($z$), cuts z=-20,10,45", fontsize=8)
    fig.savefig(OUT / "verification_adni_reproduction.png", dpi=200,
                bbox_inches="tight")
    fig.savefig(OUT / "verification_adni_reproduction.pdf", bbox_inches="tight")
    plt.close(fig)

    ad_ax = slice_coords(ad_mask, ad_img.affine, 2)
    ad_co = slice_coords(ad_mask, ad_img.affine, 1)
    manifest["slices_mm"]["adni_axial_z"] = ad_ax
    manifest["slices_mm"]["adni_coronal_y"] = ad_co
    print("ADNI slice coordinates (mm):", {"axial_z": ad_ax, "coronal_y": ad_co})

    fig = plt.figure(figsize=(7.4, 3.9))
    gsa = gridspec.GridSpec(2, 2, width_ratios=[1.0, 0.018], hspace=0.16,
                            wspace=0.02)
    row(fig, gsa[0, 0], ad_zimg, ad_maskimg, "z", ad_ax,
        "ADNI (corrected; $731$ acquisitions from $547$ participants): "
        "voxelwise GM association ($z$), axial")
    row(fig, gsa[1, 0], ad_zimg, ad_maskimg, "y", ad_co, "ADNI (corrected), coronal")
    cba = gridspec.GridSpecFromSubplotSpec(1, 1, gsa[0:2, 1])
    cohort_colorbar(fig, cba[0], "ADNI display $z$ (per-voxel)")
    fig.savefig(OUT / "ortho_views_adni.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT / "ortho_views_adni.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote ortho_views_adni.{pdf,png}")

    manifest["sources"]["adni_map_nii"] = {
        "path": str(ad_p), "sha256": sha256(ad_p),
        "content": "raw voxelwise Spearman(GM density, corrected V2 misfit, "
                   "seeds 42-46 mean), 731 acquisitions / 547 participants, "
                   "zeros outside mask"}
    manifest["transformations"]["adni"] = (
        "z-score across the 435,026 in-mask voxels (identical to the main-text "
        "composite panel); NO per-slice or per-plane scaling")
    manifest["reproduction_check"]["adni_mask_binding"] = (
        "nonzero support == recorded 435,026 voxels; "
        f"pct_negative {100.0 * ad_frac_neg:.4f} == JSON")
    manifest["reproduction_check"]["published_axial_cuts"]["adni_biology_panel"] = \
        [-20, 10, 45]
    json.dump(manifest, open(OUT / "provenance_manifest.json", "w"), indent=2)
    print("wrote provenance_manifest.json")


if __name__ == "__main__":
    main()
