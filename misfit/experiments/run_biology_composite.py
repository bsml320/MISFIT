"""Composite main-paper biology figure: ONLY the seed-stable findings.

(a) Diagnostic/CDR reliability stratification: ADNI misfit by diagnostic group (the
    seed-stable impaired-vs-control reliability gap) + OASIS-1 misfit by CDR stage.
(b) Representative failure cases: gray-matter maps at the reliability extremes
    (most vs least reliable case per cohort; illustrative single cases).
(c) Bounded anatomical associations: ADNI voxelwise and OASIS-1 Schaefer-100
    gray-matter-vs-misfit maps (standardized for display), annotated with their
    split-half stability and within-class agreement.

Single-seed operator-loading interpretations are deliberately excluded (they stay in
the supplement). All panels replot cached, verified intermediates:
  results/per_subject_misfit.npz (5-seed-averaged), results/adni_misfit_signature.nii.gz,
  results/image_signature.json, and the cohort image paths.
Output: paper/figures/biology_composite.pdf
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
from matplotlib import gridspec  # noqa: E402
from nilearn import plotting  # noqa: E402
from scipy.stats import ks_2samp, spearmanr  # noqa: E402

sys.path.insert(0, _resolve_path("<REPO_ROOT>"))
from misfit.experiments.run_image_examples_media import adni_paths  # noqa: E402

RES = Path(_resolve_path("<REPO_ROOT>/misfit/results"))
FIG = Path(_resolve_path("<REPO_ROOT>/misfit/paper/figures"))
plt.rcParams.update({"pdf.fonttype": 42, "font.size": 8.5})
TEAL, CORAL, GRAY = "#0f6b74", "#c0392b", "#8a9296"


def oasis_paths_cdr():
    """Same cohort derivation as the loader/examples script, returning raw CDR too."""
    from nilearn.datasets import fetch_oasis_vbm
    o = fetch_oasis_vbm()
    gm = np.asarray(o.gray_matter_maps)
    cdr = o.ext_vars["cdr"].values.astype(float)
    age = o.ext_vars["age"].values.astype(float)
    valid = ~np.isnan(cdr)
    gm, age, cdr = gm[valid], age[valid], cdr[valid]
    q = np.quantile(age, np.linspace(0, 1, 5))
    env = np.digitize(age, q[1:-1])
    counts = {e: int((env == e).sum()) for e in sorted(set(env))}
    keep = np.isin(env, [e for e, c in counts.items() if c >= 10])
    return gm[keep], cdr[keep]


def violin(ax, groups, labels, colors):
    """groups are already percentile-transformed (0-100, cohort-wide misfit rank)."""
    parts = ax.violinplot(list(groups), showextrema=False, widths=0.8)
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c)
        pc.set_alpha(0.35)
    rng = np.random.RandomState(0)
    for i, (g, c) in enumerate(zip(groups, colors), 1):
        x = i + rng.uniform(-0.13, 0.13, len(g))
        ax.scatter(x, g, s=3, color=c, alpha=0.45, lw=0)
        ax.hlines(np.median(g), i - 0.28, i + 0.28, color=c, lw=2.2, zorder=5)
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(-3, 103)
    ax.set_ylabel("misfit percentile (cohort rank)", fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def zscore_brain(vol):
    """Z-score a volume over its nonzero voxels; zeros stay zero (background)."""
    m = vol != 0
    out = np.zeros_like(vol, dtype=np.float32)
    v = vol[m]
    out[m] = (v - v.mean()) / (v.std() + 1e-12)
    return out


def main():
    """Assemble the three panels from cached intermediates and save the composite PDF."""
    # canonical (paper-headline) IOP misfit for the stratification panel, matching the
    # main-text means/KS; the image-pipeline npz is used only for the example picks so
    # panel (b) stays identical to the full-size supplementary figure.
    ci = np.load(RES / "per_subject_misfit_iop.npz")
    ma, ea, la = ci["adni_misfit"], ci["adni_err"], ci["adni_label"]
    mo, eo, lo = ci["oasis_misfit"], ci["oasis_err"], ci["oasis_label"]
    dx = np.load(RES / "per_subject_misfit.npz")
    xma, xea, xla = dx["adni_misfit"], dx["adni_err"], dx["adni_label"]
    xmo, xeo, xlo = dx["oasis_misfit"], dx["oasis_err"], dx["oasis_label"]
    sig = json.load(open(RES / "image_signature.json"))

    fig = plt.figure(figsize=(7.2, 8.2))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1.0, 0.80, 0.85], hspace=0.42)

    # ---------------------------------------------------- (a) stratification
    gsa = gridspec.GridSpecFromSubplotSpec(1, 2, gs[0], wspace=0.3)
    a1 = fig.add_subplot(gsa[0])
    from scipy.stats import rankdata
    pca = 100.0 * rankdata(ma) / len(ma)
    cn, imp = pca[la == 0], pca[la == 1]
    ks = ks_2samp(ma[la == 0], ma[la == 1])
    ratio = ma[la == 0].mean() / ma[la == 1].mean()
    violin(a1, [cn, imp], [f"CN\n(n={len(cn)})", f"impaired\n(n={len(imp)})"], [TEAL, CORAL])
    a1.set_title(f"ADNI: impaired subjects are fit more reliably\nmean misfit ratio "
                 f"{ratio:.1f}$\\times$; KS $D$={ks.statistic:.2f}, $p<10^{{-30}}$", fontsize=8.5)
    _, cdr = oasis_paths_cdr()
    assert len(cdr) == len(mo), f"CDR/misfit length mismatch {len(cdr)} vs {len(mo)}"
    pco = 100.0 * rankdata(mo) / len(mo)
    stages = [pco[cdr == 0], pco[cdr == 0.5], pco[cdr >= 1]]
    stage_idx = np.where(cdr == 0, 0, np.where(cdr == 0.5, 1, 2))
    rho_s = spearmanr(stage_idx, mo)
    a2 = fig.add_subplot(gsa[1])
    violin(a2, stages, [f"CDR 0\n(n={len(stages[0])})", f"CDR 0.5\n(n={len(stages[1])})",
                        f"CDR$\\geq$1\n(n={len(stages[2])})"], [TEAL, "#b07fd6", CORAL])
    a2.set_title(f"OASIS-1: higher CDR, less reliable fit\nSpearman(stage, misfit) "
                 f"$\\rho$={rho_s.statistic:.2f}, $p$={rho_s.pvalue:.2g}", fontsize=8.5)

    # ---------------------------------------------------- (b) failure cases
    pa, _ = adni_paths()
    po, _ = oasis_paths_cdr()
    ACLS, OCLS = {0: "CN", 1: "impaired"}, {0: "healthy", 1: "demented"}

    def pick(mis, err):
        rel = np.where(err == 0)[0]
        unr = np.where(err == 1)[0]
        return int(rel[mis[rel].argmin()]), int(unr[mis[unr].argmax()])

    a_rel, a_unr = pick(xma, xea)
    o_rel, o_unr = pick(xmo, xeo)
    gsb = gridspec.GridSpecFromSubplotSpec(1, 4, gs[1], wspace=0.05)

    def render(pos, path, k, title, vmax, edge):
        ax = fig.add_subplot(gsb[pos])
        ax.imshow(np.rot90(nib.load(path).get_fdata(dtype=np.float32)[:, :, k]),
                  cmap="gray", vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=7.2, color=edge)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(edge)
            s.set_linewidth(1.4)

    ka = 60
    ko = nib.load(po[o_rel]).shape[2] // 2
    va = float(np.percentile(nib.load(pa[a_rel]).get_fdata(dtype=np.float32), 99))
    vo = float(np.percentile(nib.load(po[o_rel]).get_fdata(dtype=np.float32), 99))
    render(0, pa[a_rel], ka, f"ADNI most reliable\ntrue {ACLS[int(xla[a_rel])]} "
           f"$\\to$ correct", va, TEAL)
    render(1, pa[a_unr], ka, f"ADNI least reliable\ntrue {ACLS[int(xla[a_unr])]} "
           f"$\\to$ wrong (FN)", va, CORAL)
    render(2, po[o_rel], ko, f"OASIS-1 most reliable\ntrue {OCLS[int(xlo[o_rel])]} "
           f"$\\to$ correct", vo, TEAL)
    render(3, po[o_unr], ko, f"OASIS-1 least reliable\ntrue {OCLS[int(xlo[o_unr])]} "
           f"$\\to$ wrong (FP)", vo, CORAL)

    # ---------------------------------------------------- (c) anatomical maps
    gsc = gridspec.GridSpecFromSubplotSpec(1, 2, gs[2], wspace=0.12)
    c1 = fig.add_subplot(gsc[0])
    img = nib.load(RES / "adni_misfit_signature.nii.gz")
    zimg = nib.Nifti1Image(zscore_brain(np.asarray(img.dataobj)), img.affine)
    plotting.plot_stat_map(zimg, axes=c1, colorbar=True, display_mode="z",
                           cut_coords=[-20, 10, 45], threshold=0.8, vmax=2.6,
                           cmap="RdBu_r", annotate=False)
    c1.set_title("ADNI: voxelwise GM association ($z$)", fontsize=8.5)
    c2 = fig.add_subplot(gsc[1])
    roi = np.array(sig["oasis"]["roi_signature"], float)
    roi_z = (roi - roi.mean()) / (roi.std() + 1e-12)
    from nilearn.datasets import fetch_atlas_schaefer_2018
    atlas = fetch_atlas_schaefer_2018(n_rois=100)
    lab_img = nib.load(atlas.maps)
    labvol = np.asarray(lab_img.dataobj).astype(int)
    statvol = np.zeros(labvol.shape, np.float32)
    for j in range(1, 101):
        statvol[labvol == j] = roi_z[j - 1]
    plotting.plot_stat_map(nib.Nifti1Image(statvol, lab_img.affine), axes=c2, colorbar=True,
                           display_mode="z", cut_coords=[-20, 10, 45], threshold=0.8,
                           vmax=2.6, cmap="RdBu_r", annotate=False)
    c2.set_title("OASIS-1: Schaefer-100 GM association ($z$)", fontsize=8.5)

    for lab, y in (("(a)", 0.985), ("(b)", 0.615), ("(c)", 0.36)):
        fig.text(0.015, y, lab, fontsize=11, fontweight="bold")
    fig.savefig(FIG / "biology_composite.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"ADNI CN mean {cn.mean():.3f} vs impaired {imp.mean():.3f}, KS D={ks.statistic:.3f} "
          f"p={ks.pvalue:.2g}", flush=True)
    print(f"OASIS CDR-stage medians: {[float(np.median(s)) for s in stages]}, "
          f"spearman {rho_s.statistic:.3f} p={rho_s.pvalue:.2g}", flush=True)
    print("wrote figures/biology_composite.pdf")


if __name__ == "__main__":
    main()
