"""Re-plot the anatomical-signature figures from saved maps (no recompute):
standardize (z-score across voxels/parcels) so the diffuse all-negative field
shows spatial structure on a diverging colormap, use clean panel titles (no
heavy black title bars), and make the ADNI main figure a single compact panel
to reduce its visual dominance (per reviewer feedback). All raw correlations
remain negative; standardization is for display only.
"""
from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from nilearn import plotting  # noqa: E402

RES = Path(_resolve_path("<REPO_ROOT>/misfit/results"))
FIG = Path(_resolve_path("<REPO_ROOT>/misfit/paper/figures"))
plt.rcParams.update({"pdf.fonttype": 42, "font.size": 10})


def zscore_brain(vol):
    """Z-score a volume over its nonzero voxels; zeros stay zero (background)."""
    m = vol != 0
    out = np.zeros_like(vol, dtype=np.float32)
    v = vol[m]
    out[m] = (v - v.mean()) / (v.std() + 1e-12)
    return out


# ---------------------------------------------------------------- ADNI (main)
img = nib.load(RES / "adni_misfit_signature.nii.gz")
vol = np.asarray(img.dataobj)
raw = vol[vol != 0]
zimg = nib.Nifti1Image(zscore_brain(vol), img.affine)

fig = plt.figure(figsize=(7.8, 2.7))
disp = plotting.plot_glass_brain(zimg, figure=fig, colorbar=True, plot_abs=False,
                                 vmax=2.6, cmap="RdBu_r")
fig.suptitle("ADNI: standardized gray-matter association with structural misfit",
             fontsize=11, y=1.02)
fig.savefig(FIG / "adni_misfit_signature.pdf", bbox_inches="tight", dpi=300)
plt.close(fig)
print(f"ADNI: raw corr range [{raw.min():.3f}, {raw.max():.3f}] (all negative); "
      f"z-scored for display", flush=True)

# ---------------------------------------------------------------- OASIS (supp)
res = json.load(open(RES / "image_signature.json"))
roi = np.array(res["oasis"]["roi_signature"], float)
roi_z = (roi - roi.mean()) / (roi.std() + 1e-12)
from nilearn.datasets import fetch_atlas_schaefer_2018

atlas = fetch_atlas_schaefer_2018(n_rois=100)
lab_img = nib.load(atlas.maps)
labvol = np.asarray(lab_img.dataobj).astype(int)
statvol = np.zeros(labvol.shape, np.float32)
for j in range(1, 101):
    statvol[labvol == j] = roi_z[j - 1]
stat_img = nib.Nifti1Image(statvol, lab_img.affine)

fig = plt.figure(figsize=(7.4, 2.9))
plotting.plot_stat_map(stat_img, figure=fig, colorbar=True, display_mode="z",
                       cut_coords=6, vmax=2.6, cmap="RdBu_r")
fig.suptitle("OASIS-1: standardized Schaefer-100 gray-matter association with misfit",
             fontsize=10.5, y=1.0)
fig.savefig(FIG / "oasis_misfit_signature.pdf", bbox_inches="tight", dpi=300)
plt.close(fig)
print(f"OASIS: raw ROI corr range [{roi.min():.3f}, {roi.max():.3f}] (all negative); "
      f"z-scored for display", flush=True)
print("wrote adni_misfit_signature.pdf + oasis_misfit_signature.pdf", flush=True)
