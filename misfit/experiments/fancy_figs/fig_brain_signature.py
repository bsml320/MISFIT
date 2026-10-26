"""Render the OASIS-3 anatomical-signature figure.

Input: precomputed voxelwise correlation map
data/oasis3/_fancy/misfit_gm_spearman.nii.gz (Spearman rho between
gray-matter density and per-subject structural misfit).
Output: misfit/results/oasis3/fancy_figs/fig_brain_signature.pdf
with a glass-brain overview (panel a) and an axial montage (panel b).
Run as a plain script with no arguments.
"""

from misfit.paths import resolve_path as _resolve_path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, nibabel as nib
from nilearn import plotting, image
plt.rcParams.update({"pdf.fonttype":42,"savefig.dpi":300})
MAP=_resolve_path("<REPO_ROOT>/data/oasis3/_fancy/misfit_gm_spearman.nii.gz")
img=nib.load(MAP); dat=img.get_fdata()
# Light smoothing for display only; the colormap stays on the raw rho scale.
simg=image.smooth_img(img,fwhm=4)
vmax=0.35
fig=plt.figure(figsize=(9.2,5.4))
# (a) glass brain, all projections
ax1=fig.add_axes([0.02,0.52,0.96,0.44])
gb=plotting.plot_glass_brain(simg,display_mode="lyrz",colorbar=True,plot_abs=False,
     cmap="RdBu_r",vmin=-vmax,vmax=vmax,threshold=0.08,axes=ax1,
     title="OASIS-3 anatomical signature of unreliability")
# (b) axial montage on MNI
ax2=fig.add_axes([0.02,0.04,0.96,0.42])
plotting.plot_stat_map(simg,display_mode="z",cut_coords=[-20,-8,4,20,36,52],
     colorbar=True,cmap="RdBu_r",vmax=vmax,threshold=0.08,axes=ax2,black_bg=False,
     title="")
fig.text(0.5,0.475,"Voxelwise Spearman $\\rho$ between gray-matter density and per-subject structural misfit "
         "(2,720 sessions). Blue: lower gray matter $\\leftrightarrow$ higher misfit.",
         ha="center",fontsize=8.4,color="#333")
fig.savefig(_resolve_path("<REPO_ROOT>/misfit/results/oasis3/fancy_figs/fig_brain_signature.pdf"),bbox_inches="tight")
print("wrote fig_brain_signature.pdf")
