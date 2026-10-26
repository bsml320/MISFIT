"""Render the four-panel summary dashboard for the OASIS-3 study.

Inputs: data/oasis3/_fancy/misfit_gm_spearman.nii.gz (voxelwise rho
map), data/oasis3/_fancy/per_subject.npz (tested mask, misfit, err),
misfit/results/oasis3/oasis3_per_scanner.json (per-scanner rho
mean/std), plus external reporting_references.json, key dashboard_rows.
Panels: (a) glass-brain anatomical signature, (b) cross-cohort AUROC
dot plot, (c) per-scanner rho bars, (d) selective-prediction curve.
Output: misfit/results/oasis3/fancy_figs/fig_dashboard.pdf.
Run as a plain script with no arguments.
"""

from pathlib import Path
from misfit.paths import resolve_path as _resolve_path, reporting_reference
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, json, nibabel as nib
import matplotlib.gridspec as gs
from nilearn import plotting, image
plt.rcParams.update({"pdf.fonttype":42,"savefig.dpi":300})
FF=_resolve_path("<REPO_ROOT>/data/oasis3/_fancy/"); RES=_resolve_path("<REPO_ROOT>/misfit/results/oasis3/")
PUR,ACC,GRAY,ADc,GENc="#6a1b9a","#c0392b","#9aa0a6","#8e44ad","#2c7fb8"
fig=plt.figure(figsize=(11,7.6))
G=gs.GridSpec(2,2,figure=fig,hspace=0.32,wspace=0.22,height_ratios=[1,1])

# (a) glass brain
axA=fig.add_subplot(G[0,0])
simg=image.smooth_img(nib.load(FF+"misfit_gm_spearman.nii.gz"),fwhm=4)
plotting.plot_glass_brain(simg,display_mode="lyrz",colorbar=True,plot_abs=False,cmap="RdBu_r",
    vmin=-0.35,vmax=0.35,threshold=0.08,axes=axA)
axA.set_title("(a) Anatomical signature: lower GM $\\leftrightarrow$ higher misfit",fontsize=10,fontweight="bold")

# (b) cross-cohort AUROC + rho
axB=fig.add_subplot(G[0,1])
C=reporting_reference("dashboard_rows")
y=np.arange(len(C))
for i,(nm,r,rc,au,ac,col,mk) in enumerate(C):
    axB.errorbar(au,i,xerr=ac,fmt="none",ecolor=col,elinewidth=1.6,capsize=3,alpha=.9)
    axB.scatter(au,i,s=110 if mk=="D" else 60,color=col,marker=mk,edgecolor="white",lw=1.3,zorder=3)
    axB.text(au,i+0.26,f"{au:.2f}",ha="center",fontsize=7.5,color=col,fontweight="bold" if mk=="D" else "normal")
    axB.text(0.505,i,f"$\\rho${r:.2f}",fontsize=7,color="#666",va="center")
axB.axvline(0.5,color=GRAY,ls=(0,(4,3)),lw=1); axB.text(0.5,len(C)-0.4,"chance",fontsize=7,color="#7a7f85",ha="center")
axB.set_yticks(y); axB.set_yticklabels([c[0] for c in C],fontsize=9); axB.set_xlim(0.5,1.0)
axB.set_xlabel("Failure-detection AUROC"); axB.set_title("(b) Cross-cohort comparison",fontsize=10,fontweight="bold")
import seaborn as sns; sns.despine(ax=axB,left=True); axB.tick_params(left=False); axB.grid(axis="x",alpha=.3)

# (c) per-scanner rho bars
axC=fig.add_subplot(G[1,0])
ps=[r for r in json.load(open(RES+"oasis3_per_scanner.json")) if r["rho_mean"] is not None]
names=[r["scanner"].replace("_"," ")+f"\n({'3T' if r['scanner'] in ('TrioTim','Biograph_mMR','MAGNETOM_Vida') else '1.5T'})" for r in ps]
rr=[r["rho_mean"] for r in ps]; ss=[r["rho_std"] for r in ps]
cols=[ "#5b2c8f" if r["scanner"] in ('TrioTim','Biograph_mMR','MAGNETOM_Vida') else "#b07fd6" for r in ps]
axC.bar(range(len(ps)),rr,yerr=ss,color=cols,edgecolor="white",lw=.7,capsize=3)
pooled_rho = next(row[1] for row in C if row[0] == "OASIS-3")
axC.axhline(pooled_rho,color=ACC,ls=(0,(4,3)),lw=1.3); axC.text(0.1,pooled_rho+0.016,f"pooled {pooled_rho:.2f}",color=ACC,fontsize=8)
axC.set_xticks(range(len(ps))); axC.set_xticklabels(names,fontsize=7.5)
axC.set_ylabel("Within-scanner $\\rho$"); axC.set_ylim(0,0.72)
axC.set_title("(c) Within-scanner associations (OASIS-3)",fontsize=10,fontweight="bold"); sns.despine(ax=axC)

# (d) selective + triage
axD=fig.add_subplot(G[1,1])
d=np.load(FF+"per_subject.npz",allow_pickle=True); m=d["tested"]; mis=d["misfit"][m]; err=d["err"][m]
n=len(mis); o=np.argsort(mis); cov=np.arange(0.1,1.001,0.02)
acc=[1-err[o[:max(1,int(round(c*n)))]].mean() for c in cov]; base=1-err.mean(); i50=int(np.argmin(np.abs(cov-.5)))
axD.plot(cov*100,acc,color=PUR,lw=2.4,label="accepted-set accuracy")
axD.axhline(base,color=GRAY,ls=(0,(4,3)),lw=1); axD.scatter([50],[acc[i50]],s=55,color=ACC,zorder=5,edgecolor="white",lw=1.2)
axD.annotate(f"+{(acc[i50]-base)*100:.0f} pp @50%",xy=(50,acc[i50]),xytext=(54,acc[i50]-.08),color=ACC,fontsize=9,fontweight="bold",
   arrowprops=dict(arrowstyle="->",color=ACC,lw=1.2))
axD.set_xlabel("Coverage retained (%)"); axD.set_ylabel("Accepted-set accuracy")
axD.set_title("(d) Misfit-guided selective prediction",fontsize=10,fontweight="bold"); axD.set_xlim(10,100); sns.despine(ax=axD)
fig.suptitle("Structural misfit: a per-subject reliability signal for multi-site Alzheimer MRI",fontsize=13,fontweight="bold",y=0.99)
Path(RES, "fancy_figs").mkdir(parents=True, exist_ok=True)
fig.savefig(RES+"fancy_figs/fig_dashboard.pdf",bbox_inches="tight"); print("wrote fig_dashboard.pdf")
