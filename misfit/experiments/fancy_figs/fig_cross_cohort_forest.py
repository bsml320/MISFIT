"""Render the five-cohort forest plot of per-subject reliability.

Input: external reporting_references.json, key cross_cohort_forest
(within-environment Spearman rho and failure-detection AUROC, each with
an optional 1.96*sd half-width), taken from upstream experiment runs.
Output: misfit/paper/figures/cross_cohort_forest.pdf with paired
panels for rho and AUROC; the disease-matched replication cohort is
drawn with a distinct marker and color.
Run as a plain script with no arguments.
"""

from pathlib import Path
from misfit.paths import resolve_path as _resolve_path, reporting_reference
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, seaborn as sns
sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
plt.rcParams.update({"pdf.fonttype":42,"savefig.dpi":300,"font.family":"DejaVu Sans"})
OUT=_resolve_path("<REPO_ROOT>/misfit/paper/figures/cross_cohort_forest.pdf")
# cohort: (label, sublabel, within-env rho, rho_ci(1.96*sd or None), AUROC, auroc_ci, is_AD, highlight)
C=reporting_reference("cross_cohort_forest")
ADc, GENc, HL = "#8e44ad", "#2c7fb8", "#c0392b"
fig,(a1,a2)=plt.subplots(1,2,figsize=(7.4,3.4),sharey=True)
y=np.arange(len(C))
for ax,(vi,ci_i),xl,ref,reflab,xlim in [
    (a1,(2,3),"Within-environment Spearman $\\rho$",0.0,"no signal",(-0.05,1.0)),
    (a2,(4,5),"Failure-detection AUROC",0.5,"chance",(0.5,1.0))]:
    ax.axvline(ref,color="#b0b4b8",ls=(0,(4,3)),lw=1.2,zorder=0)
    ax.text(ref,len(C)-0.35,reflab,fontsize=7,color="#7a7f85",ha="center",va="bottom")
    for i,row in enumerate(C):
        val=row[vi]; ci=row[ci_i]; col=HL if row[7] else (ADc if row[6] else GENc)
        ms=120 if row[7] else 62
        if ci: ax.errorbar(val,i,xerr=ci,fmt="none",ecolor=col,elinewidth=1.8,capsize=3,zorder=2,alpha=0.9)
        ax.scatter(val,i,s=ms,color=col,edgecolor="white",lw=1.4,zorder=3,
                   marker="D" if row[7] else "o")
        ax.text(val,i+0.28,f"{val:.3f}",fontsize=7.6,ha="center",color=col,fontweight="bold" if row[7] else "normal")
    ax.set_xlim(*xlim); ax.set_xlabel(xl,fontsize=9.5)
a1.set_yticks(y); a1.set_yticklabels([f"{c[0]}" for c in C],fontsize=9.5)
# sublabels
for i,c in enumerate(C):
    a1.text(-0.05,i-0.30,c[1],fontsize=6.7,color="#8a8f95",ha="left",va="top",transform=a1.get_yaxis_transform() if False else a1.transData)
for ax in (a1,a2): sns.despine(ax=ax,left=True); ax.tick_params(left=False)
a1.set_ylim(-0.6,len(C)-0.1)
from matplotlib.lines import Line2D
leg=[Line2D([0],[0],marker="D",color="w",markerfacecolor=HL,markersize=9,label="OASIS-3 disease-matched replication"),
     Line2D([0],[0],marker="o",color="w",markerfacecolor=ADc,markersize=8,label="Alzheimer cohorts"),
     Line2D([0],[0],marker="o",color="w",markerfacecolor=GENc,markersize=8,label="Generalization cohorts")]
fig.legend(handles=leg,loc="lower center",ncol=3,fontsize=7.6,frameon=False,bbox_to_anchor=(0.5,-0.02))
fig.suptitle("Retrospective per-subject reliability across five cohorts",
             fontsize=11,fontweight="bold",y=1.0)
fig.tight_layout(rect=(0,0.05,1,0.97))
Path(OUT).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT,bbox_inches="tight",dpi=300); print("wrote",OUT)
