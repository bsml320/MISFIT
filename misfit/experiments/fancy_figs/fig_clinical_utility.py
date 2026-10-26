"""Render the OASIS-3 clinical-utility figure (three panels).

Input: data/oasis3/_fancy/per_subject.npz with arrays tested (mask),
misfit, err, pred, true; only tested subjects are used.
Panels: (a) selective-prediction frontier (accepted-set accuracy vs
coverage when the lowest-misfit fraction is retained), (b) triage
error-capture curve (misclassifications caught when reviewing the
highest-misfit subjects first, vs random review), (c) empirical error
rate per misfit decile.
Output: misfit/results/oasis3/fancy_figs/fig_clinical_utility.pdf.
Run as a plain script with no arguments.
"""

from misfit.paths import resolve_path as _resolve_path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, seaborn as sns
sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
plt.rcParams.update({"pdf.fonttype":42,"savefig.dpi":300})
d=np.load(_resolve_path("<REPO_ROOT>/data/oasis3/_fancy/per_subject.npz"),allow_pickle=True)
m=d["tested"]; mis=d["misfit"][m]; err=d["err"][m]; pred=d["pred"][m]; true=d["true"][m]
n=len(mis); order=np.argsort(mis)
PUR,ACC,GRAY="#6a1b9a","#c0392b","#9aa0a6"
fig,ax=plt.subplots(1,3,figsize=(9.6,3.2))

# (a) selective-prediction frontier
cov=np.arange(0.1,1.001,0.02)
acc=[1-err[order[:max(1,int(round(c*n)))]].mean() for c in cov]
base=1-err.mean(); i50=int(np.argmin(np.abs(cov-0.5)))
ax[0].fill_between(cov*100,base,acc,color=PUR,alpha=0.12,lw=0)
ax[0].plot(cov*100,acc,color=PUR,lw=2.4,zorder=3)
ax[0].axhline(base,color=GRAY,ls=(0,(4,3)),lw=1.1)
ax[0].scatter([50],[acc[i50]],s=55,color=ACC,zorder=5,edgecolor="white",lw=1.2)
ax[0].annotate(f"+{(acc[i50]-base)*100:.0f} pp",xy=(50,acc[i50]),xytext=(56,acc[i50]-0.07),
   color=ACC,fontsize=9,fontweight="bold",arrowprops=dict(arrowstyle="->",color=ACC,lw=1.2))
ax[0].text(99,base,f"no abstention\n{base:.2f}",fontsize=7,color="#6b7178",ha="right",va="bottom")
ax[0].set_xlabel("Coverage retained (%)"); ax[0].set_ylabel("Accepted-set accuracy")
ax[0].set_title("(a) Selective prediction",fontsize=9.5,fontweight="bold"); ax[0].set_xlim(10,100)

# (b) error-capture (triage) curve: review highest-misfit first
eb=(err>=0.5).astype(float); tot=eb.sum()
rev=np.arange(0,1.001,0.01)
rank=np.argsort(-mis)  # highest misfit first
cap=[eb[rank[:int(round(r*n))]].sum()/tot for r in rev]
ax[1].plot(rev*100,np.array(cap)*100,color=PUR,lw=2.4,zorder=3,label="misfit-guided")
ax[1].plot([0,100],[0,100],color=GRAY,ls=(0,(4,3)),lw=1.2,label="random review")
for rr in (0.2,0.3):
    j=int(round(rr*n)); c=eb[rank[:j]].sum()/tot
    ax[1].scatter([rr*100],[c*100],s=40,color=ACC,zorder=5,edgecolor="white",lw=1)
    ax[1].text(rr*100+1,c*100-4,f"{c*100:.0f}%",color=ACC,fontsize=8,fontweight="bold")
ax[1].set_xlabel("Subjects sent to expert review (%)"); ax[1].set_ylabel("Misclassifications caught (%)")
ax[1].set_title("(b) Triage efficiency",fontsize=9.5,fontweight="bold")
ax[1].legend(loc="lower right",fontsize=7.6,frameon=True); ax[1].set_xlim(0,100); ax[1].set_ylim(0,100)

# (c) reliability diagram: misfit decile vs empirical error
q=np.quantile(mis,np.linspace(0,1,11)); binid=np.clip(np.digitize(mis,q[1:-1]),0,9)
er=[err[binid==k].mean() for k in range(10)]
ax[2].bar(np.arange(10),er,color=plt.cm.plasma(np.linspace(0.1,0.9,10)),edgecolor="white",lw=0.6)
ax[2].axhline(err.mean(),color=GRAY,ls=(0,(4,3)),lw=1.1)
ax[2].text(0.2,err.mean()+0.01,f"cohort mean {err.mean():.2f}",fontsize=7,color="#6b7178")
ax[2].set_xlabel("Structural-misfit decile (low $\\to$ high)"); ax[2].set_ylabel("Empirical error rate")
ax[2].set_title("(c) Risk stratification",fontsize=9.5,fontweight="bold")
ax[2].set_xticks([0,3,6,9]); ax[2].set_xticklabels(["1","4","7","10"])
from scipy.stats import spearmanr
ax[2].text(0.5,max(er)*0.92,f"decile$\\to$error\n$\\rho$={spearmanr(np.arange(10),er)[0]:.2f}",fontsize=7.5,color=PUR)
for a in ax: sns.despine(ax=a)
fig.suptitle("OASIS-3: structural misfit as a clinical reliability tool (1,360 participants, 6 scanners)",
             fontsize=11,fontweight="bold",y=1.02)
fig.tight_layout(); fig.savefig(_resolve_path("<REPO_ROOT>/misfit/results/oasis3/fancy_figs/fig_clinical_utility.pdf"),bbox_inches="tight")
print("wrote fig_clinical_utility.pdf; triage: review 20% catches %.0f%% of errors"%(cap[20]*100))
