"""Environment-level inference for OASIS-3 (review item #4): treat the SCANNER, not the subject or
seed, as the unit of generalization. Reports (a) a cluster bootstrap over scanners for the
pooled within-scanner rho / AUROC (uncertainty over NEW scanners, which seed/subject bootstraps
miss), and (b) scanner-wise PAIRED comparison of structural misfit vs the training-dynamics
competitors (AUM, EL2N) computed from the SAME classifier training. Leakage-free (one session
per participant). GPU 2. Output: results/oasis3/oasis3_env_inference.json.
"""

from misfit.paths import resolve_path as _resolve_path
import os, sys, json, re
# CUDA visibility is selected by the caller.
import numpy as np, torch
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score
sys.path.insert(0,_resolve_path("<REPO_ROOT>"))
from misfit.experiments.run_training_dynamics import collect_dynamics, SCORES

d=np.load(_resolve_path("<REPO_ROOT>/data/oasis3/oasis3_features.npz"),allow_pickle=True)
feats,labels=d["features"],d["label"].astype(int); scan=d["scanner"].astype(str); subj=d["subjects"].astype(str); sess=d["sessions"].astype(str)
day=np.array([int(re.search(r"_d(\d+)$",s).group(1)) for s in sess])
# one earliest session per participant (leakage-free cohort)
keep=np.array(sorted(np.where(subj==u)[0][np.argmin(day[subj==u])] for u in np.unique(subj)))
feats,labels,scan=feats[keep],labels[keep],scan[keep]
uniq,cnt=np.unique(scan,return_counts=True); order=uniq[np.argsort(-cnt)]
code={s:i for i,s in enumerate(order)}; site=torch.tensor([code[s] for s in scan])
f=torch.tensor(feats,dtype=torch.float32); t=torch.tensor(labels)
sid=site.numpy()

# seed-averaged per-subject scores (misfit + AUM + forgetting + EL2N from one training) + err
N=len(f); acc={k:np.zeros(N) for k in SCORES}; errs=[]; tested=None
for seed in range(5):
    scores,err,m=collect_dynamics(f,t,site,42+seed)
    for k in SCORES: acc[k]+=scores[k]
    errs.append(err); tested=m if tested is None else (tested&m)  # tested = intersection over seeds
for k in SCORES: acc[k]/=5.0
err=np.mean(errs,0)

def wsrho(score,e,mask):  # within-scanner mean rho over scored scanners + per-scanner
    per={}
    for i in np.unique(sid):
        mm=mask&(sid==i); ee=e[mm]
        if len(ee)>=8 and 0<((ee>=0.5).sum())<len(ee):
            r=spearmanr(score[mm],ee)[0]
            if not np.isnan(r): per[int(i)]=r
    return per

per={k:wsrho(acc[k],err,tested) for k in SCORES}
scanners_scored=sorted(set.intersection(*[set(per[k]) for k in SCORES]))
print("scored scanners:",[order[i] for i in scanners_scored],flush=True)

# (a) cluster bootstrap over scanners for pooled within-scanner rho (misfit)
rng=np.random.RandomState(0)
def clusboot(scoredict):
    """Mean over scanners, scanner-cluster bootstrap 95% CI, between-scanner SD."""
    vals=np.array([scoredict[i] for i in scanners_scored]); B=5000
    means=[vals[rng.randint(0,len(vals),len(vals))].mean() for _ in range(B)]
    return float(vals.mean()), [float(np.percentile(means,2.5)),float(np.percentile(means,97.5))], float(vals.std())
out={"scanners":[str(order[i]) for i in scanners_scored]}
for k in SCORES:
    m,ci,sd=clusboot(per[k])
    out[k]={"per_scanner":{str(order[i]):round(per[k][i],3) for i in scanners_scored},
            "mean_over_scanners":round(m,3),"cluster_bootstrap_ci":[round(x,3) for x in ci],
            "between_scanner_sd":round(sd,3)}
    print(f"  {k:11s} mean={m:.3f}  scanner-cluster 95%CI=[{ci[0]:.3f},{ci[1]:.3f}]  between-scanner SD={sd:.3f}",flush=True)

# (b) scanner-wise paired misfit vs AUM / EL2N (sign + Wilcoxon over scanners)
mis=np.array([per["misfit"][i] for i in scanners_scored])
out["paired_vs_misfit"]={}
for k in ("aum_neg","el2n","forgetting"):
    other=np.array([per[k][i] for i in scanners_scored]); diff=mis-other
    try: w=wilcoxon(mis,other).pvalue
    except Exception: w=float("nan")
    out["paired_vs_misfit"][k]={"misfit_minus_"+k+"_per_scanner":[round(x,3) for x in diff],
        "misfit_higher_on":f"{int((diff>0).sum())}/{len(diff)}","mean_diff":round(float(diff.mean()),3),
        "wilcoxon_p":round(float(w),3)}
    print(f"  misfit vs {k}: misfit higher on {int((diff>0).sum())}/{len(diff)} scanners, "
          f"mean diff {diff.mean():+.3f}, Wilcoxon p={w:.3f}",flush=True)
json.dump(out,open(_resolve_path("<REPO_ROOT>/misfit/results/oasis3/oasis3_env_inference.json"),"w"),indent=2)
print("wrote oasis3_env_inference.json")
