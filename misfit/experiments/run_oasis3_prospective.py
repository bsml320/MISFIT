"""Review item #1: a genuinely PROSPECTIVE held-scanner reliability test on OASIS-3.

For each held-out scanner (a 'new site'): train the source classifier only on the other five
scanners; split the held-out scanner into a small labeled PILOT (k=10/25/50) and an untouched
EVAL set. Score each eval subject's reliability using ONLY their image, the source model, and the
pilot -- never their own label or training history. Compare pilot-calibration and the SNT proxy
against the purely test-time baselines (max-softmax, energy). Eval labels are used only to grade
the score. Repeated pilot splits x seeds. GPU 2. Output: results/oasis3/oasis3_prospective.json.

Prespecified decision rule: if the best prospective rho is at or below 0.3, the clinical-triage
framing is not demonstrated and the method is a development-cohort audit.
"""

from misfit.paths import resolve_path as _resolve_path
import os, sys, json, re
# CUDA visibility is selected by the caller.
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neighbors import KNeighborsRegressor
DEV="cuda" if torch.cuda.is_available() else "cpu"
SEEDS, SPLITS, KS = 3, 20, [10,25,50]

d=np.load(_resolve_path("<REPO_ROOT>/data/oasis3/oasis3_features.npz"),allow_pickle=True)
feats,labels=d["features"],d["label"].astype(int); scan=d["scanner"].astype(str); subj=d["subjects"].astype(str); sess=d["sessions"].astype(str)
day=np.array([int(re.search(r"_d(\d+)$",s).group(1)) for s in sess])
# one earliest session per participant (leakage-free cohort)
keep=np.array(sorted(np.where(subj==u)[0][np.argmin(day[subj==u])] for u in np.unique(subj)))
X,y,sc=feats[keep],labels[keep],scan[keep]
uniq,cnt=np.unique(sc,return_counts=True); order=uniq[np.argsort(-cnt)]; code={s:i for i,s in enumerate(order)}
sid=np.array([code[s] for s in sc]); dim=X.shape[1]

class Net(nn.Module):
    """[64,32] MLP returning (logits, penultimate representation)."""
    def __init__(s,d):
        super().__init__(); s.f1=nn.Linear(d,64); s.f2=nn.Linear(64,32); s.h=nn.Linear(32,2)
    def forward(s,x):
        r=F.relu(s.f2(F.relu(s.f1(x)))); return s.h(r), r

def train(Xtr,ytr,seed):
    """Train Net for 100 full-batch epochs (Adam, wd 1e-4) at the given seed."""
    torch.manual_seed(seed); m=Net(dim).to(DEV)
    opt=torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-4)
    xt=torch.tensor(Xtr).to(DEV); yt=torch.tensor(ytr).long().to(DEV)
    for _ in range(100):
        opt.zero_grad(); F.cross_entropy(m(xt)[0],yt).backward(); opt.step()
    return m

def scored_scanners():
    """Scanners with at least 40 subjects and both classes present."""
    return [i for i in np.unique(sid) if (sid==i).sum()>=40 and 0<y[sid==i].sum()<(sid==i).sum()]

def auc(e,s):
    return roc_auc_score(e,s) if 0<e.sum()<len(e) else np.nan
def ap(e,s):
    return average_precision_score(e,s) if 0<e.sum()<len(e) else np.nan
def rho(e,s):
    r=spearmanr(s,e)[0]; return r if not np.isnan(r) else np.nan

METH_PROSPECTIVE=["max_softmax","energy","snt"]              # no pilot
METH_PILOT=["pilot_knn"]                                     # uses pilot labels
res={m:{"auroc":[],"ap":[],"rho":[]} for m in METH_PROSPECTIVE}
res.update({f"{m}_k{k}":{"auroc":[],"ap":[],"rho":[]} for m in METH_PILOT for k in KS})
rc={f"pilot_knn_k{k}":[] for k in KS}; rc["max_softmax"]=[]
rng=np.random.RandomState(0)

for e in scored_scanners():
    tr=sid!=e; te=sid==e
    for seed in range(SEEDS):
        m=train(X[tr],y[tr],42+seed); m.eval()
        with torch.no_grad():
            ltr,rtr=m(torch.tensor(X[tr]).to(DEV)); ltr=ltr.cpu().numpy(); rtr=rtr.cpu().numpy()
            lte,rte=m(torch.tensor(X[te]).to(DEV)); lte=lte.cpu().numpy(); rte=rte.cpu().numpy()
        # training misfit for SNT
        mis_tr=F.cross_entropy(torch.tensor(ltr),torch.tensor(y[tr]).long(),reduction="none").numpy()
        pred_te=lte.argmax(1); err_te=(pred_te!=y[te]).astype(int)
        sm=np.exp(lte-lte.max(1,keepdims=True)); sm/=sm.sum(1,keepdims=True)
        msp=1-sm.max(1); energy=-(np.log(np.exp(lte).sum(1)+1e-9))
        # SNT: kNN(10) transfer of training misfit in rep space
        from sklearn.neighbors import NearestNeighbors
        nn10=NearestNeighbors(n_neighbors=10).fit(rtr); _,idx=nn10.kneighbors(rte); snt=mis_tr[idx].mean(1)
        # prospective baselines evaluated on the whole held-out scanner
        for nm,sc_ in [("max_softmax",msp),("energy",energy),("snt",snt)]:
            res[nm]["auroc"].append(auc(err_te,sc_)); res[nm]["ap"].append(ap(err_te,sc_)); res[nm]["rho"].append(rho(err_te,sc_))
        rc["max_softmax"].append(1-err_te[np.argsort(msp)[:max(1,len(err_te)//2)]].mean())
        # pilot-calibrated kNN(rep->error), repeated splits
        n=te.sum()
        for k in KS:
            if k>=n-8: continue
            for _ in range(SPLITS):
                perm=rng.permutation(n); pil=perm[:k]; ev=perm[k:]
                perr=(pred_te[pil]!=y[te][pil]).astype(int)
                if perr.sum()==0 or perr.sum()==len(perr):  # need both classes in pilot
                    pc=np.full(len(ev),perr.mean())
                else:
                    kr=KNeighborsRegressor(n_neighbors=min(5,k)).fit(rte[pil],perr); pc=kr.predict(rte[ev])
                ee=err_te[ev]
                res[f"pilot_knn_k{k}"]["auroc"].append(auc(ee,pc)); res[f"pilot_knn_k{k}"]["ap"].append(ap(ee,pc)); res[f"pilot_knn_k{k}"]["rho"].append(rho(ee,pc))
                rc[f"pilot_knn_k{k}"].append(1-ee[np.argsort(pc)[:max(1,len(ee)//2)]].mean())

out={}
for m,v in res.items():
    out[m]={s:[round(float(np.nanmean(v[s])),3),round(float(np.nanstd(v[s])),3)] for s in ("auroc","ap","rho")}
out["risk_coverage_acc50"]={m:round(float(np.nanmean(rc[m])),3) for m in rc}
# disabled (guarded by `if False`): always None; key kept for output-schema stability
out["base_error_meanacc"]=round(float(1-np.mean([ (lambda te=(sid==e): (train(X[sid!=e],y[sid!=e],42)(torch.tensor(X[te]).to(DEV))[0].argmax(1).cpu().numpy()!=y[te]).mean())() for e in scored_scanners()])),3) if False else None
json.dump(out,open(_resolve_path("<REPO_ROOT>/misfit/results/oasis3/oasis3_prospective.json"),"w"),indent=2)
print("=== PROSPECTIVE held-scanner (eval never uses own label/history) ===")
for m in ["max_softmax","energy","snt"]+[f"pilot_knn_k{k}" for k in KS]:
    a=out[m]["auroc"]; r=out[m]["rho"]; p=out[m]["ap"]
    print(f"  {m:16s} AUROC {a[0]:.3f}±{a[1]:.3f}  AUPRC {p[0]:.3f}  rho {r[0]:.3f}±{r[1]:.3f}  acc@50%={out['risk_coverage_acc50'].get(m,'--')}")
best=max([out[m]['rho'][0] for m in out if isinstance(out[m],dict) and 'rho' in out[m]])
print(f"\nBEST prospective rho = {best:.3f}  -> {'clinical-triage NOT demonstrated (<=0.3); development-audit framing' if best<=0.3 else 'prospective signal present (>0.3)'}")
print("wrote oasis3_prospective.json")
