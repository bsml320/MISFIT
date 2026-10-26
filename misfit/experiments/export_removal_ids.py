"""Item 6: regenerate selection scores under the same seeds and persist drop-IDs,
retained class composition, and score arrays per fold/method/budget.

Rebuilds the source_scores() selection scores from run_intervention_audit for
ADNI and OASIS-3 under the committed seeding, then for each fold, method and
budget records which source subjects would be dropped (top scores first) and
the class composition of the retained set.

Outputs under results/provenance/: intervention_removal_ids_{adni,oasis3}.json,
intervention_selection_scores_{adni,oasis3}.npz (float64 score vectors plus
fold source-index arrays), and intervention_removal_ids_README.json binding
the export to the committed run and git revision. Run as a plain script.
"""

from misfit.paths import resolve_path as _resolve_path
import os, sys, json
# CUDA visibility is selected by the caller.
import numpy as np
sys.path.insert(0,_resolve_path("<REPO_ROOT>"))
sys.path.insert(0,_resolve_path("<REPO_ROOT>/misfit/experiments"))
from run_intervention_audit import source_scores, load_oasis3, BUDGETS
from misfit.experiments.run_honest_estimator_table import load_adni
out={"note":"scores regenerated under the identical seeds as the committed run "
     "(fits torch.manual_seed-ed; random via crc32(fold_tag)); drop IDs are indices "
     "into the cohort loader order; GPU nondeterminism could in principle permute "
     "near-tied ranks, flagged here rather than hidden"}
import torch
for name, loader in [("adni", load_adni), ("oasis3", load_oasis3)]:
    X,y,site=loader(); sn=site.numpy(); res={}
    arrs={}
    for held in np.unique(sn):
        te=np.where(sn==held)[0]; src=np.where(sn!=held)[0]
        if len(te)<5 or len(src)<20: continue
        S=source_scores(X,y,site,src,f"{name}{held}")
        arrs[f"fold{int(held)}_src_ids"]=src.astype(np.int64)
        for meth,sc in S.items():
            arrs[f"fold{int(held)}_{meth}"]=np.asarray(sc,dtype=np.float64)
        fold={}
        for meth,sc in S.items():
            ent={}
            for q in BUDGETS:
                k=int(round(q*len(src)))
                # Budget q drops the k highest-scoring source subjects.
                drop=src[np.argsort(-sc)[:k]]
                keepmask=np.ones(len(src),bool); keepmask[np.argsort(-sc)[:k]]=False
                keep=src[keepmask]
                ent[f"q{int(q*100)}"]={"n_dropped":int(k),
                  "dropped_ids":drop.tolist(),
                  "dropped_impaired":int(y.numpy()[drop].sum()),
                  "retained_n":int(len(keep)),
                  "retained_impaired":int(y.numpy()[keep].sum())}
            fold[meth]=ent
        res[f"fold{int(held)}"]=fold
        print(f"[{name}] fold {held} exported",flush=True)
    json.dump(res,open(_resolve_path(f"<REPO_ROOT>/misfit/results/provenance/intervention_removal_ids_{name}.json"),"w"))
    np.savez(_resolve_path(f"<REPO_ROOT>/misfit/results/provenance/intervention_selection_scores_{name}.npz"),**arrs)
import subprocess
out["full_precision_scores"]=("intervention_selection_scores_{adni,oasis3}.npz: float64 "
    "selection-score vector per fold x method over the fold's source subjects "
    "(fold*_src_ids gives their cohort-loader indices); argsort(-score)[:k] reproduces "
    "the dropped_ids lists exactly")
out["binds_to"]=("results/intervention_audit.json (the committed v2 run; its per-cell "
    "metric arrays are in that file) and the run log _interv2.log; prespec "
    "paper/feedback/PRESPEC_intervention_20261003.md @ 1200c2b")
out["producing_revision_misfit"]=subprocess.run(
    ["git","rev-parse","HEAD"],capture_output=True,text=True,
    cwd=_resolve_path("<REPO_ROOT>/misfit")).stdout.strip()
json.dump(out,open(_resolve_path("<REPO_ROOT>/misfit/results/provenance/intervention_removal_ids_README.json"),"w"),indent=2)
print("done")
