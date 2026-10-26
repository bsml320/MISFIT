"""Dump canonical (paper-headline) per-subject IOP misfit, 5-seed averaged, ADNI+OASIS-1.

For each cohort loader, runs the validated collect() harness over N_SEEDS seeds
starting at SEED_BASE, averages misfit and error fraction per subject, and saves
misfit, error, label and tested-mask arrays to
misfit/results/per_subject_misfit_iop.npz. Run as a plain script.
"""

from misfit.paths import resolve_path as _resolve_path
import os, sys
# CUDA visibility is selected by the caller.
import numpy as np
sys.path.insert(0,_resolve_path("<REPO_ROOT>"))
from misfit.experiments.run_honest_estimator_table import (N_SEEDS, SEED_BASE, load_adni, load_oasis_ds)
from misfit.experiments.run_clinical_stratified_media import collect
out={}
for name,loader in [("adni",load_adni),("oasis",load_oasis_ds)]:
    f,t,s=loader()
    mis=np.zeros(len(f)); err=[]
    for k in range(N_SEEDS):
        c=collect(f,t,s,SEED_BASE+k); mis+=c["misfit"]; err.append(c["err"])
        tested=c["tested"]  # identical across seeds; the last copy is saved
    mis/=N_SEEDS
    out[f"{name}_misfit"]=mis; out[f"{name}_err"]=np.mean(err,0)
    out[f"{name}_label"]=t.numpy().astype(int); out[f"{name}_tested"]=tested
    cn=mis[(t.numpy()==0)]; imp=mis[(t.numpy()==1)]
    print(f"[{name}] mean CN {cn.mean():.3f} vs impaired {imp.mean():.3f}",flush=True)
np.savez(_resolve_path("<REPO_ROOT>/misfit/results/per_subject_misfit_iop.npz"),**out)
print("saved per_subject_misfit_iop.npz")
