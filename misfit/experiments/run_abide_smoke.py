"""Smoke test of the misfit pipeline on ABIDE-I (single seed).

Loads ABIDE-I rs-fMRI features via misfit.data.abide_loader, runs the
leave-one-site-out collect() routine from run_clinical_stratified_media
for seed 42, and scores per-subject misfit against observed
misclassification (failure-detection AUROC and Spearman rho) on tested
subjects. Writes a plain-text summary to
misfit/results/abide_smoke.txt and echoes it to stdout.
Run as a plain script with no arguments.
"""

from misfit.paths import resolve_path as _resolve_path
import sys, numpy as np
sys.path.insert(0, _resolve_path("<REPO_ROOT>"))
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from misfit.data.abide_loader import load_abide
from misfit.experiments.run_clinical_stratified_media import collect

d = load_abide()
f, t, s = d.features, d.targets, d.site_ids
out = [f"ABIDE-I: N={len(f)} dim={f.shape[1]} sites={int(s.max())+1} (autism vs control, rs-fMRI cc200), balance={float(t.float().mean()):.3f}"]
aur, rho = [], []
for seed in [42]:
    c = collect(f, t, s, seed)
    m = c["tested"]; mis = c["misfit"][m]; err = c["err"][m]
    if len(set(err.astype(int))) > 1:
        aur.append(roc_auc_score(err, mis)); rho.append(spearmanr(mis, err)[0])
        out.append(f"  seed {seed}: base_err={err.mean():.3f}  failure-AUROC={aur[-1]:.3f}  rho={rho[-1]:.3f}")
out.append(f"MEAN failure-detection AUROC={np.mean(aur):.3f}+-{np.std(aur):.3f}  rho={np.mean(rho):.3f} (leave-one-site-out, 20 sites)")
open("misfit/results/abide_smoke.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
