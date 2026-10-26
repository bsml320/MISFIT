"""Experiment 1(a): FULLY source-only feature pipeline (mask + standardization + PCA).

Extends the fold-wise PCA sensitivity: here the brain MASK is also refit on source-fold
subjects only (mean GM > 0.1 over source subjects), so every learned feature-pipeline
stage downstream of spatial normalization is source-only. (The spatial-normalization
template axis is handled separately by the independent-template arm, Experiment 1b.)

Arms per cohort (ADNI raw smwc1 voxels; leakage-free OASIS-3 raw smwc1 voxels):
  cohort            : cohort mask + cohort std+PCA (paper design; recomputed here for
                      an identical-code comparison)
  foldwise_full     : SOURCE-FOLD mask + source-fold std + source-fold PCA-50;
                      held-out subjects projected with the source-fold transform
Score = plain-classifier final-epoch per-subject CE ([64,32] MLP, 100 ep, 3 seeds);
graded by pooled/within-environment Spearman rho and AUROC vs held-out binary error.
CPU (fold-cached SVDs). Output: results/sourceonly_features.json
"""
from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

import json
import os
import re
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

sys.path.insert(0, _resolve_path("<REPO_ROOT>"))
from misfit.experiments.run_honest_estimator_table import PlainMLP  # noqa: E402
from misfit.experiments.run_foldwise_pca_sensitivity import (  # noqa: E402
    load_adni_raw, load_oasis3_raw)

DEV = "cpu"
torch.set_num_threads(16)
SEEDS, EPOCHS, PCA_DIM = 3, 100, 50
OUT = _resolve_path("<REPO_ROOT>/misfit/results/sourceonly_features.json")


def project(Xfull, rows_fit, mask, seed):
    """mask (bool over voxels) -> standardize + PCA-50 fit on rows_fit; project all."""
    Xm = torch.from_numpy(Xfull[:, mask])
    mu = Xm[rows_fit].mean(0)
    sd = Xm[rows_fit].std(0) + 1e-6
    torch.manual_seed(seed)
    _, _, V = torch.svd_lowrank((Xm[rows_fit] - mu) / sd, q=PCA_DIM, niter=4)
    return ((Xm - mu) / sd) @ V


def run(Xraw, y, site, arm):
    """Evaluate one feature-pipeline arm ("cohort" or "foldwise_full") on one cohort.

    Fold projections are precomputed once (the per-fold SVD seed is fixed, so
    features do not vary with the classifier seed); the SEEDS classifier seeds
    then share them. Score = final-epoch per-subject training CE averaged over
    in-train folds, graded pooled and within environment against held-out
    binary error.

    Returns:
        {"pooled", "within", "auroc"} -> {"mean", "std"} over seeds.
    """
    n_env = int(site.max()) + 1
    N = len(y)
    yt = torch.from_numpy(y).long()
    res = {"pooled": [], "within": [], "auroc": []}
    # precompute fold projections once (seed-independent up to svd seed, fixed per fold)
    projs = {}
    cohort_mask = Xraw.mean(0) > 0.1
    for held in range(n_env):
        tr = np.where(site != held)[0]
        if (site == held).sum() < 5 or len(tr) < 20:
            continue
        if arm == "cohort":
            projs[held] = project(Xraw, np.arange(N), cohort_mask, 0)
        else:
            m = Xraw[tr].mean(0) > 0.1
            projs[held] = project(Xraw, tr, m, held)
        print(f"    [{arm}] fold {held}: features ready (voxels="
              f"{int(cohort_mask.sum()) if arm=='cohort' else int((Xraw[tr].mean(0)>0.1).sum())})",
              flush=True)
    for s in range(SEEDS):
        msum, mcnt = np.zeros(N), np.zeros(N)
        pred = np.full(N, -1)
        tested = np.zeros(N, bool)
        for held, Z in projs.items():
            tr = np.where(site != held)[0]
            te = np.where(site == held)[0]
            torch.manual_seed(s)
            mdl = PlainMLP(PCA_DIM, 2)
            opt = torch.optim.Adam(mdl.parameters(), lr=1e-3, weight_decay=1e-4)
            Ztr = Z[tr]
            mdl.train()
            for _ in range(EPOCHS):
                opt.zero_grad()
                F.cross_entropy(mdl(Ztr), yt[tr]).backward()
                opt.step()
            mdl.eval()
            with torch.no_grad():
                losses = F.cross_entropy(mdl(Ztr), yt[tr], reduction="none").numpy()
                pred[te] = mdl(Z[te]).argmax(1).numpy()
            msum[tr] += losses
            mcnt[tr] += 1
            tested[te] = True
        err = (pred != y).astype(float)
        sc = msum / np.maximum(mcnt, 1)
        m_ = tested
        res["pooled"].append(float(spearmanr(sc[m_], err[m_])[0]))
        res["auroc"].append(float(roc_auc_score(err[m_], sc[m_]))
                            if 0 < err[m_].sum() < m_.sum() else float("nan"))
        wr = []
        for e in np.unique(site[m_]):
            mm = m_ & (site == e)
            ee = err[mm]
            if len(ee) >= 8 and 0 < ee.sum() < len(ee):
                r = spearmanr(sc[mm], ee)[0]
                if np.isfinite(r):
                    wr.append(r)
        res["within"].append(float(np.mean(wr)))
        print(f"    [{arm}] seed {s}: within={res['within'][-1]:.3f}", flush=True)
    return {k: {"mean": round(float(np.mean(v)), 3), "std": round(float(np.std(v)), 3)}
            for k, v in res.items()}


def main():
    """Run both arms on ADNI and leakage-free OASIS-3 raw voxels; write JSON."""
    out = {}
    for name, loader in [("adni", load_adni_raw), ("oasis3", load_oasis3_raw)]:
        print(f"[{name}] loading raw voxels ...", flush=True)
        X, y, site = loader()
        print(f"  X={X.shape}", flush=True)
        out[name] = {}
        for arm in ("cohort", "foldwise_full"):
            out[name][arm] = run(X, y, site, arm)
            r = out[name][arm]
            print(f"  [{name}/{arm}] within={r['within']['mean']}±{r['within']['std']} "
                  f"AUROC={r['auroc']['mean']}", flush=True)
        del X
    json.dump(out, open(OUT, "w"), indent=2)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
