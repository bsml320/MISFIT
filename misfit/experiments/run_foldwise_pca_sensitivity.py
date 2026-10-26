"""Fold-wise PCA sensitivity (AT-S01 closure): does refitting the learned preprocessing
inside each source fold change the per-subject reliability ranking?

The paper's pipelines fit PCA (and the OASIS-3 DARTEL template) once per cohort before
cross-validation. This script compares, on ADNI (8 protocols) and leakage-free OASIS-3
(6 scanners), two configurations that are identical except for preprocessing scope:
  cohort   : voxelwise standardization + PCA-50 fit on ALL subjects (paper design)
  foldwise : voxelwise standardization + PCA-50 fit on the SOURCE-FOLD subjects only,
             held-out subjects projected with the source-fold transform
Score = plain-classifier final-epoch per-subject training CE (the plain-loss analogue of
the misfit; [64,32] MLP, Adam lr 1e-3, wd 1e-4, 100 epochs full batch), 3 seeds; graded
by pooled and within-environment Spearman rho and failure-detection AUROC. The brain
mask (mean GM > 0.1) is shared by both configurations. GPU 2.
Output: results/foldwise_pca_sensitivity.json
"""
from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

import json
import os
import re
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU: GPU2 is contended; SVDs cached per fold
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

sys.path.insert(0, _resolve_path("<REPO_ROOT>"))
from misfit.experiments.run_honest_estimator_table import PlainMLP  # noqa: E402

DEV = "cpu"
torch.set_num_threads(16)
SEEDS, SEED_BASE, EPOCHS, PCA_DIM = 3, 42, 100, 50
RES = _resolve_path("<REPO_ROOT>/misfit/results/foldwise_pca_sensitivity.json")


def load_adni_raw():
    """Load masked raw ADNI voxels, labels and protocol sites.

    Rebuilds the cohort with the same path/label/protocol filters as the
    Phase 1A ADNI experiment, reads the smwc1 volumes in parallel, and applies
    the shared brain mask (mean GM > 0.1).
    """
    import pandas as pd
    from misfit.data.adni_loader import (
        IMPAIRED_GROUPS, LABELS_CSV, MIN_SUBJECTS_PER_SITE, NII_DIR, PROTOCOLS)
    df = pd.read_csv(LABELS_CSV)
    lut = {}
    for _, r in df.iterrows():
        lut.setdefault((str(r["dataset_id"]).strip(), str(r["file_name"]).strip()),
                       str(r["Group"]).strip())
    paths, labs, sites = [], [], []
    si = 0
    for protocol in PROTOCOLS:
        pdir = NII_DIR / protocol
        if not pdir.exists():
            continue
        pp, pl = [], []
        for sdir in sorted(pdir.iterdir()):
            f = sdir / "smwc1pT1.nii"
            k = (protocol, sdir.name)
            if not f.exists() or k not in lut or lut[k] not in {"CN"} | IMPAIRED_GROUPS:
                continue
            pp.append(str(f))
            pl.append(0 if lut[k] == "CN" else 1)
        if len(pp) < MIN_SUBJECTS_PER_SITE:
            continue
        paths += pp
        labs += pl
        sites += [si] * len(pp)
        si += 1
    with ThreadPoolExecutor(max_workers=16) as ex:
        vols = list(ex.map(lambda p: nib.load(p).get_fdata(dtype=np.float32).ravel(), paths))
    X = np.stack(vols)
    mask = X.mean(0) > 0.1
    return X[:, mask], np.array(labs), np.array(sites)


def load_oasis3_raw():
    """Load masked raw OASIS-3 voxels, labels and scanner sites.

    Keeps the earliest session per participant (leakage-free convention),
    codes scanners by descending frequency, reads the route-B smwc1 volumes in
    parallel, and applies the shared brain mask (mean GM > 0.1).
    """
    d = np.load(_resolve_path("<REPO_ROOT>/data/oasis3/oasis3_features.npz"), allow_pickle=True)
    subj = d["subjects"].astype(str)
    sess = d["sessions"].astype(str)
    day = np.array([int(re.search(r"_d(\d+)$", s).group(1)) for s in sess])
    keep = np.array(sorted(np.where(subj == u)[0][np.argmin(day[subj == u])]
                           for u in np.unique(subj)))
    sess = sess[keep]
    lab = d["label"].astype(int)[keep]
    scan = d["scanner"].astype(str)[keep]
    uniq, cnt = np.unique(scan, return_counts=True)
    order = uniq[np.argsort(-cnt)]
    code = {s: i for i, s in enumerate(order)}
    site = np.array([code[x] for x in scan])
    W = _resolve_path("<REPO_ROOT>/data/oasis3/route_b_work")
    with ThreadPoolExecutor(max_workers=16) as ex:
        vols = list(ex.map(
            lambda s: nib.load(f"{W}/{s}/smwc1pT1.nii").get_fdata(dtype=np.float32).ravel(),
            sess))
    X = np.stack(vols)
    mask = X.mean(0) > 0.1
    return X[:, mask], lab, site


def pca_project(Xg, fit_rows, seed):
    """Standardize + PCA-50 fit on fit_rows; project all rows. Xg on GPU float32."""
    mu = Xg[fit_rows].mean(0)
    sd = Xg[fit_rows].std(0) + 1e-6
    Zfit = (Xg[fit_rows] - mu) / sd
    torch.manual_seed(seed)
    _, _, V = torch.svd_lowrank(Zfit, q=PCA_DIM, niter=4)
    return ((Xg - mu) / sd) @ V


def build_projections(Xg, site):
    """PCA is seed-independent: compute the cohort projection once and each
    fold-wise projection once, shared across seeds and configs."""
    n_sites = int(site.max()) + 1
    Z_cohort = pca_project(Xg, np.arange(Xg.shape[0]), 0)
    Z_fold = {}
    for held in range(n_sites):
        tr = np.where(site != held)[0]
        te = np.where(site == held)[0]
        if len(te) < 5 or len(tr) < 20:
            continue
        Z_fold[held] = pca_project(Xg, tr, held)
        print(f"    fold {held}: PCA done", flush=True)
    return Z_cohort, Z_fold


def run_config(Zs, y, site, foldwise, seed):
    """Score one preprocessing configuration for one seed.

    Leave-one-site-out loop: trains a PlainMLP per fold on the cohort or
    fold-wise projection, accumulates each training subject's final-epoch CE
    (averaged over the folds it appears in) as the reliability score, and
    records held-out predictions.

    Returns:
        Tuple (pooled rho, mean within-environment rho, failure AUROC).
    """
    Z_cohort, Z_fold = Zs
    n_sites = int(site.max()) + 1
    N = len(y)
    msum, mcnt = np.zeros(N), np.zeros(N)
    pred = np.full(N, -1)
    tested = np.zeros(N, bool)
    yt = torch.from_numpy(y).long().to(DEV)
    for held in range(n_sites):
        tr = np.where(site != held)[0]
        te = np.where(site == held)[0]
        if len(te) < 5 or len(tr) < 20:
            continue
        Z = Z_fold[held] if foldwise else Z_cohort
        torch.manual_seed(seed)
        m = PlainMLP(PCA_DIM, 2).to(DEV)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
        Ztr = Z[tr].to(DEV)
        m.train()
        for _ in range(EPOCHS):
            opt.zero_grad()
            F.cross_entropy(m(Ztr), yt[tr]).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            losses = F.cross_entropy(m(Ztr), yt[tr], reduction="none").cpu().numpy()
            pred[te] = m(Z[te].to(DEV)).argmax(1).cpu().numpy()
        msum[tr] += losses
        mcnt[tr] += 1
        tested[te] = True
    err = (pred != y).astype(float)
    sc = msum / np.maximum(mcnt, 1)
    m_ = tested
    pooled = float(spearmanr(sc[m_], err[m_])[0])
    auc = float(roc_auc_score(err[m_], sc[m_])) if 0 < err[m_].sum() < m_.sum() else float("nan")
    wr = []
    for e in np.unique(site[m_]):
        mm = m_ & (site == e)
        ee = err[mm]
        if len(ee) >= 8 and 0 < ee.sum() < len(ee):
            r = spearmanr(sc[mm], ee)[0]
            if not np.isnan(r):
                wr.append(r)
    return pooled, float(np.mean(wr)), auc


def main():
    """Run both cohorts and configurations; write the sensitivity JSON with deltas."""
    out = {}
    for name, loader in [("adni", load_adni_raw), ("oasis3", load_oasis3_raw)]:
        print(f"[{name}] loading raw voxels ...", flush=True)
        X, y, site = loader()
        print(f"  X={X.shape}", flush=True)
        Xg = torch.from_numpy(X)
        del X
        Zs = build_projections(Xg, site)
        del Xg
        res = {}
        for cfg in ("cohort", "foldwise"):
            pr, wr, au = [], [], []
            for k in range(SEEDS):
                p, w, a = run_config(Zs, y, site, cfg == "foldwise", SEED_BASE + k)
                pr.append(p)
                wr.append(w)
                au.append(a)
            res[cfg] = {"pooled_rho": float(np.mean(pr)), "pooled_rho_std": float(np.std(pr)),
                        "within_env_rho": float(np.mean(wr)), "within_env_rho_std": float(np.std(wr)),
                        "auroc": float(np.nanmean(au)), "auroc_std": float(np.nanstd(au))}
            print(f"  [{name}/{cfg}] pooled={res[cfg]['pooled_rho']:.3f} "
                  f"within-env={res[cfg]['within_env_rho']:.3f} AUROC={res[cfg]['auroc']:.3f}",
                  flush=True)
        res["delta"] = {k: round(res["foldwise"][k] - res["cohort"][k], 4)
                        for k in ("pooled_rho", "within_env_rho", "auroc")}
        out[name] = res
        del Zs
    json.dump(out, open(RES, "w"), indent=2)
    print("wrote", RES, flush=True)


if __name__ == "__main__":
    main()
