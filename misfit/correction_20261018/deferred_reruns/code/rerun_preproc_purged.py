"""R3: corrected ADNI arms of the plain-MLP preprocessing sensitivities.

R3a mirrors misfit/experiments/run_foldwise_pca_sensitivity.py (arms
`cohort` vs `foldwise`, shared cohort brain mask, svd_lowrank(q=50, niter=4)
with the historical per-fold seeding, seeds 42-44).
R3b mirrors misfit/experiments/run_sourceonly_features.py (arms `cohort`
vs `foldwise_full` with source-fold mask + standardization + PCA, seeds 0-2).

Plan delta for both (RERUN_PLAN.md Section 4): the fold structure is
participant-purged (purge_masks) in BOTH arms, so every source fold and every
MLP training pool excludes the held-out protocol's participants entirely; the
preprocessing SCOPE remains the manipulated variable (`cohort` arms keep the
all-acquisition preprocessing fit, which is the contrast under test). All
hyperparameters, metrics (pooled / within-protocol Spearman, AUROC) and SD
conventions (np.std ddof=0) unchanged. CPU, as the historical scripts.

Cohort binding: the raw-voxel loader records participant IDs and asserts
(protocols, labels, participants) equal to the verified correction cache.

Outputs: deferred_reruns/results/foldwise_pca_purged.json
         deferred_reruns/results/sourceonly_purged.json
"""
from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU, as the historical scripts
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rerun_common import load, purge_masks  # noqa: E402

sys.path.insert(0, _resolve_path("<REPO_ROOT>"))
from misfit.experiments.run_honest_estimator_table import PlainMLP  # noqa: E402

torch.set_num_threads(16)
PCA_DIM, EPOCHS = 50, 100
SEEDS_A, SEED_BASE_A = 3, 42   # R3a (historical run_foldwise_pca_sensitivity)
SEEDS_B = 3                    # R3b seeds 0-2 (historical run_sourceonly_features)
RESDIR = Path(__file__).resolve().parents[1] / "results"


def load_adni_raw_with_ptids():
    """Historical load_adni_raw + participant IDs, bound to the cache cohort."""
    import pandas as pd
    from misfit.data.adni_loader import (
        IMPAIRED_GROUPS, LABELS_CSV, MIN_SUBJECTS_PER_SITE, NII_DIR, PROTOCOLS)
    df = pd.read_csv(LABELS_CSV)
    lut = {}
    for _, r in df.iterrows():
        lut.setdefault((str(r["dataset_id"]).strip(), str(r["file_name"]).strip()),
                       str(r["Group"]).strip())
    paths, labs, sites, ptids = [], [], [], []
    si = 0
    for protocol in PROTOCOLS:
        pdir = NII_DIR / protocol
        if not pdir.exists():
            continue
        pp, pl, pt = [], [], []
        for sdir in sorted(pdir.iterdir()):
            f = sdir / "smwc1pT1.nii"
            k = (protocol, sdir.name)
            if not f.exists() or k not in lut or lut[k] not in {"CN"} | IMPAIRED_GROUPS:
                continue
            pp.append(str(f))
            pl.append(0 if lut[k] == "CN" else 1)
            pt.append(sdir.name)
        if len(pp) < MIN_SUBJECTS_PER_SITE:
            continue
        paths += pp
        labs += pl
        ptids += pt
        sites += [si] * len(pp)
        si += 1
    with ThreadPoolExecutor(max_workers=16) as ex:
        vols = list(ex.map(lambda p: nib.load(p).get_fdata(dtype=np.float32).ravel(), paths))
    X = np.stack(vols)
    y = np.array(labs)
    site = np.array(sites)
    ptids = np.array(ptids)
    cache = load()
    assert np.array_equal(ptids, cache["ptids"]), "participant order != cache"
    assert np.array_equal(site, cache["site_ids"].numpy()), "protocols != cache"
    assert np.array_equal(y, cache["targets"].numpy().astype(int)), "labels != cache"
    return X, y, site, ptids


def purged_folds(site, ptids):
    """Purged (train_idx, test_idx) pairs per eligible held-out protocol."""
    site_t = torch.tensor(site)
    folds = {}
    for held in range(int(site.max()) + 1):
        tr_m, te_m = purge_masks(site_t, ptids, held)
        if te_m.sum() < 5 or tr_m.sum() < 20:
            continue
        folds[held] = (np.where(tr_m.numpy())[0], np.where(te_m.numpy())[0])
    return folds


# ---------------- R3a: fold-wise PCA scope (shared cohort mask) ----------------

def pca_project(Xg, fit_rows, seed):
    """Historical pca_project: standardize + svd_lowrank-50 fit on fit_rows."""
    mu = Xg[fit_rows].mean(0)
    sd = Xg[fit_rows].std(0) + 1e-6
    zfit = (Xg[fit_rows] - mu) / sd
    torch.manual_seed(seed)
    _, _, v = torch.svd_lowrank(zfit, q=PCA_DIM, niter=4)
    return ((Xg - mu) / sd) @ v


def grade_run(sc, err, site, tested):
    """Pooled Spearman, mean within-protocol Spearman (cells with n >= 8 and
    both outcomes), and pooled AUROC over tested scans."""
    m_ = tested
    pooled = float(spearmanr(sc[m_], err[m_])[0])
    auc = float(roc_auc_score(err[m_], sc[m_])) if 0 < err[m_].sum() < m_.sum() else float("nan")
    wr = []
    for e in np.unique(site[m_]):
        mm = m_ & (site == e)
        ee = err[mm]
        if len(ee) >= 8 and 0 < ee.sum() < len(ee):
            r = spearmanr(sc[mm], ee)[0]
            if np.isfinite(r):
                wr.append(r)
    return pooled, float(np.mean(wr)), auc


def mlp_pass(Z, y, folds, seed):
    """One LOSO pass: train PlainMLP per fold, average per-scan training CE
    over contributing folds, collect held-out predictions, and grade.

    Z is either a per-fold dict (fold-wise arms) or one shared matrix.
    Grading uses the module-level SITE array set in main().
    """
    n = len(y)
    msum, mcnt = np.zeros(n), np.zeros(n)
    pred = np.full(n, -1)
    tested = np.zeros(n, bool)
    yt = torch.from_numpy(y).long()
    for held, (tr, te) in sorted(folds.items()):
        z = Z[held] if isinstance(Z, dict) else Z
        torch.manual_seed(seed)
        m = PlainMLP(PCA_DIM, 2)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
        ztr = z[tr]
        m.train()
        for _ in range(EPOCHS):
            opt.zero_grad()
            F.cross_entropy(m(ztr), yt[tr]).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            losses = F.cross_entropy(m(ztr), yt[tr], reduction="none").numpy()
            pred[te] = m(z[te]).argmax(1).numpy()
        msum[tr] += losses
        mcnt[tr] += 1
        tested[te] = True
    err = (pred != y).astype(float)
    return grade_run(msum / np.maximum(mcnt, 1), err, SITE, tested)


def agg(rows):
    pr, wr, au = zip(*rows)
    return {"pooled_rho": float(np.mean(pr)), "pooled_rho_std": float(np.std(pr)),
            "within_env_rho": float(np.mean(wr)), "within_env_rho_std": float(np.std(wr)),
            "auroc": float(np.nanmean(au)), "auroc_std": float(np.nanstd(au))}


def r3a(X, y, folds):
    """R3a: cohort-fit vs fold-fit PCA scope on the shared cohort brain mask."""
    mask = X.mean(0) > 0.1
    xg = torch.from_numpy(X[:, mask])
    z_cohort = pca_project(xg, np.arange(len(y)), 0)
    z_fold = {}
    for held, (tr, _) in sorted(folds.items()):
        z_fold[held] = pca_project(xg, tr, held)
        print(f"    [r3a] fold {held}: PCA done (purged n_fit={len(tr)})", flush=True)
    del xg
    res = {}
    for cfg, Z in (("cohort", z_cohort), ("foldwise", z_fold)):
        rows = [mlp_pass(Z, y, folds, SEED_BASE_A + k) for k in range(SEEDS_A)]
        res[cfg] = agg(rows)
        print(f"  [r3a/{cfg}] pooled={res[cfg]['pooled_rho']:.3f} "
              f"within={res[cfg]['within_env_rho']:.3f} AUROC={res[cfg]['auroc']:.3f}",
              flush=True)
    res["delta"] = {k: round(res["foldwise"][k] - res["cohort"][k], 4)
                    for k in ("pooled_rho", "within_env_rho", "auroc")}
    return res


# ------------- R3b: fully source-only pipeline (mask refit per fold) -------------

def project_masked(Xfull, rows_fit, mask, seed):
    """Historical sourceonly project(): mask -> standardize + PCA-50 on rows_fit."""
    xm = torch.from_numpy(Xfull[:, mask])
    mu = xm[rows_fit].mean(0)
    sd = xm[rows_fit].std(0) + 1e-6
    torch.manual_seed(seed)
    _, _, v = torch.svd_lowrank((xm[rows_fit] - mu) / sd, q=PCA_DIM, niter=4)
    return ((xm - mu) / sd) @ v


def r3b(X, y, folds):
    """R3b: cohort pipeline vs fully source-only pipeline (brain mask refit per fold)."""
    cohort_mask = X.mean(0) > 0.1
    res = {}
    for arm in ("cohort", "foldwise_full"):
        projs = {}
        z_cohort = None
        for held, (tr, _) in sorted(folds.items()):
            if arm == "cohort":
                if z_cohort is None:
                    z_cohort = project_masked(X, np.arange(len(y)), cohort_mask, 0)
                projs[held] = z_cohort
                nvox = int(cohort_mask.sum())
            else:
                # Pinned voxel universe (engineering review finding 1): the
                # historical script received an already cohort-masked matrix,
                # so its fold mask was implicitly intersected with the cohort
                # mask. Reproduce that intersection explicitly.
                m = cohort_mask & (X[tr].mean(0) > 0.1)
                projs[held] = project_masked(X, tr, m, held)
                nvox = int(m.sum())
            print(f"    [r3b/{arm}] fold {held}: features ready (voxels={nvox})", flush=True)
        rows = [mlp_pass(projs, y, folds, s) for s in range(SEEDS_B)]
        pr, wr, au = zip(*rows)
        res[arm] = {"pooled": {"mean": round(float(np.mean(pr)), 3),
                               "std": round(float(np.std(pr)), 3)},
                    "within": {"mean": round(float(np.mean(wr)), 3),
                               "std": round(float(np.std(wr)), 3)},
                    "auroc": {"mean": round(float(np.nanmean(au)), 3),
                              "std": round(float(np.nanstd(au)), 3)}}
        print(f"  [r3b/{arm}] within={res[arm]['within']['mean']}±{res[arm]['within']['std']} "
              f"AUROC={res[arm]['auroc']['mean']}", flush=True)
    res["delta"] = {k: round(res["foldwise_full"][k]["mean"] - res["cohort"][k]["mean"], 4)
                    for k in ("pooled", "within", "auroc")}
    return res


def main():
    """Load raw voxels bound to the cache cohort, build purged folds, run R3a
    and R3b, and write both result JSONs."""
    global SITE
    print("[adni purged] loading raw voxels ...", flush=True)
    X, y, site, ptids = load_adni_raw_with_ptids()
    SITE = site
    print(f"  X={X.shape}", flush=True)
    folds = purged_folds(site, ptids)
    print(f"  purged folds: {sorted(folds)}", flush=True)
    out_a = {"adni_purged": r3a(X, y, folds)}
    json.dump(out_a, open(RESDIR / "foldwise_pca_purged.json", "w"), indent=2)
    out_b = {"adni_purged": r3b(X, y, folds)}
    json.dump(out_b, open(RESDIR / "sourceonly_purged.json", "w"), indent=2)
    print("wrote", RESDIR / "foldwise_pca_purged.json", "and",
          RESDIR / "sourceonly_purged.json", flush=True)


if __name__ == "__main__":
    main()
