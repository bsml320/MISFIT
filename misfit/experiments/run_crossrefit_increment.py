"""AT-S14: isolate the incremental value of the CROSS-REFIT construction.

The plain ERM comparator also averages per-subject loss across environment-excluding
refits, so its proximity to the misfit tests the operator, not refit averaging. Here
the score-construction axis is varied directly, everything else fixed:

  crossrefit       : mean per-subject training CE over the E-1 environment-excluding
                     refits containing the subject (the paper's construction)  [E fits]
  single_refit     : the same quantity from ONE environment-excluding refit (the first
                     fold in which the subject trains)                          [reuses]
  single_all       : one classifier fit on ALL environments; its per-subject final CE
                                                                                [1 fit]
  seed_avg         : mean CE over E-1 all-data fits differing only by seed     [E-1 fits]
  random_partition : mean CE over E refits that each exclude a random pseudo-
                     environment (subject subsets matched in size to the real
                     environments)                                              [E fits]

Common target: per-subject binary error from environment-excluding PlainMLP target
classifiers (target seed disjoint from score seeds). Per (cohort, replicate s in 0..2):
error vector from target seed 1000+s; all five scores built at score seed s; graded by
pooled/within-env Spearman and AUROC. Rank stability = mean Spearman between a score's
vectors across replicates. Advantage axis is predefined as discrimination + rank
stability + fit count (cost). Cohorts: ADNI, OASIS-1, leakage-free OASIS-3. GPU 2.
Output: results/crossrefit_increment.json
"""
from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

import json
import os
import re
import sys

# CUDA visibility is selected by the caller.
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

sys.path.insert(0, _resolve_path("<REPO_ROOT>"))
from misfit.experiments.run_honest_estimator_table import (  # noqa: E402
    PlainMLP, load_adni, load_oasis_ds)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, REPS = 100, 3
OUT = _resolve_path("<REPO_ROOT>/misfit/results/crossrefit_increment.json")


def load_oasis3():
    """Leakage-free OASIS-3 loader: earliest session per participant, scanner sites."""
    d = np.load(_resolve_path("<REPO_ROOT>/data/oasis3/oasis3_features.npz"), allow_pickle=True)
    subj = d["subjects"].astype(str)
    sess = d["sessions"].astype(str)
    day = np.array([int(re.search(r"_d(\d+)$", x).group(1)) for x in sess])
    keep = np.array(sorted(np.where(subj == u)[0][np.argmin(day[subj == u])]
                           for u in np.unique(subj)))
    lab = d["label"].astype(int)[keep]
    scan = d["scanner"].astype(str)[keep]
    uniq, cnt = np.unique(scan, return_counts=True)
    order = uniq[np.argsort(-cnt)]
    code = {s: i for i, s in enumerate(order)}
    return (torch.tensor(d["features"][keep], dtype=torch.float32),
            torch.tensor(lab), torch.tensor([code[x] for x in scan]))


def fit_losses(X, y, train_idx, seed, want_pred_idx=None):
    """Train PlainMLP on train_idx; return per-subject CE on train_idx (+ preds)."""
    torch.manual_seed(seed)
    m = PlainMLP(X.shape[1], 2).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt, yt = X[train_idx].to(DEV), y[train_idx].long().to(DEV)
    m.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        F.cross_entropy(m(Xt), yt).backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        ce = F.cross_entropy(m(Xt), yt, reduction="none").cpu().numpy()
        pred = None
        if want_pred_idx is not None:
            pred = m(X[want_pred_idx].to(DEV)).argmax(1).cpu().numpy()
    return ce, pred


def target_errors(X, y, site, seed):
    """Per-subject binary error from environment-excluding target classifiers.

    The target seed is kept disjoint from the score seeds so the error vector
    being predicted is not produced by the same fits as the scores.
    """
    N = len(y)
    pred = np.full(N, -1)
    tested = np.zeros(N, bool)
    for e in np.unique(site.numpy()):
        tr = np.where(site.numpy() != e)[0]
        te = np.where(site.numpy() == e)[0]
        if len(te) < 5:
            continue
        _, p = fit_losses(X, y, tr, seed, want_pred_idx=te)
        pred[te] = p
        tested[te] = True
    return (pred != y.numpy()).astype(float), tested


def build_scores(X, y, site, seed, rng):
    """Build the five score constructions described in the module docstring.

    Returns:
        Tuple (scores dict name -> per-subject vector, fits dict name -> fit count).
    """
    N = len(y)
    sn = site.numpy()
    envs = np.unique(sn)
    E = len(envs)
    scores, fits = {}, {}

    # crossrefit + single_refit (reuse the same env-excluding fits)
    ssum, cnt = np.zeros(N), np.zeros(N)
    single = np.full(N, np.nan)
    for e in envs:
        tr = np.where(sn != e)[0]
        ce, _ = fit_losses(X, y, tr, seed * 31 + int(e))
        ssum[tr] += ce
        cnt[tr] += 1
        first = np.isnan(single[tr])
        single[tr[first]] = ce[first]
    scores["crossrefit"] = ssum / np.maximum(cnt, 1)
    scores["single_refit"] = single
    fits["crossrefit"], fits["single_refit"] = E, f"reuses the {E}"

    # single_all
    allidx = np.arange(N)
    ce, _ = fit_losses(X, y, allidx, seed * 31 + 500)
    scores["single_all"] = ce
    fits["single_all"] = 1

    # seed_avg: E-1 all-data fits, different seeds
    acc = np.zeros(N)
    for k in range(E - 1):
        ce, _ = fit_losses(X, y, allidx, seed * 31 + 600 + k)
        acc += ce
    scores["seed_avg"] = acc / (E - 1)
    fits["seed_avg"] = E - 1

    # random_partition: E refits, each excluding a random pseudo-env matched in size
    sizes = [int((sn == e).sum()) for e in envs]
    perm = rng.permutation(N)
    bounds = np.cumsum([0] + sizes)
    ssum, cnt = np.zeros(N), np.zeros(N)
    for k in range(E):
        excl = perm[bounds[k]:bounds[k + 1]]
        tr = np.setdiff1d(np.arange(N), excl)
        ce, _ = fit_losses(X, y, tr, seed * 31 + 700 + k)
        ssum[tr] += ce
        cnt[tr] += 1
    scores["random_partition"] = ssum / np.maximum(cnt, 1)
    fits["random_partition"] = E
    return scores, fits


def grade(score, err, sn, tested):
    """Score one construction: (pooled rho, mean within-environment rho, AUROC)."""
    m = tested & np.isfinite(score)
    pooled = float(spearmanr(score[m], err[m])[0])
    auc = float(roc_auc_score(err[m], score[m])) if 0 < err[m].sum() < m.sum() else float("nan")
    wr = []
    for e in np.unique(sn[m]):
        mm = m & (sn == e)
        ee = err[mm]
        if len(ee) >= 8 and 0 < ee.sum() < len(ee):
            r = spearmanr(score[mm], ee)[0]
            if np.isfinite(r):
                wr.append(r)
    return pooled, float(np.mean(wr)), auc


def main():
    """Run all cohorts and replicates; write results/crossrefit_increment.json."""
    res = {}
    for name, loader in [("adni", load_adni), ("oasis1", load_oasis_ds), ("oasis3", load_oasis3)]:
        X, y, site = loader()
        sn = site.numpy()
        print(f"[{name}] N={len(y)} E={len(np.unique(sn))}", flush=True)
        per = {}
        vectors = {}
        fits_note = None
        for s in range(REPS):
            err, tested = target_errors(X, y, site, 1000 + s)
            rng = np.random.RandomState(2000 + s)
            scores, fits = build_scores(X, y, site, s, rng)
            fits_note = fits
            for k, v in scores.items():
                p, w, a = grade(v, err, sn, tested)
                per.setdefault(k, {"pooled": [], "within": [], "auroc": []})
                per[k]["pooled"].append(p)
                per[k]["within"].append(w)
                per[k]["auroc"].append(a)
                vectors.setdefault(k, []).append(v)
            print(f"  rep {s} done", flush=True)
        res[name] = {}
        for k, v in per.items():
            stab = np.mean([spearmanr(vectors[k][i], vectors[k][j])[0]
                            for i in range(REPS) for j in range(i + 1, REPS)])
            res[name][k] = {
                "pooled_rho": float(np.mean(v["pooled"])), "pooled_std": float(np.std(v["pooled"])),
                "within_env_rho": float(np.mean(v["within"])), "within_std": float(np.std(v["within"])),
                "auroc": float(np.nanmean(v["auroc"])), "auroc_std": float(np.nanstd(v["auroc"])),
                "rank_stability": float(stab), "n_fits": str(fits_note[k])}
            r = res[name][k]
            print(f"  {k:17s} pooled={r['pooled_rho']:.3f} within={r['within_env_rho']:.3f} "
                  f"AUROC={r['auroc']:.3f} stab={r['rank_stability']:.3f} fits={r['n_fits']}",
                  flush=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
