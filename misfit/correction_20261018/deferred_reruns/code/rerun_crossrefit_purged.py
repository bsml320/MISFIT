"""R2: corrected ADNI arm of the loss-construction comparison.

Rerun of misfit/experiments/run_crossrefit_increment.py (ADNI arm) with
the plan deltas (RERUN_PLAN.md Section 3):
  - target classifiers and the crossrefit / single_refit constructions use
    purged masks and per-fold source-only features (shared fold projections);
  - single_all and seed_avg are unchanged (all-data fits on the historical
    cohort feature matrix: no held-out environment, so purging does not apply
    and the cohort preprocessing is their own training preprocessing);
  - random_partition keeps the historical acquisition-level pseudo-environment
    partition VERBATIM (RandomState(2000+s), rng.permutation over the rows,
    contiguous size-matched blocks) and applies the SAME purge rule as the
    real folds: refit k excludes pseudo-environment k together with every
    acquisition whose participant has an acquisition in pseudo-environment k,
    and fits its features source-only on its own training rows (statistical
    review B1).
Unchanged: PlainMLP, 100 epochs, wd 1e-4, replicates s=0..2, target seeds
1000+s, score-fit seeds s*31+{env, 500, 600+k, 700+k}, grading (pooled and
within-real-protocol Spearman, AUROC), rank stability across replicates,
np.std ddof=0. GPU 2.

Output: deferred_reruns/results/crossrefit_purged.json
"""
from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

import json
import os
import sys
from pathlib import Path

# CUDA visibility is selected by the caller.
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rerun_common import fold_pca, load, purged_fold_projections  # noqa: E402

sys.path.insert(0, _resolve_path("<REPO_ROOT>"))
from misfit.experiments.run_honest_estimator_table import PlainMLP  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, REPS = 100, 3
OUT = Path(__file__).resolve().parents[1] / "results" / "crossrefit_purged.json"


def fit_losses(Z, y, train_idx, seed, want_pred_idx=None):
    """Train PlainMLP on train_idx in feature space Z; per-subject CE (+preds)."""
    torch.manual_seed(seed)
    m = PlainMLP(Z.shape[1], 2).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt, yt = Z[train_idx].to(DEV), y[train_idx].long().to(DEV)
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
            pred = m(Z[want_pred_idx].to(DEV)).argmax(1).cpu().numpy()
    return ce, pred


def target_errors(folds, y, seed):
    """Held-out error indicator and tested mask from one target classifier per fold."""
    n = len(y)
    pred = np.full(n, -1)
    tested = np.zeros(n, bool)
    for e in sorted(folds):
        f = folds[e]
        tr = np.where(f["train"])[0]
        te = np.where(f["test"])[0]
        _, p = fit_losses(f["Z"], y, tr, seed, want_pred_idx=te)
        pred[te] = p
        tested[te] = True
    return (pred != y.numpy()).astype(float), tested


def build_scores(folds, raw, cohort_X, y, site, ptids, seed, rng):
    """All loss-construction scores for one replicate.

    crossrefit / single_refit reuse the purged env-excluding refits with
    per-fold features; single_all and seed_avg fit on all rows with cohort
    features (unchanged by design); random_partition rebuilds the historical
    pseudo-environments verbatim and applies the same participant purge and
    source-only features as the real folds.
    """
    n = len(y)
    envs = sorted(folds)
    e_count = len(envs)
    scores, fits = {}, {}

    # crossrefit + single_refit (purged env-excluding refits, per-fold features)
    ssum, cnt = np.zeros(n), np.zeros(n)
    single = np.full(n, np.nan)
    for e in envs:
        f = folds[e]
        tr = np.where(f["train"])[0]
        ce, _ = fit_losses(f["Z"], y, tr, seed * 31 + int(e))
        ssum[tr] += ce
        cnt[tr] += 1
        first = np.isnan(single[tr])
        single[tr[first]] = ce[first]
    scores["crossrefit"] = ssum / np.maximum(cnt, 1)
    scores["single_refit"] = single
    fits["crossrefit"], fits["single_refit"] = e_count, f"reuses the {e_count}"

    # single_all (all-data fit, cohort features; unchanged by design)
    allidx = np.arange(n)
    ce, _ = fit_losses(cohort_X, y, allidx, seed * 31 + 500)
    scores["single_all"] = ce
    fits["single_all"] = 1

    # seed_avg: E-1 all-data fits, different seeds (unchanged by design)
    acc = np.zeros(n)
    for k in range(e_count - 1):
        ce, _ = fit_losses(cohort_X, y, allidx, seed * 31 + 600 + k)
        acc += ce
    scores["seed_avg"] = acc / (e_count - 1)
    fits["seed_avg"] = e_count - 1

    # random_partition: historical acquisition-level pseudo-environments
    # (verbatim RNG stream) + the same purge rule as the real folds
    sizes = [int((site == e).sum()) for e in envs]
    perm = rng.permutation(n)
    bounds = np.cumsum([0] + sizes)
    ssum, cnt = np.zeros(n), np.zeros(n)
    for k in range(e_count):
        excl = perm[bounds[k]:bounds[k + 1]]
        pe = set(ptids[excl])
        tr_mask = ~np.isin(ptids, sorted(pe))
        tr_mask[excl] = False
        tr = np.where(tr_mask)[0]
        proj = fold_pca(raw, tr)
        zfull = proj(np.arange(n))
        ce, _ = fit_losses(zfull, y, tr, seed * 31 + 700 + k)
        ssum[tr] += ce
        cnt[tr] += 1
        print(f"    random_partition refit {k}: pseudo-env {len(excl)} rows, "
              f"purged train pool {len(tr)}", flush=True)
    scores["random_partition"] = ssum / np.maximum(cnt, 1)
    fits["random_partition"] = e_count
    return scores, fits


def grade(score, err, sn, tested):
    """Pooled Spearman, mean within-protocol Spearman (cells with n >= 8 and
    both outcomes), and pooled AUROC over tested rows."""
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
    """Run the 3 replicates, aggregate per score with cross-replicate rank
    stability, and write the output JSON."""
    d = load()
    raw, y, site, ptids = d["raw"], d["targets"], d["site_ids"], d["ptids"]
    cohort_X = d["features"]
    folds = purged_fold_projections(raw, site, ptids)
    assert len(folds) == 8, f"fold set changed: {sorted(folds)} (plan C9)"
    sn = site.numpy()
    print(f"[adni purged] N={len(y)} E={len(folds)}", flush=True)
    per, vectors, fits_note = {}, {}, None
    for s in range(REPS):
        err, tested = target_errors(folds, y, 1000 + s)
        rng = np.random.RandomState(2000 + s)
        scores, fits = build_scores(folds, raw, cohort_X, y, sn, ptids, s, rng)
        fits_note = fits
        for k, v in scores.items():
            p, w, a = grade(v, err, sn, tested)
            per.setdefault(k, {"pooled": [], "within": [], "auroc": []})
            per[k]["pooled"].append(p)
            per[k]["within"].append(w)
            per[k]["auroc"].append(a)
            vectors.setdefault(k, []).append(v)
        print(f"  rep {s} done", flush=True)
    res = {"adni_purged": {}}
    for k, v in per.items():
        stab = np.mean([spearmanr(vectors[k][i], vectors[k][j])[0]
                        for i in range(REPS) for j in range(i + 1, REPS)])
        res["adni_purged"][k] = {
            "pooled_rho": float(np.mean(v["pooled"])), "pooled_std": float(np.std(v["pooled"])),
            "within_env_rho": float(np.mean(v["within"])), "within_std": float(np.std(v["within"])),
            "auroc": float(np.nanmean(v["auroc"])), "auroc_std": float(np.nanstd(v["auroc"])),
            "rank_stability": float(stab), "n_fits": str(fits_note[k])}
        r = res["adni_purged"][k]
        print(f"  {k:17s} pooled={r['pooled_rho']:.3f} within={r['within_env_rho']:.3f} "
              f"AUROC={r['auroc']:.3f} stab={r['rank_stability']:.3f} fits={r['n_fits']}",
              flush=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
