"""Experiment 2: audit-guided intervention, exactly per the frozen prespecification
(paper/feedback/PRESPEC_intervention_20261003.md, committed before this run).

One action (remove top-q source subjects by score; q in {0.10, 0.20}), seven selection
methods computed from SOURCE data only, identical PlainMLP retraining (3 retrain seeds),
evaluation on the untouched held-out environment: accuracy, balanced accuracy,
sensitivity, specificity. Primary endpoint = delta balanced accuracy vs `none`,
equal weight per fold x retrain-seed cell. All cells reported; no post-hoc selection.
Cohorts: ADNI (PCA-50, 8 envs), leakage-free OASIS-3 (PCA-50, 6 envs). GPU.
Output: results/intervention_audit.json
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

sys.path.insert(0, _resolve_path("<REPO_ROOT>"))
from misfit.experiments.run_honest_estimator_table import (  # noqa: E402
    PlainMLP, load_adni)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 100
BUDGETS = [0.10, 0.20]
RETRAIN_SEEDS = [100, 101, 102]
OUT = _resolve_path("<REPO_ROOT>/misfit/results/intervention_audit.json")


def load_oasis3():
    """Leakage-free OASIS-3: one earliest session per participant, scanner
    codes ordered by frequency. Returns (features, targets, site_ids)."""
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


def fit(X, y, idx, seed, weights=None, track=None):
    """Train PlainMLP on idx. track='dyn' also records AUM area + EL2N@20.
    Returns (per-subject final CE on idx, model, aum, el2n)."""
    torch.manual_seed(seed)
    m = PlainMLP(X.shape[1], 2).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt, yt = X[idx].to(DEV), y[idx].long().to(DEV)
    w = None if weights is None else torch.tensor(weights, dtype=torch.float32, device=DEV)
    margin_sum = torch.zeros(len(idx), device=DEV)
    el2n = None
    m.train()
    for ep in range(1, EPOCHS + 1):
        opt.zero_grad()
        logits = m(Xt)
        loss = F.cross_entropy(logits, yt, reduction="none")
        (loss.mean() if w is None else (loss * w).sum() / w.sum()).backward()
        opt.step()
        if track == "dyn":
            with torch.no_grad():
                lg = m(Xt)
                true = lg.gather(1, yt[:, None]).squeeze(1)
                other = lg.clone()
                other.scatter_(1, yt[:, None], -1e9)
                margin_sum += true - other.max(1).values
                if ep == 20:
                    p = F.softmax(lg, dim=1)
                    onehot = F.one_hot(yt, 2).float()
                    el2n = (p - onehot).norm(dim=1).cpu().numpy()
    m.eval()
    with torch.no_grad():
        ce = F.cross_entropy(m(Xt), yt, reduction="none").cpu().numpy()
    aum = (-(margin_sum / EPOCHS).cpu().numpy()) if track == "dyn" else None
    return ce, m, aum, el2n


def evaluate(m, X, y, te):
    """Held-out accuracy, balanced accuracy, sensitivity and specificity."""
    with torch.no_grad():
        pred = m(X[te].to(DEV)).argmax(1).cpu().numpy()
    t = y[te].numpy()
    acc = float((pred == t).mean())
    sens = float((pred[t == 1] == 1).mean()) if (t == 1).sum() else float("nan")
    spec = float((pred[t == 0] == 0).mean()) if (t == 0).sum() else float("nan")
    bal = float(np.nanmean([sens, spec]))
    return {"acc": acc, "bal": bal, "sens": sens, "spec": spec}


def source_scores(X, y, site, src_idx, fold_tag):
    """All selection scores on source subjects only (prespec methods 1-6)."""
    ssite = site[src_idx].numpy()
    envs = np.unique(ssite)
    n = len(src_idx)
    S = {}
    # 1. misfit_nested: inner leave-one-source-env-out cross-refit CE, 3 inner seeds
    acc = np.zeros(n)
    cnt = np.zeros(n)
    for isd in range(3):
        for e in envs:
            inner_tr = np.where(ssite != e)[0]
            if len(inner_tr) < 20:
                continue
            ce, _, _, _ = fit(X, y, src_idx[inner_tr], 7000 + isd * 31 + int(e))
            acc[inner_tr] += ce
            cnt[inner_tr] += 1
    S["misfit_nested"] = acc / np.maximum(cnt, 1)
    # 2-5. all-source fits with dynamics (3 seeds)
    ords, aums, el2ns = [], [], []
    for isd in range(3):
        ce, _, aum, el2n = fit(X, y, src_idx, 8000 + isd, track="dyn")
        ords.append(ce)
        aums.append(aum)
        el2ns.append(el2n)
    S["ordinary_loss"] = np.mean(ords, 0)
    S["aum"] = np.mean(aums, 0)
    S["el2n"] = np.mean(el2ns, 0)
    sa = []
    for isd in range(5):
        ce, _, _, _ = fit(X, y, src_idx, 8500 + isd)
        sa.append(ce)
    S["seedavg_loss"] = np.mean(sa, 0)
    # 6. random
    import zlib
    S["random"] = np.random.RandomState(zlib.crc32(fold_tag.encode()) % (2**31)).rand(n)
    return S


def main():
    """Run every fold x method x budget x retrain-seed cell on both cohorts
    and write the aggregated audit JSON (deltas vs `none`, bootstrap CIs,
    fold-clustered SEs)."""
    out = {"prespec": "paper/feedback/PRESPEC_intervention_20261003.md @ commit 1200c2b"}
    for name, loader in [("adni", load_adni), ("oasis3", load_oasis3)]:
        X, y, site = loader()
        sn = site.numpy()
        envs = np.unique(sn)
        print(f"[{name}] N={len(y)} E={len(envs)}", flush=True)
        cells = {}  # method|budget -> list of per-cell metric dicts (fold x retrain seed)
        for held in envs:
            te = np.where(sn == held)[0]
            src_idx = np.where(sn != held)[0]
            if len(te) < 5 or len(src_idx) < 20:
                continue
            S = source_scores(X, y, site, src_idx, f"{name}{held}")
            print(f"  fold {held}: scores ready", flush=True)
            for method in ("none", "misfit_nested", "ordinary_loss", "seedavg_loss",
                           "aum", "el2n", "random"):
                budgets = [None] if method == "none" else BUDGETS
                for q in budgets:
                    if q is None:
                        keep = src_idx
                    else:
                        k = int(round(q * len(src_idx)))
                        drop = np.argsort(-S[method])[:k]
                        keepmask = np.ones(len(src_idx), bool)
                        keepmask[drop] = False
                        keep = src_idx[keepmask]
                    for rs in RETRAIN_SEEDS:
                        _, mdl, _, _ = fit(X, y, keep, rs)
                        met = evaluate(mdl, X, y, te)
                        key = method if q is None else f"{method}@{int(q*100)}"
                        cells.setdefault(key, []).append(met)
        # aggregate: equal weight per fold x retrain-seed cell; deltas vs none for ALL
        # endpoints; per-cell arrays persisted; paired cell bootstrap (descriptive, B=2000)
        # and fold-clustered SE (cells within a fold share the fold)
        res = {}
        base = cells["none"]
        fold_ids = [i // len(RETRAIN_SEEDS) for i in range(len(base))]
        rngb = np.random.RandomState(11)
        for key, v in cells.items():
            agg = {m: (round(float(np.nanmean([c[m] for c in v])), 4),
                       round(float(np.nanstd([c[m] for c in v])), 4)) for m in ("acc", "bal", "sens", "spec")}
            agg["cells"] = {m: [round(float(c[m]), 4) for c in v] for m in ("acc", "bal", "sens", "spec")}
            if key != "none":
                for m in ("acc", "bal", "sens", "spec"):
                    d = np.array([v[i][m] - base[i][m] for i in range(len(v))], float)
                    entry = {"mean": round(float(np.nanmean(d)), 4),
                             "sd": round(float(np.nanstd(d)), 4)}
                    boots = [np.nanmean(d[rngb.randint(0, len(d), len(d))]) for _ in range(2000)]
                    entry["ci95_cell_bootstrap"] = [round(float(np.percentile(boots, 2.5)), 4),
                                                    round(float(np.percentile(boots, 97.5)), 4)]
                    fm = [np.nanmean(d[np.array(fold_ids) == f]) for f in sorted(set(fold_ids))]
                    entry["fold_means"] = [round(float(x), 4) for x in fm]
                    entry["clustered_se"] = round(float(np.nanstd(fm, ddof=1) / np.sqrt(len(fm))), 4)
                    agg[f"delta_{m}_vs_none"] = entry
                agg["delta_bal_vs_none_legacy"] = (agg["delta_bal_vs_none"]["mean"],
                                                   agg["delta_bal_vs_none"]["sd"])
            res[key] = agg
            if key != "none":
                e = agg["delta_bal_vs_none"]
                print(f"  {key:20s} bal={agg['bal'][0]:.4f}  Δbal={e['mean']:+.4f} "
                      f"CI{e['ci95_cell_bootstrap']} clSE={e['clustered_se']}", flush=True)
            else:
                print(f"  {key:20s} bal={agg['bal'][0]:.4f}±{agg['bal'][1]:.4f}", flush=True)
        out[name] = res
    json.dump(out, open(OUT, "w"), indent=2)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
