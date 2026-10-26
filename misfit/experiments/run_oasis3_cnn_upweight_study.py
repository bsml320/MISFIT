"""Controlled upweighting study, OASIS-3 CNN, exactly per the frozen prespecification
(paper/feedback/PRESPEC_cnn_upweight_20261011.md, committed @181c072 before this run).

Six arms (uniform, misfit, ordinary, aum, el2n, stage_random), top-20% budget,
weight 3 renormalized to mean 1, 3 fresh common seeds (52-54), 5 held scanners.
The outer evaluation scanner stays outside preprocessing fitting (per-fold
source-pool normalization, a stated deviation from the backbone's global
normalization), scoring, selection and calibration; argmax operating point only.
Stages: train | analyze | all. Resume with RESUME=1 (skips folds already saved).
Outputs: results/oasis3/oasis3_cnn_upweight_arrays.npz, .../oasis3_cnn_upweight_study.json
GPU 2.
"""
from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

import json
import os
import sys
import time

# CUDA visibility is selected by the caller.
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

sys.path.insert(0, _resolve_path("<REPO_ROOT>/misfit/experiments"))
from misfit.experiments.run_oasis3_cnn_backbone import (  # noqa: E402
    BS, DEV, EPOCHS, WORK, CNN3D, load_cohort)

RES = _resolve_path("<REPO_ROOT>/misfit/results/oasis3")
PROV = _resolve_path("<REPO_ROOT>/misfit/results/provenance")
ARR = f"{RES}/oasis3_cnn_upweight_arrays.npz"
OUT = f"{RES}/oasis3_cnn_upweight_study.json"
RECORDS = f"{PROV}/oasis3_cdr_participant_records.csv"
SCORE_SEED = 41
FRESH_SEEDS = [52, 53, 54]
ARMS = ["uniform", "misfit", "ordinary", "aum", "el2n", "stage_random"]
BUDGET, WEIGHT, EL2N_EP = 0.20, 3.0, 20
SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    EPOCHS, FRESH_SEEDS, EL2N_EP = 2, [52], 1
    ARR = os.environ.get("SMOKE_ARR", "/tmp/smoke_upweight_arrays.npz")
    OUT = os.environ.get("SMOKE_OUT", "/tmp/smoke_upweight_study.json")


def load_volumes_raw(sess):
    """Backbone loader minus its global normalization (fit per fold instead)."""
    def one(s):
        v = nib.load(f"{WORK}/{s}/smwc1pT1.nii").get_fdata(dtype=np.float32)
        t = torch.from_numpy(v)[None, None]
        return F.avg_pool3d(t, 2)[0, 0]
    with ThreadPoolExecutor(max_workers=16) as ex:
        vols = list(ex.map(one, sess))
    return torch.stack(vols)[:, None]


def batched_logits(m, X, idx):
    outs = []
    with torch.no_grad():
        for i in range(0, len(idx), 128):
            outs.append(m(X[idx[i:i + 128]].to(DEV)).cpu())
    return torch.cat(outs)


def fit(Xn, yt, tr, seed, perm_key, w=None, record=None, mode="full"):
    """One CNN fit on tr. w: per-subject weights over all N (mean 1 on tr) or None
    (exactly the backbone loss). record: dict to fill over tr; mode 'ce50' records
    only the final-epoch training CE (inner misfit fits), 'full' adds aum + el2n
    (designated scoring fit)."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(perm_key)
    m = CNN3D().to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-4, weight_decay=1e-4)
    wt = None if w is None else torch.from_numpy(w).float()
    margin_sum = np.zeros(len(tr))
    for ep in range(1, EPOCHS + 1):
        m.train()
        perm = rng.permutation(tr)
        for i in range(0, len(perm), BS):
            b = perm[i:i + BS]
            opt.zero_grad()
            ce = F.cross_entropy(m(Xn[b].to(DEV)), yt[b].to(DEV), reduction="none")
            loss = ce.mean() if wt is None else (wt[b].to(DEV) * ce).mean()
            loss.backward()
            opt.step()
        want_full = record is not None and mode == "full"
        want_now = want_full or (record is not None and ep == EPOCHS)
        if want_now:
            m.eval()
            lg = batched_logits(m, Xn, tr)
            if want_full:
                true = lg.gather(1, yt[tr][:, None]).squeeze(1)
                other = lg.clone()
                other.scatter_(1, yt[tr][:, None], -1e9)
                margin_sum += (true - other.max(1).values).numpy()
                if ep == EL2N_EP:
                    p = F.softmax(lg, dim=1)
                    record["el2n"] = (p - F.one_hot(yt[tr], 2).float()).norm(dim=1).numpy()
            if ep == EPOCHS:
                record["ce50"] = F.cross_entropy(lg, yt[tr], reduction="none").numpy()
                if want_full:
                    record["aum"] = -margin_sum / EPOCHS
    m.eval()
    return m


def topk(score, tr, k):
    """Top-k of tr by score (higher = selected); stable argsort on float64."""
    o = np.argsort(-score[tr].astype(np.float64), kind="stable")
    return tr[o[:k]]


def stage_train():
    sess, y, site, order = load_cohort()
    N = len(y)
    rec = pd.read_csv(RECORDS)
    assert list(rec["session"].astype(str)) == list(sess), "records order mismatch"
    stage = rec["cdr_stage_0_05_1plus"].to_numpy()
    print(f"N={N} impaired={y.sum()} scanners={order}", flush=True)
    X = load_volumes_raw(sess)
    print(f"volumes {tuple(X.shape)} (raw, per-fold normalization)", flush=True)
    yt = torch.from_numpy(y).long()
    n_sites = int(site.max()) + 1
    outer = [h for h in range(n_sites) if (site == h).sum() >= 5]
    if SMOKE:
        outer = outer[-1:]
    nS = len(FRESH_SEEDS)
    nan = lambda *s: np.full(s, np.nan)  # noqa: E731
    scores = {nm: nan(n_sites, N) for nm in ["misfit", "ordinary", "aum", "el2n"]}
    sel_fixed = {nm: np.zeros((n_sites, N), bool) for nm in ["misfit", "ordinary", "aum", "el2n"]}
    sel_rand = np.zeros((n_sites, nS, N), bool)
    pred = np.full((len(ARMS), nS, N), -1, dtype=np.int64)
    prob = nan(len(ARMS), nS, N)
    norm_mu, norm_sd = nan(n_sites), nan(n_sites)
    done = []
    if os.environ.get("RESUME") == "1" and os.path.exists(ARR):
        d = np.load(ARR, allow_pickle=True)
        done = [int(x) for x in d["done_folds"]]
        for nm in scores:
            scores[nm] = d[f"score_{nm}"]
            sel_fixed[nm] = d[f"sel_{nm}"]
        sel_rand, pred, prob = d["sel_stage_random"], d["pred"], d["prob"]
        norm_mu, norm_sd = d["norm_mu"], d["norm_sd"]
        print(f"resuming; folds done: {done}", flush=True)

    def save():
        np.savez(ARR, sessions=np.array(sess), label=y, site=site, stage=stage,
                 scanners=np.array(order), arms=np.array(ARMS),
                 seeds=np.array(FRESH_SEEDS), outer_folds=np.array(outer),
                 done_folds=np.array(done), norm_mu=norm_mu, norm_sd=norm_sd,
                 sel_stage_random=sel_rand, pred=pred, prob=prob,
                 **{f"score_{nm}": scores[nm] for nm in scores},
                 **{f"sel_{nm}": sel_fixed[nm] for nm in sel_fixed})

    for held in outer:
        if held in done:
            continue
        t0 = time.time()
        tr = np.where(site != held)[0]
        te = np.where(site == held)[0]
        mu = float(X[tr].mean())
        sd = float(X[tr].std())
        norm_mu[held], norm_sd[held] = mu, sd
        Xn = (X - mu) / (sd + 1e-6)
        print(f"fold held={held} ({order[held]}): n_tr={len(tr)} n_te={len(te)} "
              f"mu={mu:.4f} sd={sd:.4f}", flush=True)
        # ---- scoring: inner LOSO misfit (training CE ep50, cross-fit mean) ----
        ce_sum, ce_cnt = np.zeros(N), np.zeros(N)
        for ih in sorted(np.unique(site[tr])):
            itr = tr[site[tr] != ih]
            r = {}
            fit(Xn, yt, itr, SCORE_SEED, SCORE_SEED * 100 + ih, record=r, mode="ce50")
            ce_sum[itr] += r["ce50"]
            ce_cnt[itr] += 1
            print(f"  inner held {ih}: done", flush=True)
        msk = ce_cnt > 0
        scores["misfit"][held, msk] = ce_sum[msk] / ce_cnt[msk]
        # ---- scoring: designated full-source fit (ordinary, aum, el2n) ----
        r = {}
        fit(Xn, yt, tr, SCORE_SEED, SCORE_SEED * 100 + 99, record=r)
        for nm, key in [("ordinary", "ce50"), ("aum", "aum"), ("el2n", "el2n")]:
            scores[nm][held, tr] = r[key]
        print("  designated scoring fit: done", flush=True)
        # ---- selection ----
        k = round(BUDGET * len(tr))
        for nm in ["misfit", "ordinary", "aum", "el2n"]:
            sel_fixed[nm][held, topk(scores[nm][held], tr, k)] = True
        mis_sel = np.where(sel_fixed["misfit"][held])[0]
        stage_counts = {v: int((stage[mis_sel] == v).sum()) for v in (0, 1, 2)}
        for s_i in range(nS):
            rng = np.random.RandomState(9000 + 10 * held + s_i)
            for v, c in stage_counts.items():
                pool = tr[stage[tr] == v]
                if c > 0:
                    sel_rand[held, s_i, rng.choice(pool, c, replace=False)] = True
        print(f"  k={k} misfit stage profile={stage_counts}", flush=True)
        # ---- arms ----
        for s_i, seed in enumerate(FRESH_SEEDS):
            for a_i, arm in enumerate(ARMS):
                if arm == "uniform":
                    w = None
                else:
                    sel = (sel_rand[held, s_i] if arm == "stage_random"
                           else sel_fixed[arm][held])
                    w = np.ones(N)
                    w[sel] = WEIGHT
                    w[tr] *= len(tr) / w[tr].sum()
                m = fit(Xn, yt, tr, seed, seed * 100 + held, w=w)
                lg = batched_logits(m, Xn, te)
                pred[a_i, s_i, te] = lg.argmax(1).numpy()
                prob[a_i, s_i, te] = F.softmax(lg, dim=1)[:, 1].numpy()
            print(f"  seed {seed}: all {len(ARMS)} arms done", flush=True)
        done.append(held)
        save()
        print(f"fold held={held} complete in {(time.time()-t0)/60:.1f} min; saved",
              flush=True)
    del X
    print("train stage complete", flush=True)


def pooled_sens(pred_a, subj):
    """Pooled fraction predicted impaired over subjects x seeds (equal trials)."""
    return float((pred_a[:, subj] == 1).mean())


def boot_diff(pred_m, pred_c, subj, B=2000, seed=7):
    rng = np.random.RandomState(seed)
    n = len(subj)
    vals = []
    for _ in range(B):
        r = subj[rng.randint(0, n, n)]
        vals.append((pred_m[:, r] == 1).mean() - (pred_c[:, r] == 1).mean())
    return [round(float(np.percentile(vals, q)), 4) for q in (2.5, 97.5)]


def stage_analyze():
    d = np.load(ARR, allow_pickle=True)
    y, site, stage = d["label"], d["site"], d["stage"]
    order = [str(s) for s in d["scanners"]]
    outer = [int(h) for h in d["outer_folds"]]
    pred, prob = d["pred"], d["prob"]
    tested = np.isin(site, outer)
    i05 = np.where(tested & (y == 1) & (stage == 1))[0]
    i1p = np.where(tested & (y == 1) & (stage == 2))[0]
    ictl = np.where(tested & (stage == 0))[0]
    iall = np.where(tested)[0]
    res = {"design": {"prespec": "paper/feedback/PRESPEC_cnn_upweight_20261011.md",
                      "prespec_commit": "181c072", "arms": ARMS,
                      "seeds": [int(s) for s in d["seeds"]], "budget": BUDGET,
                      "weight": WEIGHT, "score_seed": SCORE_SEED,
                      "outer_folds": [order[h] for h in outer],
                      "n_tested": int(tested.sum()), "n_cdr05": len(i05),
                      "n_cdr1plus": len(i1p), "n_cdr0": len(ictl)},
           "arms": {}, "primary": {}, "contrasts": {}, "overlap": {}}
    for a_i, arm in enumerate(ARMS):
        p = pred[a_i]
        per_seed = [float((p[s][i05] == 1).mean()) for s in range(p.shape[0])]
        aur = [float(roc_auc_score(y[iall], prob[a_i, s, iall])) for s in range(p.shape[0])]
        per_sc = {order[h]: (float((p[:, idx] == 1).mean()) if len(idx) else None)
                  for h in outer
                  for idx in [i05[site[i05] == h]]}
        res["arms"][arm] = {
            "sens_cdr05": round(pooled_sens(p, i05), 4),
            "sens_cdr05_per_seed": [round(v, 4) for v in per_seed],
            "sens_cdr1plus": round(pooled_sens(p, i1p), 4),
            "specificity": round(float((p[:, ictl] == 0).mean()), 4),
            "accuracy": round(float((p[:, iall] == y[iall]).mean()), 4),
            "auroc_mean": round(float(np.mean(aur)), 4),
            "auroc_std": round(float(np.std(aur)), 4),
            "per_scanner_sens_cdr05": {k: (round(v, 4) if v is not None else None)
                                       for k, v in per_sc.items()}}
    comp = ["ordinary", "aum", "el2n", "stage_random"]
    strongest = max(comp, key=lambda c: res["arms"][c]["sens_cdr05"])
    mi, ci_ = ARMS.index("misfit"), ARMS.index(strongest)
    diff = res["arms"]["misfit"]["sens_cdr05"] - res["arms"][strongest]["sens_cdr05"]
    ci = boot_diff(pred[mi], pred[ci_], i05)
    spec_diff = res["arms"]["misfit"]["specificity"] - res["arms"][strongest]["specificity"]
    safeguard = spec_diff >= -0.02
    excl = ci[0] > 0 or ci[1] < 0
    res["primary"] = {
        "endpoint": "pooled held-scanner CDR-0.5 sensitivity, argmax",
        "strongest_comparator": strongest,
        "diff_misfit_minus_strongest": round(diff, 4), "ci95": ci,
        "ci_excludes_zero": bool(excl),
        "specificity_diff": round(spec_diff, 4), "safeguard_pass": bool(safeguard),
        "benefit_claimed": bool(excl and diff > 0 and safeguard)}
    for c in comp + ["uniform"]:
        res["contrasts"][f"misfit_vs_{c}"] = {
            "diff": round(res["arms"]["misfit"]["sens_cdr05"]
                          - res["arms"][c]["sens_cdr05"], 4),
            "ci95": boot_diff(pred[mi], pred[ARMS.index(c)], i05)}
    for c in comp:
        res["contrasts"][f"{c}_vs_uniform"] = {
            "diff": round(res["arms"][c]["sens_cdr05"]
                          - res["arms"]["uniform"]["sens_cdr05"], 4),
            "ci95": boot_diff(pred[ARMS.index(c)], pred[ARMS.index("uniform")], i05)}
    for nm in ["ordinary", "aum", "el2n"]:
        js = []
        for h in outer:
            a, b = d["sel_misfit"][h], d[f"sel_{nm}"][h]
            js.append((a & b).sum() / (a | b).sum())
        res["overlap"][f"misfit_vs_{nm}_jaccard"] = round(float(np.mean(js)), 3)
    js = []
    for h in outer:
        a = d["sel_misfit"][h]
        for s_i in range(d["sel_stage_random"].shape[1]):
            b = d["sel_stage_random"][h, s_i]
            js.append((a & b).sum() / (a | b).sum())
    res["overlap"]["misfit_vs_stage_random_jaccard"] = round(float(np.mean(js)), 3)
    prof = {}
    for nm in ["misfit", "ordinary", "aum", "el2n"]:
        cnt = np.array([[int((stage[d[f"sel_{nm}"][h]] == v).sum()) for v in (0, 1, 2)]
                        for h in outer], float)
        prof[nm] = [round(float(x), 1) for x in cnt.mean(0)]
    res["selection_stage_profile_mean_counts_0_05_1plus"] = prof
    res["honesty"] = {
        "notes": ["cuDNN conv3d backward nondeterminism can prevent bit-exact reproduction",
                  "Sonata contributes no CDR-0.5 impaired tested subjects; the primary "
                  "pool spans 4 held scanners",
                  "per-fold source-pool normalization is a stated deviation from the "
                  "backbone's global normalization; absolute accuracies are not "
                  "directly comparable to the committed backbone run",
                  "CDR also defines the label; retrospective cohort; argmax operating "
                  "point, no calibration anywhere"]}
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps(res["primary"], indent=2))
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    os.makedirs(RES, exist_ok=True)
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("train", "all"):
        stage_train()
    if what in ("analyze", "all"):
        stage_analyze()
