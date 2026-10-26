"""18-fit OASIS-3 CNN study, exactly per the frozen prespecification
(paper/feedback/PRESPEC_cnn18_20261010.md, committed @38c6927 before this run).

Fits are identical to run_oasis3_cnn_backbone.py (same cohort lines, same CNN3D,
same hyperparameters, same seeds, same small-fold guard; the training RNG stream is
untouched, additions are eval-mode recordings only). Stages: train | analyze | all.
Outputs: results/oasis3/oasis3_cnn18_arrays.npz, results/oasis3/oasis3_cnn18_study.json
GPU 2.
"""
from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

import json
import os
import sys

# CUDA visibility is selected by the caller.
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

sys.path.insert(0, _resolve_path("<REPO_ROOT>/misfit/experiments"))
from misfit.experiments.run_oasis3_cnn_backbone import (  # noqa: E402
    BS, CKPTS, DEV, EPOCHS, SEED_BASE, SEEDS, CNN3D, batched_ce, load_cohort,
    load_volumes)

RES = _resolve_path("<REPO_ROOT>/misfit/results/oasis3")
PROV = _resolve_path("<REPO_ROOT>/misfit/results/provenance")
ARR = f"{RES}/oasis3_cnn18_arrays.npz"
OUT = f"{RES}/oasis3_cnn18_study.json"
EL2N_EP = 20


def batched_logits(m, X, idx):
    outs = []
    with torch.no_grad():
        for i in range(0, len(idx), 128):
            outs.append(m(X[idx[i:i + 128]].to(DEV)).cpu())
    return torch.cat(outs)


def stage_train():
    sess, y, site, order = load_cohort()
    N = len(y)
    print(f"N={N} impaired={y.sum()} scanners={order}", flush=True)
    X = load_volumes(sess)
    print(f"volumes {tuple(X.shape)}", flush=True)
    yt = torch.from_numpy(y).long()
    n_sites = int(site.max()) + 1
    nan = lambda *s: np.full(s, np.nan)  # noqa: E731
    ce = {ep: nan(SEEDS, n_sites, N) for ep in CKPTS}
    aum = nan(SEEDS, n_sites, N)
    el2n = nan(SEEDS, n_sites, N)
    pred = np.full((SEEDS, N), -1)
    tested = np.zeros((SEEDS, N), bool)
    for k in range(SEEDS):
        seed = SEED_BASE + k
        for held in range(n_sites):
            tr = np.where(site != held)[0]
            te = np.where(site == held)[0]
            if len(te) < 5:
                continue
            torch.manual_seed(seed)
            rng = np.random.RandomState(seed * 100 + held)
            m = CNN3D().to(DEV)
            opt = torch.optim.Adam(m.parameters(), lr=1e-4, weight_decay=1e-4)
            margin_sum = np.zeros(len(tr))
            for ep in range(1, EPOCHS + 1):
                m.train()
                perm = rng.permutation(tr)
                for i in range(0, len(perm), BS):
                    b = perm[i:i + BS]
                    opt.zero_grad()
                    F.cross_entropy(m(X[b].to(DEV)), yt[b].to(DEV)).backward()
                    opt.step()
                m.eval()
                lg = batched_logits(m, X, tr)
                true = lg.gather(1, yt[tr][:, None]).squeeze(1)
                other = lg.clone()
                other.scatter_(1, yt[tr][:, None], -1e9)
                margin_sum += (true - other.max(1).values).numpy()
                if ep == EL2N_EP:
                    p = F.softmax(lg, dim=1)
                    el2n[k, held, tr] = (p - F.one_hot(yt[tr], 2).float()).norm(dim=1).numpy()
                if ep in CKPTS:
                    ce[ep][k, held, tr] = batched_ce(m, X, yt, tr)
            aum[k, held, tr] = -margin_sum / EPOCHS
            with torch.no_grad():
                for i in range(0, len(te), 64):
                    b = te[i:i + 64]
                    pred[k, b] = m(X[b].to(DEV)).argmax(1).cpu().numpy()
            tested[k, te] = True
            print(f"  seed {seed} held {held} ({order[held]}): done", flush=True)
    np.savez(ARR, sessions=np.array(sess), label=y, site=site,
             scanners=np.array(order), seeds=np.arange(SEED_BASE, SEED_BASE + SEEDS),
             ce_ep10=ce[10], ce_ep25=ce[25], ce_ep50=ce[50],
             aum=aum, el2n=el2n, pred=pred, tested=tested)
    print("wrote", ARR, flush=True)


def rank_partial(x, y_, Z, idx=None):
    """Rank-partial Spearman of x,y given columns Z (S15 convention)."""
    if idx is None:
        idx = np.arange(len(x))
    rx, ry = rankdata(x[idx]), rankdata(y_[idx])
    C = np.column_stack([rankdata(z[idx]) for z in Z] + [np.ones(len(idx))])
    bx = np.linalg.lstsq(C, rx, rcond=None)[0]
    by = np.linalg.lstsq(C, ry, rcond=None)[0]
    ex, ey = rx - C @ bx, ry - C @ by
    return float(np.corrcoef(ex, ey)[0, 1])


def boot_ci(fn, n, B=2000, seed=7):
    rng = np.random.RandomState(seed)
    vals = [fn(rng.randint(0, n, n)) for _ in range(B)]
    return [round(float(np.percentile(vals, q)), 3) for q in (2.5, 97.5)]


def stage_analyze():
    d = np.load(ARR, allow_pickle=True)
    sess, y, site = d["sessions"].astype(str), d["label"], d["site"]
    prior = np.load(f"{PROV}/oasis3_cdr_participant_arrays.npz", allow_pickle=True)
    assert np.array_equal(prior["sessions"].astype(str), sess), "cohort order mismatch"
    rec = pd.read_csv(f"{PROV}/oasis3_cdr_participant_records.csv")
    assert list(rec["session"].astype(str)) == list(sess), "records order mismatch"

    tested = d["tested"]
    tested_all = tested.all(0)  # same mask every seed (fold structure fixed)
    err_freq = np.where(tested, (d["pred"] != y[None, :]), np.nan).astype(float)
    target = np.nanmean(err_freq, 0)  # per-subject CNN held-out error frequency
    M = tested_all
    prior_err = prior["err_per_seed"].mean(0)

    def crossfit(a):  # (S, sites, N) -> per-subject mean over fits then seeds
        return np.nanmean(np.nanmean(a, axis=1), axis=0)

    def singlefit(a):  # designated fit: fold 0, fold 1 for TrioTim subjects
        des = np.where(site == 0, 1, 0)
        out = a[:, des, np.arange(len(site))]  # (S, N)
        return out.mean(0)

    scores = {
        "cnn_misfit": crossfit(d["ce_ep50"]),
        "ordinary_singlefit": singlefit(d["ce_ep50"]),
        "aum": singlefit(d["aum"]),
        "el2n": singlefit(d["el2n"]),
        "prior_errors": prior_err,
    }
    secondary = {"ordinary_crossfit": crossfit(d["ce_ep50"]),
                 "aum_crossfit": crossfit(d["aum"]), "el2n_crossfit": crossfit(d["el2n"]),
                 "cnn_misfit_ep10": crossfit(d["ce_ep10"]),
                 "cnn_misfit_ep25": crossfit(d["ce_ep25"])}
    secondary["ordinary_crossfit"] = scores["cnn_misfit"]  # identical by construction

    res = {"prespec": "paper/feedback/PRESPEC_cnn18_20261010.md @ 38c6927",
           "n": int(len(y)), "n_tested": int(M.sum()),
           "fits_executed": int(np.isfinite(d["aum"]).any(-1).sum()),
           "target": "CNN held-out error frequency over 3 seeds",
           "base_acc": float(1 - np.nanmean(target[M])),
           "backbone_reproduction": {"note": "Historical comparison metrics are supplied with external results."}}

    # --- Endpoint A: incremental value
    A = {"marginal": {}}
    anyerr = (target[M] > 0).astype(int)
    for nm, s in {**scores, **secondary}.items():
        rho = float(spearmanr(s[M], target[M])[0])
        auc = float(roc_auc_score(anyerr, s[M])) if 0 < anyerr.sum() < len(anyerr) else None
        A["marginal"][nm] = {"rho": round(rho, 3),
                             "auroc_any_error": round(auc, 3) if auc else None}
    mi = scores["cnn_misfit"][M]
    tg = target[M]
    comps = [scores[c][M] for c in ("ordinary_singlefit", "aum", "el2n", "prior_errors")]
    A["partial_given_each"] = {}
    for cn, cv in zip(("ordinary_singlefit", "aum", "el2n", "prior_errors"), comps):
        pr = rank_partial(mi, tg, [cv])
        ci = boot_ci(lambda ii, cv=cv: rank_partial(mi, tg, [cv], ii), len(mi))
        A["partial_given_each"][cn] = {"partial_rho": round(pr, 3), "ci95": ci}
    prj = rank_partial(mi, tg, comps)
    cij = boot_ci(lambda ii: rank_partial(mi, tg, comps, ii), len(mi))
    A["partial_given_all_four"] = {"partial_rho": round(prj, 3), "ci95": cij,
                                   "incremental_value_claimed": bool(cij[0] > 0 or cij[1] < 0)}
    A["symmetric_ordinary_given_misfit"] = round(
        rank_partial(scores["ordinary_singlefit"][M], tg, [mi]), 3)
    z = M & (prior_err == 0)
    za = (target[z] > 0).astype(int)
    A["zero_prior_error_subgroup"] = {
        "n": int(z.sum()), "n_with_cnn_error": int(za.sum()),
        "rho": round(float(spearmanr(scores["cnn_misfit"][z], target[z])[0]), 3),
        "auroc_any_error": round(float(roc_auc_score(za, scores["cnn_misfit"][z])), 3)
        if 0 < za.sum() < len(za) else None}
    res["endpoint_A_incremental_value"] = A

    # --- Endpoint B: clinical stage
    cdr = rec["cdr_raw"].to_numpy()
    stage = np.where(cdr == 0, 0, np.where(cdr == 0.5, 1, 2))
    age = rec["age"].to_numpy()
    sex = rec["sex_male1"].to_numpy()
    mis = scores["cnn_misfit"]
    pct = 100 * rankdata(mis) / len(mis)
    per_seed_rho = [round(float(spearmanr(stage.astype(float),
                                          np.nanmean(d["ce_ep50"][k], 0))[0]), 3)
                    for k in range(SEEDS)]
    r = spearmanr(stage.astype(float), mis)
    B_ = {"misfit_vs_cdr_stage": {"rho": round(float(r.statistic), 3),
                                  "p": float(f"{r.pvalue:.3g}")},
          "per_seed_cdr_rho": per_seed_rho,
          "stage_n": [int((stage == s).sum()) for s in (0, 1, 2)],
          "stage_median_misfit_percentile": [
              round(float(np.median(pct[stage == s])), 1) for s in (0, 1, 2)]}
    # FN among impaired, tested: binomial GLM, ref = CDR>=1
    imp = M & (y == 1)
    fn_cnt = np.nansum(np.where(tested, (d["pred"] == 0) & (y[None, :] == 1), np.nan), 0)
    import statsmodels.api as sm
    df = pd.DataFrame({"fn": fn_cnt[imp], "stage": stage[imp],
                       "age_z": (age[imp] - age[imp].mean()) / age[imp].std(),
                       "sex": sex[imp]})
    B_["fn_rate_unadjusted_by_stage"] = {
        str(s): {"n": int((df.stage == s).sum()),
                 "fn_rate": round(float(df.fn[df.stage == s].mean() / 3), 3)}
        for s in sorted(df.stage.unique())}
    # impaired participants are CDR 0.5 or >=1 by construction (CDR defines the
    # label), so the only stage contrast among them is CDR 0.5 vs >=1
    Xd = np.column_stack([np.ones(len(df)), (df.stage == 1).astype(float),
                          df.age_z, df.sex])
    glm = sm.GLM(np.column_stack([df.fn, 3 - df.fn]), Xd,
                 family=sm.families.Binomial()).fit()
    names = ["intercept_cdr_ge1", "stage_cdr05_vs_ge1", "age_z", "sex_male"]
    ci = glm.conf_int()
    B_["fn_adjusted_glm"] = {
        nm: {"or": round(float(np.exp(glm.params[i])), 3),
             "ci95": [round(float(np.exp(ci[i][0])), 3), round(float(np.exp(ci[i][1])), 3)],
             "p": float(f"{glm.pvalues[i]:.3g}")}
        for i, nm in enumerate(names)}
    sexbar = float(df.sex.mean())
    B_["fn_adjusted_predicted_prob"] = {
        lab: round(float(1 / (1 + np.exp(-(glm.params[0] + d1 * glm.params[1]
                                           + sexbar * glm.params[3])))), 3)
        for lab, d1 in (("cdr05", 1), ("cdr_ge1", 0))}
    sfn = err_freq.copy()
    top = pct > 80
    fn_mask = np.where(tested, (d["pred"] == 0) & (y[None, :] == 1), False)
    B_["fn_capture_top20"] = round(float(fn_mask[:, top].sum() / max(fn_mask.sum(), 1)), 3)
    res["endpoint_B_clinical"] = B_
    del sfn

    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps(res, indent=1)[:2500], flush=True)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    os.makedirs(RES, exist_ok=True)
    st = sys.argv[1] if len(sys.argv) > 1 else "all"
    if st in ("train", "all"):
        stage_train()
    if st in ("analyze", "all"):
        stage_analyze()
