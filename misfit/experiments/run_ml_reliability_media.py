"""ML-reliability / explainability experiments for the MedIA paper.

Runs ALL FOUR candidate directions; only those that survive honest scrutiny
are kept downstream. Reuses the honest harness low-level functions verbatim
(same IOP recipe; binary per-subject-error target; ADNI N=731 / 8 protocols;
OASIS-1 N=226 / 4 quartiles; 5 seeds). Per-dataset sanity anchors for the
deployable misfit rho are supplied externally and printed next to each
seed's realized rho.

Blocks
------
A. CONFORMAL-CALIBRATED ABSTENTION under shift. Calibrate an abstention
   threshold on misfit to hit a target accepted-set error alpha; measure the
   realized accepted-set error in two regimes: (i) EXCHANGEABLE (random
   calibration/test split) and (ii) SHIFTED (calibrate on held-in protocols,
   test on a held-out protocol). Honest question: does the guarantee survive the
   protocol shift, and does a little target-protocol data fix it (few-shot)?

B. FAILURE-DETECTOR AUROC. Re-score per-subject error prediction as
   misclassification detection: AUROC / AUPRC of misfit vs. baselines
   (ERM loss, conformal quantile, max-softmax, energy, MC dropout). Plus ROC
   points for the figure.

C. PER-REGION MISFIT ATTRIBUTION. Input-gradient attribution of the per-subject
   misfit, aggregated over high-misfit subjects. Seed-stability of the
   per-feature importance ranking decides whether it is reportable. OASIS
   features are Schaefer ROIs (directly interpretable).

D. PER-PROGRAM MISFIT DECOMPOSITION. Ablate each program (zero its loading
   column) and measure the misfit change; aggregate over high-misfit subjects.
   Seed-stability of the per-program importance decides whether the operator
   structure buys explainability the plain loss cannot. Expected to be weak
   because the loadings are not learned -- tested honestly either way.

Numerics (repo rules): float64 covariance; never torch.inverse
(linalg.solve / SVD pinv via the reused helpers); condition-number guards.

Output: misfit/results/ml_reliability.{json,md}
"""

from __future__ import annotations
from misfit.paths import reporting_reference

from misfit.paths import resolve_path as _resolve_path

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as sp_stats
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

sys.path.insert(0, _resolve_path("<REPO_ROOT>"))

from misfit.experiments.run_honest_estimator_table import (  # noqa: E402
    DEVICE,
    MC_DROPOUT,
    MC_PASSES,
    N_PROGRAMS,
    N_SEEDS,
    SEED_BASE,
    _predict_loadings,
    _rho,
    _train_erm,
    _train_fire,
    load_adni,
    load_oasis_ds,
)
from misfit.core.utils import seed_everything  # noqa: E402

warnings.filterwarnings("ignore")

OUT = Path(_resolve_path("<REPO_ROOT>/misfit/results"))
OUT.mkdir(parents=True, exist_ok=True)

ALPHAS = [0.05, 0.10, 0.15, 0.20]
HIGH_MISFIT_FRAC = 0.30  # "flagged unreliable" = top 30% misfit


# ===========================================================================
# Shared per-fold collection: per-subject deployable misfit, binary error,
# site id, test-time baseline scores, input-grad attribution, program ablation.
# Mirrors run_seed's leave-one-environment-out construction exactly.
# ===========================================================================
def collect_seed(features, targets, site_ids, seed):
    """One leave-one-environment-out pass at a fixed seed, collecting
    everything blocks A-D consume.

    Training side (averaged over the folds a subject trains in): per-subject
    CE misfit, ERM-loss baseline, |input-gradient| attribution of the misfit,
    per-program ablation deltas. Test side (single held-out fold per subject):
    binary error plus test-time UQ baselines (1 - max softmax, energy,
    MC-dropout predictive variance).

    Returns:
        Dict of per-subject arrays; "conformal_q" is the rank transform of
        misfit.
    """
    n_sites = int(site_ids.max().item()) + 1
    dim = features.shape[1]
    N = features.shape[0]
    seed_everything(seed)

    subject_correct = torch.full((N,), float("nan"))
    tested = torch.zeros(N, dtype=torch.bool)

    # training-side accumulators (mean over in-train folds)
    mis_sum = np.zeros(N); mis_cnt = np.zeros(N)
    erm_sum = np.zeros(N)
    attr_sum = np.zeros((N, dim))           # |input grad| of misfit
    prog_sum = np.zeros((N, N_PROGRAMS))    # per-program ablation delta
    # test-side single held-out scores
    max_softmax = np.full(N, np.nan)
    energy = np.full(N, np.nan)
    mc_drop = np.full(N, np.nan)

    for held_out in range(n_sites):
        train_mask = site_ids != held_out
        test_mask = site_ids == held_out
        if test_mask.sum() < 5 or train_mask.sum() < 20:
            continue
        train_feat = features[train_mask]; train_tgt = targets[train_mask]
        train_sites = site_ids[train_mask]
        test_feat = features[test_mask]; test_tgt = targets[test_mask]
        unique = train_sites.unique()
        smap = {int(s_): i for i, s_ in enumerate(unique)}
        remapped = torch.tensor([smap[int(s_)] for s_ in train_sites])
        nts = len(unique)
        ti_train = torch.where(train_mask)[0].numpy()
        ti_test = torch.where(test_mask)[0].numpy()

        model, loadings = _train_fire(train_feat, train_tgt, remapped, nts, dim, seed)
        model.eval()

        # --- training-side misfit (per-subject CE) ---
        with torch.no_grad():
            tr_pred = model(train_feat, loadings, remapped)
            tr_loss = F.cross_entropy(tr_pred, train_tgt.long(), reduction="none").numpy()
        mis_sum[ti_train] += tr_loss; mis_cnt[ti_train] += 1.0

        # --- Block C: input-gradient attribution of misfit ---
        tf_g = train_feat.clone().requires_grad_(True)
        gp = model(tf_g, loadings.detach(), remapped)
        ce = F.cross_entropy(gp, train_tgt.long(), reduction="none")
        g = torch.autograd.grad(ce.sum(), tf_g)[0].detach().abs().numpy()
        attr_sum[ti_train] += g

        # --- Block D: per-program ablation (zero each loading column) ---
        with torch.no_grad():
            for k in range(N_PROGRAMS):
                lk = loadings.clone(); lk[:, k] = 0.0
                ce_k = F.cross_entropy(model(train_feat, lk, remapped),
                                       train_tgt.long(), reduction="none").numpy()
                prog_sum[ti_train, k] += (ce_k - tr_loss)

        # --- ERM loss (training-side, baseline) ---
        erm = _train_erm(train_feat, train_tgt, dim, seed)
        erm.eval()
        with torch.no_grad():
            erm_tr = F.cross_entropy(erm(train_feat.to(DEVICE)),
                                     train_tgt.long().to(DEVICE),
                                     reduction="none").cpu().numpy()
        erm_sum[ti_train] += erm_tr

        # --- held-out evaluation: error target + test-time UQ ---
        tl = _predict_loadings(train_feat, loadings, test_feat)
        test_env = torch.zeros(test_feat.shape[0], dtype=torch.long)
        with torch.no_grad():
            tp = model(test_feat, tl, test_env)
            pl = tp.argmax(dim=-1)
            probs_te = torch.softmax(tp, dim=-1)
        subject_correct[ti_test] = (pl == test_tgt.long()).float()
        tested[ti_test] = True
        max_softmax[ti_test] = (1.0 - probs_te.max(dim=-1).values.numpy())
        energy[ti_test] = (-torch.logsumexp(tp, dim=-1)).numpy()

        # MC dropout (test-time UQ)
        mc = _train_erm(train_feat, train_tgt, dim, seed, dropout=MC_DROPOUT)
        mc.train()
        mc_probs = []
        with torch.no_grad():
            Xte_d = test_feat.to(DEVICE)
            for _ in range(MC_PASSES):
                mc_probs.append(torch.softmax(mc(Xte_d), dim=-1).cpu())
        mc_drop[ti_test] = torch.stack(mc_probs).var(0).mean(-1).numpy()

    cnt = np.maximum(mis_cnt, 1.0)
    misfit = mis_sum / cnt
    erm_loss = erm_sum / cnt
    attr = attr_sum / cnt[:, None]
    prog = prog_sum / cnt[:, None]
    err = 1.0 - subject_correct.numpy()
    t = tested.numpy()
    return {
        "tested": t, "err": err, "site": site_ids.numpy(),
        "misfit": misfit, "erm_loss": erm_loss,
        "conformal_q": _per_fold_rank(misfit),  # rank transform of misfit
        "max_softmax": max_softmax, "energy": energy, "mc_dropout": mc_drop,
        "attr": attr, "prog": prog,
    }


def _per_fold_rank(x):
    r = np.full_like(x, np.nan, dtype=np.float64)
    m = ~np.isnan(x)
    r[m] = sp_stats.rankdata(x[m]) / m.sum()
    return r


# ===========================================================================
# Block A -- conformal-calibrated abstention under shift
# ===========================================================================
def _threshold_for_alpha(mis_cal, err_cal, alpha):
    """Largest misfit threshold tau s.t. accepted-set error on cal <= alpha.

    Accept = misfit <= tau. Sort ascending; take the longest low-misfit prefix
    whose cumulative error rate <= alpha. Conformal (n+1)/n plug-in correction.
    """
    order = np.argsort(mis_cal)
    e = err_cal[order]
    n = len(e)
    cum = np.cumsum(e)
    rates = cum / np.arange(1, n + 1)
    ok = np.where(rates <= alpha)[0]
    if len(ok) == 0:
        return -np.inf, 0.0  # accept no one
    cut = ok.max()
    tau = mis_cal[order][cut]
    return float(tau), float((cut + 1) / n)


def block_a_conformal(data_by_seed):
    """Exchangeable vs shifted calibration; realized accepted-set error."""
    res = {}
    rng = np.random.RandomState(0)
    for alpha in ALPHAS:
        exch_err, exch_cov, shift_err, shift_cov, fewshot_err = [], [], [], [], []
        for d in data_by_seed:
            t = d["tested"]
            mis = d["misfit"][t]; err = d["err"][t]; site = d["site"][t]
            # (i) exchangeable: random 50/50 cal/test, 10 repeats
            for _ in range(10):
                idx = rng.permutation(len(mis))
                h = len(mis) // 2
                cal, te = idx[:h], idx[h:]
                tau, _ = _threshold_for_alpha(mis[cal], err[cal], alpha)
                acc = mis[te] <= tau
                if acc.sum() > 0:
                    exch_err.append(float(err[te][acc].mean()))
                    exch_cov.append(float(acc.mean()))
            # (ii) shifted: leave-one-protocol-out calibration
            for p in np.unique(site):
                cal = site != p; te = site == p
                if te.sum() < 5 or cal.sum() < 20:
                    continue
                tau, _ = _threshold_for_alpha(mis[cal], err[cal], alpha)
                acc = mis[te] <= tau
                if acc.sum() > 0:
                    shift_err.append(float(err[te][acc].mean()))
                    shift_cov.append(float(acc.mean()))
                # (iii) few-shot: add 20 labelled target-protocol pts to cal
                tgt_idx = np.where(te)[0]
                if len(tgt_idx) > 25:
                    fs = rng.choice(tgt_idx, 20, replace=False)
                    rest = np.setdiff1d(tgt_idx, fs)
                    cal_mis = np.concatenate([mis[cal], mis[fs]])
                    cal_err = np.concatenate([err[cal], err[fs]])
                    tau2, _ = _threshold_for_alpha(cal_mis, cal_err, alpha)
                    acc2 = mis[rest] <= tau2
                    if acc2.sum() > 0:
                        fewshot_err.append(float(err[rest][acc2].mean()))
        res[f"{alpha:.2f}"] = {
            "target": alpha,
            "exch_err": float(np.mean(exch_err)) if exch_err else None,
            "exch_cov": float(np.mean(exch_cov)) if exch_cov else None,
            "shift_err": float(np.mean(shift_err)) if shift_err else None,
            "shift_cov": float(np.mean(shift_cov)) if shift_cov else None,
            "fewshot_err": float(np.mean(fewshot_err)) if fewshot_err else None,
            # dispersion over calibration replicates (added 2026-10-08; the means
            # above are untouched and the RNG stream is not consumed by these)
            "exch_err_std": float(np.std(exch_err)) if exch_err else None,
            "exch_cov_std": float(np.std(exch_cov)) if exch_cov else None,
            "shift_err_std": float(np.std(shift_err)) if shift_err else None,
            "shift_cov_std": float(np.std(shift_cov)) if shift_cov else None,
            "fewshot_err_std": float(np.std(fewshot_err)) if fewshot_err else None,
            "n_replicates": {"exch_seed_x_split": len(exch_err),
                             "shift_seed_x_heldenv": len(shift_err)},
        }
    return res


# ===========================================================================
# Block B -- failure-detector AUROC / AUPRC
# ===========================================================================
B_ESTIMATORS = ["misfit", "erm_loss", "conformal_q", "max_softmax", "energy", "mc_dropout"]


def block_b_auroc(data_by_seed):
    """Failure-detection AUROC/AUPRC per estimator and seed, plus subsampled
    ROC points (first seed) for the figure."""
    res = {}
    roc_pts = {}
    for key in B_ESTIMATORS:
        aurocs, auprcs, rhos = [], [], []
        for d in data_by_seed:
            t = d["tested"] & ~np.isnan(d[key])
            s = d[key][t]; e = d["err"][t]
            if e.sum() == 0 or e.sum() == len(e):
                continue
            # All B_ESTIMATORS are stored error-oriented by construction
            # (misfit/erm_loss = loss, conformal_q = loss rank, max_softmax =
            # 1 - maxprob, energy = -logsumexp, mc_dropout = predictive variance:
            # higher => more error). Score directly; do NOT pick the orientation
            # from the evaluation labels, which would leak the target into AUROC.
            r = _rho(s, e)  # reported for context only, not used to orient
            aurocs.append(roc_auc_score(e, s))
            auprcs.append(average_precision_score(e, s))
            rhos.append(r)
        res[key] = {
            "auroc": float(np.mean(aurocs)) if aurocs else None,
            "auroc_std": float(np.std(aurocs)) if aurocs else None,
            "auprc": float(np.mean(auprcs)) if auprcs else None,
            "auprc_std": float(np.std(auprcs)) if auprcs else None,
            "rho": float(np.nanmean(rhos)) if rhos else None,
            "auroc_per_seed": [float(x) for x in aurocs],
            "auprc_per_seed": [float(x) for x in auprcs],
        }
    # ROC curve points (first seed, misfit + max_softmax + energy) for the figure
    d0 = data_by_seed[0]
    for key in ["misfit", "max_softmax", "energy"]:
        t = d0["tested"] & ~np.isnan(d0[key])
        s = d0[key][t]; e = d0["err"][t]
        # error-oriented by construction (see block_b_auroc); no label-based flip
        fpr, tpr, _ = roc_curve(e, s)
        # subsample to <= 50 points
        k = max(1, len(fpr) // 50)
        roc_pts[key] = {"fpr": fpr[::k].tolist(), "tpr": tpr[::k].tolist()}
    return res, roc_pts


# ===========================================================================
# Block C -- per-region misfit attribution + seed-stability
# ===========================================================================
def block_c_attribution(data_by_seed, feature_names=None):
    """Seed-stability of per-feature misfit attribution.

    Per seed: importance = normalized mean |input gradient| over the top
    HIGH_MISFIT_FRAC subjects. Stability = mean pairwise Spearman across seeds
    and top-10 Jaccard; also reports the seed-mean top features.
    """
    # per-seed importance vector = mean |attr| over high-misfit subjects
    vecs = []
    for d in data_by_seed:
        t = d["tested"]
        mis = d["misfit"][t]; attr = d["attr"][t]
        thr = np.quantile(mis, 1.0 - HIGH_MISFIT_FRAC)
        hi = mis >= thr
        if hi.sum() < 3:
            continue
        v = attr[hi].mean(0)
        vecs.append(v / (v.sum() + 1e-12))
    vecs = np.array(vecs)
    # cross-seed stability: mean pairwise Spearman of importance vectors
    rhos = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            rhos.append(sp_stats.spearmanr(vecs[i], vecs[j]).correlation)
    # top-10 feature Jaccard across seeds
    dim = vecs.shape[1]
    topk = min(10, dim)
    tops = [set(np.argsort(v)[-topk:]) for v in vecs]
    jacc = []
    for i in range(len(tops)):
        for j in range(i + 1, len(tops)):
            u = len(tops[i] | tops[j]); inter = len(tops[i] & tops[j])
            jacc.append(inter / u if u else 0.0)
    mean_imp = vecs.mean(0)
    top_idx = np.argsort(mean_imp)[-topk:][::-1]
    top = [{"feature": int(ix),
            "name": (feature_names[ix] if feature_names is not None and ix < len(feature_names) else f"feat_{ix}"),
            "importance": float(mean_imp[ix])} for ix in top_idx]
    return {
        "stability_spearman": float(np.nanmean(rhos)) if rhos else None,
        "top10_jaccard": float(np.mean(jacc)) if jacc else None,
        "n_features": int(dim),
        "top_features": top,
    }


# ===========================================================================
# Block D -- per-program decomposition + seed-stability
# ===========================================================================
def block_d_programs(data_by_seed):
    """Seed-stability of the per-program ablation decomposition over
    high-misfit subjects (pairwise Spearman; argmax-program consistency)."""
    vecs = []
    argmax_progs = []
    for d in data_by_seed:
        t = d["tested"]
        mis = d["misfit"][t]; prog = d["prog"][t]
        thr = np.quantile(mis, 1.0 - HIGH_MISFIT_FRAC)
        hi = mis >= thr
        if hi.sum() < 3:
            continue
        v = prog[hi].mean(0)  # mean misfit-change per program
        vecs.append(v)
        argmax_progs.append(int(np.argmax(v)))
    vecs = np.array(vecs)
    rhos = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            rhos.append(sp_stats.spearmanr(vecs[i], vecs[j]).correlation)
    # is the top program the same across seeds?
    from collections import Counter
    c = Counter(argmax_progs)
    top_prog, top_cnt = c.most_common(1)[0]
    return {
        "stability_spearman": float(np.nanmean(rhos)) if rhos else None,
        "n_programs": int(vecs.shape[1]),
        "per_seed_importance": vecs.tolist(),
        "argmax_program_per_seed": argmax_progs,
        "top_program_consistency": f"{top_cnt}/{len(argmax_progs)}",
    }


# ===========================================================================
# Driver
# ===========================================================================
def run_dataset(name, loader):
    """Collect per-seed data once and run blocks A-D on one cohort."""
    t0 = time.time()
    features, targets, site_ids = loader()
    print(f"[{name}] N={features.shape[0]} d={features.shape[1]} "
          f"E={int(site_ids.max())+1}", flush=True)
    data_by_seed = []
    for s in range(N_SEEDS):
        seed = SEED_BASE + s
        d = collect_seed(features, targets, site_ids, seed)
        t = d["tested"]
        sanity = _rho(d["misfit"][t], d["err"][t])
        print(f"  [{name}] seed {seed}: misfit rho={sanity:.3f} "
              f"(anchor {(reporting_reference('misfit_anchor', required=False) or {}).get(name)}) err_rate={d['err'][t].mean():.3f}",
              flush=True)
        data_by_seed.append(d)

    fnames = None  # ROI names if available (OASIS Schaefer); kept generic otherwise
    a = block_a_conformal(data_by_seed)
    print(f"  [{name}] Block A (conformal) done", flush=True)
    b, roc = block_b_auroc(data_by_seed)
    print(f"  [{name}] Block B (AUROC) done: misfit AUROC={b['misfit']['auroc']:.3f}",
          flush=True)
    c = block_c_attribution(data_by_seed, fnames)
    print(f"  [{name}] Block C (attribution) stability rho="
          f"{c['stability_spearman']}", flush=True)
    dd = block_d_programs(data_by_seed)
    print(f"  [{name}] Block D (programs) stability rho="
          f"{dd['stability_spearman']} top-prog {dd['top_program_consistency']}",
          flush=True)
    return {
        "n": int(features.shape[0]), "dim": int(features.shape[1]),
        "misfit_rho_per_seed": [float(_rho(d["misfit"][d["tested"]], d["err"][d["tested"]]))
                                for d in data_by_seed],
        "block_a_conformal": a, "block_b_auroc": b, "roc_points": roc,
        "block_c_attribution": c, "block_d_programs": dd,
        "wall_s": round(time.time() - t0, 1),
    }


def main():
    result = {}
    result["adni"] = run_dataset("adni", load_adni)
    result["oasis"] = run_dataset("oasis", load_oasis_ds)
    (OUT / "ml_reliability.json").write_text(json.dumps(result, indent=2))
    _write_md(result)
    print(f"\nSaved {OUT / 'ml_reliability.json'} and ml_reliability.md", flush=True)


def _write_md(result):
    """Render the markdown summary tables for both cohorts."""
    L = ["# MedIA ML-reliability / explainability experiments", "",
         "Honest IOP harness (binary per-subject-error target, leave-one-environment-out, "
         f"{N_SEEDS} seeds).", ""]
    for name in ["adni", "oasis"]:
        r = result[name]
        L += [f"## {name.upper()} (N={r['n']}, d={r['dim']})",
              f"misfit rho per seed: {[round(x,3) for x in r['misfit_rho_per_seed']]}", ""]
        # Block A
        L += ["### Block A -- conformal abstention (realized accepted-set error)",
              "| target alpha | exch err | exch cov | shift err | shift cov | few-shot err |",
              "|---|---|---|---|---|---|"]
        for k, v in r["block_a_conformal"].items():
            def f(x): return f"{x:.3f}" if x is not None else "--"
            L.append(f"| {v['target']:.2f} | {f(v['exch_err'])} | {f(v['exch_cov'])} | "
                     f"{f(v['shift_err'])} | {f(v['shift_cov'])} | {f(v['fewshot_err'])} |")
        L.append("")
        # Block B
        L += ["### Block B -- failure-detector AUROC",
              "| estimator | AUROC | AUPRC | rho |", "|---|---|---|---|"]
        for key in B_ESTIMATORS:
            v = r["block_b_auroc"][key]
            def f(x): return f"{x:.3f}" if x is not None else "--"
            L.append(f"| {key} | {f(v['auroc'])} | {f(v['auprc'])} | {f(v['rho'])} |")
        L.append("")
        # Block C
        c = r["block_c_attribution"]
        L += ["### Block C -- per-region attribution",
              f"seed-stability Spearman = **{c['stability_spearman']}**, "
              f"top-10 Jaccard = {c['top10_jaccard']} (of {c['n_features']} features)",
              "top features: " + ", ".join(f"{t['name']}({t['importance']:.3f})"
                                            for t in c["top_features"][:5]), ""]
        # Block D
        dd = r["block_d_programs"]
        L += ["### Block D -- per-program decomposition",
              f"seed-stability Spearman = **{dd['stability_spearman']}**, "
              f"top-program consistency = {dd['top_program_consistency']}, "
              f"argmax program per seed = {dd['argmax_program_per_seed']}", ""]
    (OUT / "ml_reliability.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
