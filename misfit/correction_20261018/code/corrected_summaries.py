"""Derived corrected analyses from the per-scan rerun records (plan section 2
item 5 + addendum A2/A4/A9), mirroring the historical scripts' exact
definitions (run_clinical_multiseed.stratified_perseed / selective_perseed,
run_class_env_strata_s16, run_bootstrap_balanced, run_subgroup_media,
run_clinical_insight_panel fn-capture, enrichment deciles, ml_reliability
block A) while adding the corrected inference units.

Usage: python3 corrected_summaries.py <variant>   (v0|v1|v2|v1r|v3)
Reads  reruns/purged_<v>_perscan_local.npz (+ _dynamics_local.npz, linkage csv)
Writes results/corrected_<v>_summaries.json (aggregate only)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

COR = Path(__file__).resolve().parent.parent
N_SEEDS = 5
COVERAGES = [1.0, 0.8, 0.5, 0.3]
ALPHAS = [0.05, 0.10, 0.15, 0.20]
BUDGETS = [0.10, 0.20, 0.30, 0.50]
B_BOOT = 2000


def _rho(a, b):
    r = spearmanr(a, b)[0]
    return float(r) if not np.isnan(r) else float("nan")


def _auroc(y, s):
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None


def _sens_spec(true, pred, m):
    t, p = true[m], pred[m]
    sens = float((p[t == 1] == 1).mean()) if (t == 1).sum() else float("nan")
    spec = float((p[t == 0] == 0).mean()) if (t == 0).sum() else float("nan")
    return sens, spec


def _agg(vals):
    a = np.array([v for v in vals
                  if v is not None and not (isinstance(v, float) and np.isnan(v))],
                 float)
    if len(a) == 0:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(a.mean()), "std": float(a.std()), "n": int(len(a))}


def _threshold_for_alpha(mis_cal, err_cal, alpha):
    """Largest calibration misfit threshold whose running accepted-set error
    rate (lowest misfit first) stays <= alpha; -inf when none qualifies."""
    order = np.argsort(mis_cal)
    e = err_cal[order]
    n = len(e)
    rates = np.cumsum(e) / np.arange(1, n + 1)
    ok = np.where(rates <= alpha)[0]
    if len(ok) == 0:
        return -np.inf
    return float(mis_cal[order][ok.max()])


def seed_frames(d, label, site, ptids):
    """One dict per seed bundling tested mask, error, misfit, labels, derived
    predictions, and the optional per-scan score channels."""
    frames = []
    for s in range(N_SEEDS):
        t = d[f"tested_s{s}"].astype(bool)
        err = d[f"err_s{s}"]
        pred = np.where(err == 0, label, 1 - label)  # binary task: prediction recovered from the error indicator
        frames.append({
            "tested": t, "err": err, "misfit": d[f"misfit_s{s}"],
            "true": label, "pred": pred, "site": site, "ptid": ptids,
            "test_ce": d.get(f"test_ce_s{s}"), "energy": d.get(f"energy_s{s}"),
            "max_softmax": d.get(f"max_softmax_s{s}"),
            "maha_test": d.get(f"maha_test_s{s}"),
            "deep_ensemble5": d.get(f"deep_ensemble5_s{s}"),
        })
    return frames


def stratified(frames):
    """Per-seed stratified metrics, mirroring run_clinical_multiseed.stratified_perseed.

    Aggregates mean/std over seeds; the single-class and balanced protocol
    lists are those computed on the last seed's tested set.
    """
    keys = ["auroc_overall", "auroc_impaired", "auroc_control", "rho_overall",
            "rho_impaired", "rho_control", "balanced_accuracy", "sensitivity",
            "specificity", "auroc_excl_singleclass", "rho_excl_singleclass",
            "rho_balanced_only"]
    per = {k: [] for k in keys}
    single = balanced = prev = None
    for d in frames:
        t = d["tested"]
        mis, err, true, pred, site = (d["misfit"][t], d["err"][t], d["true"][t],
                                      d["pred"][t], d["site"][t])
        per["auroc_overall"].append(_auroc(err, mis))
        per["auroc_impaired"].append(_auroc(err[true == 1], mis[true == 1]))
        per["auroc_control"].append(_auroc(err[true == 0], mis[true == 0]))
        per["rho_overall"].append(_rho(mis, err))
        per["rho_impaired"].append(_rho(mis[true == 1], err[true == 1]))
        per["rho_control"].append(_rho(mis[true == 0], err[true == 0]))
        per["balanced_accuracy"].append(float(balanced_accuracy_score(true, pred)))
        s, sp = _sens_spec(true, pred, np.ones(len(true), bool))
        per["sensitivity"].append(s)
        per["specificity"].append(sp)
        prev = {int(e_): float(true[site == e_].mean()) for e_ in np.unique(site)}
        single = [e_ for e_, p in prev.items() if p in (0.0, 1.0) or p < 0.05 or p > 0.95]
        balanced = [e_ for e_, p in prev.items() if 0.35 <= p <= 0.65]
        keep = ~np.isin(site, single)
        bal = np.isin(site, balanced)
        per["auroc_excl_singleclass"].append(_auroc(err[keep], mis[keep]) if keep.sum() else None)
        per["rho_excl_singleclass"].append(_rho(mis[keep], err[keep]) if keep.sum() else None)
        per["rho_balanced_only"].append(_rho(mis[bal], err[bal]) if bal.sum() > 5 else None)
    out = {k: _agg(v) for k, v in per.items()}
    out["single_class_protocols"] = [int(x) for x in single]
    out["balanced_protocols"] = [int(x) for x in balanced]
    return out


def selective(frames):
    """Coverage sweep: accept the lowest-misfit fraction per coverage; report
    accuracy, sensitivity, specificity, FN rate among accepted, and the
    impaired fraction among abstained."""
    per = {cov: {k: [] for k in ["accuracy", "sensitivity", "specificity",
                                 "fn_among_accepted", "frac_abstained_impaired"]}
           for cov in COVERAGES}
    for d in frames:
        t = d["tested"]
        mis, err, true, pred = d["misfit"][t], d["err"][t], d["true"][t], d["pred"][t]
        order = np.argsort(mis)
        for cov in COVERAGES:
            n = max(1, int(round(cov * len(mis))))
            m = np.zeros(len(mis), bool)
            m[order[:n]] = True
            sens, spec = _sens_spec(true, pred, m)
            per[cov]["accuracy"].append(float(1 - err[m].mean()))
            per[cov]["sensitivity"].append(sens)
            per[cov]["specificity"].append(spec)
            per[cov]["fn_among_accepted"].append(
                float(((pred[m] == 0) & (true[m] == 1)).mean()))
            per[cov]["frac_abstained_impaired"].append(
                float((true[~m] == 1).mean()) if (~m).sum() else float("nan"))
    return {str(cov): {k: _agg(v) for k, v in per[cov].items()} for cov in COVERAGES}


def within_env(frames):
    """Within-protocol rho over (protocol x seed) cells: per-protocol table
    plus the pooled cell mean/std and positive-cell count (NaN cells dropped)."""
    envs = sorted(set(frames[0]["site"]))
    cells, table = [], {}
    for e in envs:
        vals = []
        for d in frames:
            m = d["tested"] & (d["site"] == e)
            vals.append(_rho(d["misfit"][m], d["err"][m]))
        vals = [v for v in vals if not np.isnan(v)]
        table[str(e)] = {"rho_mean": float(np.mean(vals)), "rho_std": float(np.std(vals)),
                         "n_test": int((frames[0]["site"] == e).sum())}
        cells += vals
    return {"per_protocol": table,
            "mean_within_env_rho": float(np.mean(cells)),
            "std_within_env_rho": float(np.std(cells)),
            "n_positive": int(sum(1 for v in cells if v > 0)),
            "n_cells": len(cells)}


def strata16(frames):
    """Protocol-by-class strata: cells with >= 5 tested scans and both error
    outcomes present; aggregate mean/std over estimable cells."""
    envs = sorted(set(frames[0]["site"]))
    rows, estimable = [], set()
    for d in frames:
        for e in envs:
            for cls in (0, 1):
                m = d["tested"] & (d["site"] == e) & (d["true"] == cls)
                if m.sum() >= 5 and 0 < d["err"][m].sum() < m.sum():
                    r = _rho(d["misfit"][m], d["err"][m])
                    if not np.isnan(r):
                        rows.append(r)
                        estimable.add((e, cls))
    return {"n_strata_estimable": len(estimable), "n_strata_total": 2 * len(envs),
            "n_cells": len(rows), "rho_mean": float(np.mean(rows)),
            "rho_std": float(np.std(rows)),
            "pct_positive": float(np.mean([r > 0 for r in rows]) * 100)}


def boot_cis(frames, unit_ids, B=B_BOOT, seed=0):
    """AUROC + balanced-accuracy CIs; draws over the given unit ids (acquisition
    ids or participant ids), each drawn unit carrying all its seed rows."""
    units = np.array(sorted(set(unit_ids)))
    rows_by_unit = {u: np.where(unit_ids == u)[0] for u in units}
    rng = np.random.RandomState(seed)
    draws = rng.randint(0, len(units), (B, len(units)))
    mis = np.concatenate([d["misfit"][d["tested"]] for d in frames])
    err = np.concatenate([d["err"][d["tested"]] for d in frames])
    n_per_seed = int(frames[0]["tested"].sum())
    base = np.arange(N_SEEDS)[:, None] * n_per_seed
    aurocs, dropped = [], 0
    for b in range(B):
        ridx = np.concatenate([rows_by_unit[units[i]] for i in draws[b]])
        idx = (base + ridx[None, :]).ravel()
        e = err[idx]
        if 0 < e.sum() < len(e):
            aurocs.append(roc_auc_score(e, mis[idx]))
        else:
            dropped += 1
    return {"auroc_ci95": [float(np.percentile(aurocs, 2.5)),
                           float(np.percentile(aurocs, 97.5))],
            "auroc_boot_mean": float(np.mean(aurocs)),
            "B": B, "dropped_single_class": dropped}


def fn_capture(frames):
    """Fraction of false negatives (and false positives) captured when
    flagging the top-misfit budget fraction of tested scans."""
    out = {}
    for budget in BUDGETS:
        fn_caps, fp_caps = [], []
        for d in frames:
            t = d["tested"]
            mis, err, true, pred = d["misfit"][t], d["err"][t], d["true"][t], d["pred"][t]
            k = max(1, int(round(budget * len(mis))))
            flagged = np.zeros(len(mis), bool)
            flagged[np.argsort(-mis)[:k]] = True
            fn = (pred == 0) & (true == 1)
            fp = (pred == 1) & (true == 0)
            if fn.sum():
                fn_caps.append(float(fn[flagged].sum() / fn.sum()))
            if fp.sum():
                fp_caps.append(float(fp[flagged].sum() / fp.sum()))
        out[f"{int(budget*100)}pct"] = {"fn_capture": _agg(fn_caps),
                                        "fp_capture": _agg(fp_caps)}
    return out


def deciles(frames):
    """Decile bins of min-max normalized pooled misfit: per-bin error rate
    plus strict and non-decreasing monotonicity flags."""
    mis = np.concatenate([d["misfit"][d["tested"]] for d in frames])
    err = np.concatenate([d["err"][d["tested"]] for d in frames])
    lo, hi = mis.min(), mis.max()
    nm = (mis - lo) / (hi - lo + 1e-12)
    order = np.argsort(nm, kind="stable")
    rows = []
    for i, chunk in enumerate(np.array_split(order, 10)):
        rows.append({"bin": i + 1, "n": int(len(chunk)),
                     "err_rate": float(err[chunk].mean()),
                     "mean_norm_misfit": float(nm[chunk].mean())})
    mono = all(rows[i]["err_rate"] <= rows[i + 1]["err_rate"] for i in range(9))
    return {"bins": rows, "strictly_monotone": bool(
        all(rows[i]["err_rate"] < rows[i + 1]["err_rate"] for i in range(9))),
        "monotone_nondecreasing": bool(mono)}


def panel(frames):
    """Light estimator panel: pooled rho per saved score channel (None when a
    channel is absent from the archive), plus the within-protocol CE-rank
    diagnostic explained in the emitted note."""
    keys = ["misfit", "test_ce", "energy", "max_softmax", "maha_test",
            "deep_ensemble5"]
    out = {}
    for k in keys:
        vals = []
        for d in frames:
            s = d[k]
            if s is None:
                break
            m = d["tested"] & ~np.isnan(s)
            vals.append(_rho(s[m], d["err"][m]))
        out[k] = _agg(vals) if vals else None
    # CE-rank diagnostic: test CE ranked WITHIN the scan's own held-out
    # protocol, pooled (the ENR construction; distinct from the raw pooled
    # test-CE oracle above)
    vals = []
    for d in frames:
        if d["test_ce"] is None:
            break
        t = d["tested"] & ~np.isnan(d["test_ce"])
        r = np.full(len(d["test_ce"]), np.nan)
        for e in np.unique(d["site"][t]):
            m = t & (d["site"] == e)
            from scipy.stats import rankdata
            r[m] = rankdata(d["test_ce"][m]) / m.sum()
        vals.append(_rho(r[t], d["err"][t]))
    out["ce_rank_perfold"] = _agg(vals) if vals else None
    out["note"] = ("test_ce = raw pooled test-CE oracle (ENR "
                   "structural_misfit_test lineage); "
                   "ce_rank_perfold = test CE ranked within protocol; the historical ENR "
                   "CE-rank construction remains unresolved; deep-kNN "
                   "not recomputed (listed unresolved)")
    return out


def conformal(frames, ptids):
    """Conformal-style selective blocks per alpha.

    Three constructions per seed: historical acquisition-level exchangeable
    half-splits, participant-blocked exchangeable half-splits (A9), and
    shifted calibration per test protocol, with and without excluding the
    test protocol's participants from calibration.
    """
    res = {}
    rng_hist = np.random.RandomState(0)
    rng_part = np.random.RandomState(0)
    for alpha in ALPHAS:
        acc = {k: [] for k in ["exch_err", "exch_err_blocked", "shift_err",
                               "shift_err_blocked", "exch_cov", "shift_cov"]}
        for d in frames:
            t = d["tested"]
            mis, err, site = d["misfit"][t], d["err"][t], d["site"][t]
            pt = ptids[t]
            # historical acquisition-level exchangeable (10 random splits)
            for _ in range(10):
                idx = rng_hist.permutation(len(mis))
                h = len(mis) // 2
                cal, te = idx[:h], idx[h:]
                tau = _threshold_for_alpha(mis[cal], err[cal], alpha)
                a = mis[te] <= tau
                if a.sum():
                    acc["exch_err"].append(float(err[te][a].mean()))
                    acc["exch_cov"].append(float(a.mean()))
            # participant-blocked exchangeable (A9)
            uu = np.array(sorted(set(pt)))
            for _ in range(10):
                perm = rng_part.permutation(len(uu))
                cal_u = set(uu[perm[:len(uu) // 2]])
                cal = np.isin(pt, list(cal_u))
                tau = _threshold_for_alpha(mis[cal], err[cal], alpha)
                a = (mis <= tau) & ~cal
                if a.sum():
                    acc["exch_err_blocked"].append(float(err[a].mean()))
            # shifted: historical (cal = other protocols) and blocked (cal
            # additionally excludes acquisitions of test-protocol participants)
            for p in np.unique(site):
                te_m = site == p
                if te_m.sum() < 5:
                    continue
                cal_m = ~te_m
                if cal_m.sum() < 20:
                    continue
                tau = _threshold_for_alpha(mis[cal_m], err[cal_m], alpha)
                a = te_m & (mis <= tau)
                if a.sum():
                    acc["shift_err"].append(float(err[a].mean()))
                    acc["shift_cov"].append(float(a[te_m].mean()))
                pe = set(pt[te_m])
                cal_b = cal_m & ~np.isin(pt, list(pe))
                if cal_b.sum() < 20:
                    continue
                tau = _threshold_for_alpha(mis[cal_b], err[cal_b], alpha)
                a = te_m & (mis <= tau)
                if a.sum():
                    acc["shift_err_blocked"].append(float(err[a].mean()))
        res[f"{alpha:.2f}"] = {k: _agg(v) for k, v in acc.items()}
    return res


def paired_dynamics(dyn, label, ptids, out_units):
    """Pinned A4 paired bootstrap on the training-dynamics scores."""
    tested = dyn["tested_s0"].astype(bool)
    assert all(np.array_equal(tested, dyn[f"tested_s{s}"].astype(bool))
               for s in range(N_SEEDS))
    pt = ptids[tested]
    scores = {k: np.concatenate([dyn[f"{k}_s{s}"][tested] for s in range(N_SEEDS)])
              for k in ["misfit", "aum_neg", "el2n", "forgetting"]}
    err = np.concatenate([dyn[f"err_s{s}"][tested] for s in range(N_SEEDS)])
    n_rows = int(tested.sum())
    base = np.arange(N_SEEDS)[:, None] * n_rows
    results = {}
    for unit_name, unit_ids in out_units.items():
        uids = unit_ids[tested]
        units = np.array(sorted(set(uids)))
        rows_by_unit = {u: np.where(uids == u)[0] for u in units}
        rng = np.random.RandomState(0)
        draws = rng.randint(0, len(units), (B_BOOT, len(units)))
        per_comp = {}
        for comp in ["aum_neg", "el2n", "forgetting"]:
            d_auc, d_rho, dropped = [], [], 0
            for b in range(B_BOOT):
                ridx = np.concatenate([rows_by_unit[units[i]] for i in draws[b]])
                idx = (base + ridx[None, :]).ravel()
                e = err[idx]
                if 0 < e.sum() < len(e):
                    d_auc.append(roc_auc_score(e, scores["misfit"][idx])
                                 - roc_auc_score(e, scores[comp][idx]))
                else:
                    dropped += 1
                rm = spearmanr(scores["misfit"][idx], e)[0]
                rc = spearmanr(scores[comp][idx], e)[0]
                if not (np.isnan(rm) or np.isnan(rc)):
                    d_rho.append(rm - rc)
            ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
            per_comp[comp] = {"dAUROC_ci": ci(d_auc), "dRho_ci": ci(d_rho),
                              "dAUROC_mean": float(np.mean(d_auc)),
                              "dropped_single_class": dropped,
                              "tie_dAUROC": bool(ci(d_auc)[0] <= 0 <= ci(d_auc)[1])}
        results[unit_name] = {"n_units": len(units), "common_draw_matrix": True,
                              "comparators": per_comp}
    return results


def main():
    """Load the variant's per-scan archives, compute every summary block, and
    write results/corrected_<variant>_summaries.json."""
    v = sys.argv[1]
    tag = "dedup_v3" if v == "v3" else f"purged_{v}"
    d = np.load(COR / f"reruns/{tag}_perscan_local.npz", allow_pickle=False)
    ptids = d["ptids"].astype(str)
    label = d["label"]
    site = d["site"]
    link = pd.read_csv(COR / "audit/adni_linkage_local.csv")
    copies = link.groupby("ptid").size()
    meta = link.drop_duplicates("ptid").set_index("ptid")
    frames = seed_frames(d, label, site, ptids)
    k1 = np.array([copies.get(p, 1) == 1 for p in ptids])

    out = {"variant": v, "sd_convention": "np.std ddof=0 across seeds/cells",
           "estimand": "acquisition-level; uncertainty units as labeled"}
    out["stratified"] = stratified(frames)
    out["selective"] = selective(frames)
    out["within_env_canonical"] = within_env(frames)
    out["strata16"] = strata16(frames)
    out["fn_capture"] = fn_capture(frames)
    out["reliability_deciles"] = deciles(frames)
    out["estimator_panel_light"] = panel(frames)
    out["conformal"] = conformal(frames, ptids)
    # pooled rho per seed (headline) + k=1 and copies-stratified (A2 S-a/S-c)
    pooled, pooled_k1, by_copies = [], [], {1: [], 2: [], 3: []}
    auroc_all, auroc_k1 = [], []
    for f in frames:
        t = f["tested"]
        pooled.append(_rho(f["misfit"][t], f["err"][t]))
        auroc_all.append(_auroc(f["err"][t], f["misfit"][t]))
        m1 = t & k1
        pooled_k1.append(_rho(f["misfit"][m1], f["err"][m1]))
        auroc_k1.append(_auroc(f["err"][m1], f["misfit"][m1]))
        for c in (1, 2, 3):
            mc = t & np.array([copies.get(p, 1) == c for p in ptids])
            if mc.sum() >= 20:
                by_copies[c].append(_rho(f["misfit"][mc], f["err"][mc]))
    out["pooled_rho"] = _agg(pooled)
    out["pooled_auroc"] = _agg(auroc_all)
    out["k1_only"] = {"n_scans": int(k1.sum()), "pooled_rho": _agg(pooled_k1),
                      "pooled_auroc": _agg(auroc_k1),
                      "within_env": within_env([{**f, "tested": f["tested"] & k1}
                                                for f in frames])}
    out["rho_by_n_copies"] = {str(c): _agg(v_) for c, v_ in by_copies.items()}
    # bootstrap CIs: participant unit (corrected) + acquisition unit (historical)
    out["auroc_boot"] = {
        "participant_unit": boot_cis(frames, ptids[frames[0]["tested"]]),
        "acquisition_unit_historical_convention": boot_cis(
            frames, np.arange(len(ptids))[frames[0]["tested"]]),
    }
    # subgroup (sex / age median) with participant draws
    t0 = frames[0]["tested"]
    sex = np.array([meta.loc[p, "sex"] if p in meta.index else "?" for p in ptids])
    age = np.array([float(meta.loc[p, "age"]) if p in meta.index else np.nan
                    for p in ptids])
    med = float(np.nanmedian(age[t0]))
    sub = {}
    for name, mask in [("female", sex == "F"), ("male", sex == "M"),
                       ("younger", age < med), ("older", age >= med)]:
        vals = [_auroc(f["err"][f["tested"] & mask], f["misfit"][f["tested"] & mask])
                for f in frames]
        sub[name] = _agg(vals)
    sub["age_median_note"] = f"median over acquisitions = {med:.1f} (scan-level)"
    out["subgroup"] = sub
    # paired training-dynamics bootstrap (v3's driver stores aggregates only)
    dyn_path = COR / f"reruns/{tag}_dynamics_local.npz"
    if dyn_path.exists():
        dyn = np.load(dyn_path, allow_pickle=False)
        out["paired_dynamics"] = paired_dynamics(
            dyn, label, ptids,
            {"participant": ptids, "acquisition_historical_convention":
             np.arange(len(ptids)).astype(str)})
    else:
        out["paired_dynamics"] = None
    (COR / f"results/corrected_{v}_summaries.json").write_text(
        json.dumps(out, indent=2))
    print(f"[{v}] summaries written")


if __name__ == "__main__":
    main()
