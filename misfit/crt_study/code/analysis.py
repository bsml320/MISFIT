"""CRT study: frozen endpoint, uncertainty, and secondary diagnostics.

Primary endpoint: within-environment loss-gap prediction MSE
  E_gk(M) = (1/C(n,2)) sum_{i<j} (u_i - u_j)^2 = (2/(n-1)) sum_i (u_i - ubar)^2,
  u = q - q_hat(M), averaged EQUALLY over primary environments and confirmation
  repetitions. Delta_M = E(M) - E(M3); positive favors the curvature model.
Uncertainty: participant-block bootstrap (resample participants within each
environment, keep each participant's full method x repetition vector), B=2000;
repetition-level spread; crossed participant x repetition bootstrap (few-block
caveat). Ties: |gap| <= TIE_TOL counts as a tie, excluded from sign-based
inversion counts and reported separately.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau, spearmanr

TIE_TOL = 1e-6
B_BOOT = 2000
BOOT_SEED = 20261018


def pair_mse_naive(u):
    n = len(u)
    s = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            s += (u[i] - u[j]) ** 2
    return s / (n * (n - 1) / 2)


def pair_mse(u):
    n = len(u)
    if n < 2:
        return np.nan
    return float(2.0 * ((u - u.mean()) ** 2).sum() / (n - 1))


def endpoint(resid, env, rep, envs, reps):
    """resid: dict method -> array over rows; returns E[method][g][k] and means."""
    out = {}
    for m, u in resid.items():
        tab = np.full((len(envs), len(reps)), np.nan)
        for gi, g in enumerate(envs):
            for ki, k in enumerate(reps):
                sel = (env == g) & (rep == k)
                tab[gi, ki] = pair_mse(u[sel])
        out[m] = tab
    return out


def summarize(tabs):
    return {m: float(np.nanmean(t)) for m, t in tabs.items()}


def boot_participants(df, resid, envs, reps, B=B_BOOT, seed=BOOT_SEED,
                      also_reps=False):
    """Percentile CIs for Delta_M = E(M) - E(M3) under participant-block
    resampling (optionally crossed with repetition resampling)."""
    rng = np.random.RandomState(seed)
    env = df["env"].to_numpy()
    rep = df["rep"].to_numpy()
    uid = df["uid"].to_numpy()
    methods = list(resid)
    draws = {m: np.empty(B) for m in methods if m != "M3"}
    cell_uids = {g: np.unique(uid[env == g]) for g in envs}
    row_map = {g: [np.where((env == g) & (uid == u_))[0]
                   for u_ in cell_uids[g]] for g in envs}
    for b in range(B):
        rows = []
        for g in envs:
            uu = cell_uids[g]
            pick = rng.choice(len(uu), len(uu), replace=True)
            rows.extend(row_map[g][p] for p in pick)
        rows = np.concatenate(rows)
        use_reps = (list(np.array(reps)[rng.randint(0, len(reps), len(reps))])
                    if also_reps else reps)
        sub_env, sub_rep = env[rows], rep[rows]
        e3 = _mean_E(resid["M3"][rows], sub_env, sub_rep, envs, use_reps)
        for m in draws:
            em = _mean_E(resid[m][rows], sub_env, sub_rep, envs, use_reps)
            draws[m][b] = em - e3
    out = {}
    for m, d in draws.items():
        ok = np.isfinite(d)
        out[m] = {"est": None,
                  "ci95": [float(np.nanpercentile(d[ok], 2.5)) if ok.any() else float("nan"),
                           float(np.nanpercentile(d[ok], 97.5)) if ok.any() else float("nan")],
                  "frac_degenerate": float(np.mean(~ok))}
    return out


def _mean_E(u, env, rep, envs, reps):
    vals = []
    for g in envs:
        for k in reps:
            x = u[(env == g) & (rep == k)]
            if len(x) >= 2:
                vals.append(pair_mse(x))
    return float(np.mean(vals))


def per_rep_delta(resid, env, rep, envs, reps):
    out = {}
    for m in resid:
        if m == "M3":
            continue
        out[m] = {str(k): float(np.nanmean(
            [pair_mse(resid[m][(env == g) & (rep == k)])
             - pair_mse(resid["M3"][(env == g) & (rep == k)])
             for g in envs])) for k in reps}
    return out


def influence_loo_env(resid, env, rep, envs, reps):
    out = {}
    for g_out in envs:
        keep = [g for g in envs if g != g_out]
        out[g_out] = {m: float(np.nanmean(
            [pair_mse(resid[m][(env == g) & (rep == k)])
             - pair_mse(resid["M3"][(env == g) & (rep == k)])
             for g in keep for k in reps])) for m in resid if m != "M3"}
    return out


def rank_metrics(q, qhat, d_source, env, rep, envs, reps):
    """Within-cell rank preservation, inversions, discrimination, correlation."""
    sp, kt, inv_prev, tie_frac, ap = [], [], [], [], []
    for g in envs:
        for k in reps:
            sel = (env == g) & (rep == k)
            qq, qh, dd = q[sel], qhat[sel], d_source[sel]
            n = len(qq)
            if n < 3:
                continue
            sp.append(spearmanr(qh, qq)[0])
            kt.append(kendalltau(qh, qq)[0])
            iu, ju = np.triu_indices(n, 1)
            r_ij = qq[iu] - qq[ju]
            d_ij = dd[iu] - dd[ju]
            tie = (np.abs(r_ij) <= TIE_TOL) | (np.abs(d_ij) <= TIE_TOL)
            inv = (np.sign(r_ij) != np.sign(d_ij)) & ~tie
            tie_frac.append(float(tie.mean()))
            inv_prev.append(float(inv[~tie].mean()) if (~tie).sum() else np.nan)
            g_ij = (qh[iu] - qh[ju])
            score = -np.sign(d_ij) * g_ij          # predicted move against source order
            ap.append(_avg_precision(inv[~tie], score[~tie]))
    return {"spearman": float(np.nanmean(sp)), "kendall": float(np.nanmean(kt)),
            "inversion_prevalence": float(np.nanmean(inv_prev)),
            "tie_fraction": float(np.nanmean(tie_frac)),
            "inversion_avg_precision": float(np.nanmean(ap))}


def _avg_precision(ylab, score):
    if ylab.sum() == 0 or ylab.sum() == len(ylab):
        return np.nan
    order = np.argsort(-score)
    y = ylab[order]
    prec = np.cumsum(y) / (np.arange(len(y)) + 1)
    return float((prec * y).sum() / y.sum())
