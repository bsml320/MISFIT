"""CRT study: development-only calibration models M0-M3.

Observations are (probe, repetition) rows. Outcome q = target final CE; all
models predict q_hat = L + f(features) where f is ridge regression on a fixed
natural-cubic-spline expansion (5 knots at development quantiles .1/.3/.5/.7/.9
-> 4 columns per continuous feature) plus linear sex. Feature sets:
  M0: f = 0
  M1: B = {L, first-refit CE, mean AUM(-), mean EL2N, refit CE std, age} + sex
  M2: B + b (matched gradient-only transport feature, same spline basis)
  M3: B + b + a (curvature transport feature, same spline basis)
Fitting objective matches the primary endpoint: outcome and features are
centered within (environment, repetition) blocks before ridge (so block
intercepts can never masquerade as rank-transfer skill). Ridge lambda per model
and the damping constant c (M3) are tuned by participant-blocked GroupKFold on
development data only, scored by the within-block centered pair-MSE. Everything
(knots, standardization, lambda, c, coefficients) is frozen before confirmation.
"""
from __future__ import annotations

import numpy as np

RIDGE_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
KNOT_QS = (0.1, 0.3, 0.5, 0.7, 0.9)
N_FOLDS = 5
B_CONT = ("L", "ce_first", "aum_neg", "el2n", "ce_std", "age")


def ncs_basis(x, knots):
    """Natural cubic spline basis: x plus K-2 restricted cubic columns."""
    k = np.asarray(knots, float)
    K = len(k)
    def d(j):
        return (np.maximum(x - k[j], 0) ** 3
                - np.maximum(x - k[-1], 0) ** 3) / (k[-1] - k[j])
    cols = [x] + [d(j) - d(K - 2) for j in range(K - 2)]
    return np.column_stack(cols)


def fit_basis(dev_vals):
    knots = np.quantile(dev_vals, KNOT_QS)
    if len(np.unique(knots)) < len(knots):
        knots = np.unique(knots)
    return knots


def design(tbl, spec_cols, bases, stand=None):
    """Expanded design matrix for the given continuous columns + sex."""
    mats = [ncs_basis(tbl[c].to_numpy(float), bases[c]) for c in spec_cols]
    X = np.column_stack(mats + [tbl["sex_male1"].to_numpy(float)[:, None]])
    if stand is None:
        stand = (X.mean(0), X.std(0) + 1e-12)
    return (X - stand[0]) / stand[1], stand


def center_blocks(v, blocks):
    out = v.astype(float).copy()
    for b in np.unique(blocks):
        m = blocks == b
        out[m] -= out[m].mean(0)
    return out


def ridge_fit(Xc, yc, lam):
    P = Xc.shape[1]
    A = Xc.T @ Xc + lam * np.eye(P)
    return np.linalg.solve(A, Xc.T @ yc)


def block_pair_mse(u, blocks):
    """Mean over blocks of (2/(n-1)) * sum (u - ubar)^2; needs n >= 2."""
    vals = []
    for b in np.unique(blocks):
        x = u[blocks == b]
        if len(x) >= 2:
            vals.append(2.0 * ((x - x.mean()) ** 2).sum() / (len(x) - 1))
    return float(np.mean(vals))


def group_folds(uids, n_folds=N_FOLDS, seed=7):
    uu = np.unique(uids)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(uu))
    fold_of = {u: perm[i] % n_folds for i, u in enumerate(uu)}
    return np.array([fold_of[u] for u in uids])


def cv_score(tbl, cols, bases, lam, extra=None):
    """Participant-blocked CV of ridge on centered residual q - L."""
    y = tbl["q"].to_numpy(float) - tbl["L"].to_numpy(float)
    blocks = (tbl["env"].astype(str) + "|" + tbl["rep"].astype(str)).to_numpy()
    folds = group_folds(tbl["uid"].to_numpy())
    scores = []
    for f in range(N_FOLDS):
        tr, te = folds != f, folds == f
        X, stand = design(tbl[tr], cols, bases)
        yc = center_blocks(y[tr], blocks[tr])
        Xc = np.column_stack([center_blocks(X[:, j], blocks[tr])
                              for j in range(X.shape[1])])
        beta = ridge_fit(Xc, yc, lam)
        Xte, _ = design(tbl[te], cols, bases, stand)
        u = y[te] - Xte @ beta
        scores.append(block_pair_mse(u, blocks[te]))
    return float(np.mean(scores))


def tune_and_fit(tbl, cols):
    """Tune lambda on the dev table; fit final coefficients on all dev rows."""
    bases = {c: fit_basis(tbl[c].to_numpy(float)) for c in cols}
    best = min(RIDGE_GRID, key=lambda lam: cv_score(tbl, cols, bases, lam))
    y = tbl["q"].to_numpy(float) - tbl["L"].to_numpy(float)
    blocks = (tbl["env"].astype(str) + "|" + tbl["rep"].astype(str)).to_numpy()
    X, stand = design(tbl, cols, bases)
    yc = center_blocks(y, blocks)
    Xc = np.column_stack([center_blocks(X[:, j], blocks)
                          for j in range(X.shape[1])])
    beta = ridge_fit(Xc, yc, best)
    return {"cols": list(cols), "bases": {c: bases[c].tolist() for c in cols},
            "stand": [stand[0].tolist(), stand[1].tolist()],
            "lam": best, "beta": beta.tolist(),
            "cv": cv_score(tbl, cols, bases, best)}


def cv_predictions(tbl, cols, lam=None):
    """Participant-blocked CV predictions of q for one model spec (used by the
    null-calibration simulation; same machinery as cv_score)."""
    y = tbl["q"].to_numpy(float) - tbl["L"].to_numpy(float)
    blocks = (tbl["env"].astype(str) + "|" + tbl["rep"].astype(str)).to_numpy()
    folds = group_folds(tbl["uid"].to_numpy())
    bases = {c: fit_basis(tbl[c].to_numpy(float)) for c in cols}
    if lam is None:
        lam = min(RIDGE_GRID, key=lambda l: cv_score(tbl, cols, bases, l))
    pred = np.full(len(tbl), np.nan)
    for f in range(N_FOLDS):
        tr, te = folds != f, folds == f
        X, stand = design(tbl[tr], cols, bases)
        yc = center_blocks(y[tr], blocks[tr])
        Xc = np.column_stack([center_blocks(X[:, j], blocks[tr])
                              for j in range(X.shape[1])])
        beta = ridge_fit(Xc, yc, lam)
        Xte, _ = design(tbl[te], cols, bases, stand)
        pred[te] = tbl["L"].to_numpy(float)[te] + Xte @ beta
    return pred


def predict(model, tbl):
    if model is None:                                   # M0
        return tbl["L"].to_numpy(float)
    bases = {c: np.asarray(v) for c, v in model["bases"].items()}
    stand = (np.asarray(model["stand"][0]), np.asarray(model["stand"][1]))
    X, _ = design(tbl, model["cols"], bases, stand)
    return tbl["L"].to_numpy(float) + X @ np.asarray(model["beta"])


MODEL_COLS = {"M1": B_CONT, "M2": B_CONT + ("b",), "M3": B_CONT + ("b", "a")}
