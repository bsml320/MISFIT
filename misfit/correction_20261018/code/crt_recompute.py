"""CRT reporting corrections: endpoint regeneration + inversion-AP diagnostics.

Regenerates, from the frozen server-side confirmation records
(crt_study/results/<cohort>_conf_rows_local.csv), everything the correction
needs, WITHOUT touching any frozen file:

1. Endpoint verification: pair-MSE identity (formula vs explicit pair
   enumeration), per-cell alignment counts, equal env x rep weighting, and the
   reported E(M0..M3), raw-transport E, delta_M2 and relative improvements,
   compared against crt_study/results/combined_summary.json.
2. Corrected inversion-AP diagnostics (plan item 4c), replacing the frozen
   "chance = 0.5" reading:
   - AP per model on identical eligible cells (eligibility is method-
     independent: ties and label composition depend only on q, L),
   - inversion prevalence computed on exactly the AP-contributing cells
     (cell-mean and pair-pooled),
   - an empirical finite-sample random-ranking reference (seeded permutations;
     distinguished from prevalence; descriptive only -- pair-label permutation
     is NOT a valid significance test because pairs share participants),
   - eligible participant / pair / inversion / cell counts and exclusion
     reasons (n<3 cells, all-tie cells, no-positive / all-positive cells),
   - both tie thresholds: frozen 1e-6 and sensitivity 1e-2,
   - POST-HOC (labeled): paired M3-minus-M2 AP uncertainty via a
     participant-block bootstrap that resamples confirmation participants with
     replacement and REBUILDS the pair sets inside every draw (pair rows are
     never resampled directly); a drawn participant carries its rows in all
     repetitions; duplicate-draw self-pairs produce exact ties and are
     excluded by the tie rule; replicates with no eligible cell are counted
     invalid and reported.

Outputs: correction_20261018/crt/crt_recompute_verification.json
         correction_20261018/crt/crt_ap_diagnostics.json
"""

from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

CRT = Path(_resolve_path("<REPO_ROOT>/misfit/crt_study"))
OUT = Path(_resolve_path("<REPO_ROOT>/misfit/correction_20261018/crt"))
COHORTS = ["oasis3", "adni", "oasis1"]
MODELS = ["M0", "M1", "M2", "M3"]
TIE_TOL_FROZEN = 1e-6
TIE_TOL_SENS = 1e-2
B_BOOT = 2000
BOOT_SEED = 20261019  # post-hoc analysis, new seed, documented
N_RAND = 200
RAND_SEED = 20261020


def pair_mse_formula(u: np.ndarray) -> float:
    """Pair MSE via the O(n) identity 2 * sum((u - mean)^2) / (n - 1)."""
    return float(2.0 * ((u - u.mean()) ** 2).sum() / (len(u) - 1))


def pair_mse_enum(u: np.ndarray) -> float:
    """Pair MSE by explicit enumeration of all unordered pairs (O(n^2) reference)."""
    return float(np.mean([(u[i] - u[j]) ** 2
                          for i, j in itertools.combinations(range(len(u)), 2)]))


def identity_test() -> dict:
    """Check pair_mse_formula against pair_mse_enum on fixed and seeded random vectors."""
    rng = np.random.RandomState(7)
    cases = [np.array([0.0, 1.0]), np.array([1.0, 2.0, 4.0]),
             np.array([3.0, 3.0, 3.0, 3.0]), rng.randn(5), rng.randn(17) * 3 + 1]
    worst = max(abs(pair_mse_formula(u) - pair_mse_enum(u)) for u in cases)
    return {"n_cases": len(cases), "max_abs_gap": worst, "pass": bool(worst < 1e-12)}


def cell_endpoint(df: pd.DataFrame, resid_col: str) -> float:
    """Equal-weight mean over (env, rep) cells with n>=2 of the pair MSE."""
    vals = []
    for (_, _), g in df.groupby(["env", "rep"]):
        u = g[resid_col].to_numpy(float)
        if len(u) >= 2:
            vals.append(pair_mse_formula(u))
    return float(np.mean(vals))


def ap_score(y: np.ndarray, s: np.ndarray) -> float:
    """Average precision of scores s against binary labels y (descending-score sweep)."""
    order = np.argsort(-s)
    yy = y[order]
    prec = np.cumsum(yy) / (np.arange(len(yy)) + 1)
    return float((prec * yy).sum() / yy.sum())


def cell_pairs(g: pd.DataFrame, tol: float):
    """Pair-level arrays for one (env, rep) cell: labels + per-model scores."""
    q = g["q"].to_numpy(float)
    L = g["L"].to_numpy(float)
    n = len(g)
    iu, ju = np.triu_indices(n, 1)
    r_ij = q[iu] - q[ju]
    d_ij = L[iu] - L[ju]
    tie = (np.abs(r_ij) <= tol) | (np.abs(d_ij) <= tol)
    inv = (np.sign(r_ij) != np.sign(d_ij)) & ~tie
    keep = ~tie
    scores = {}
    for m in MODELS:
        qh = g[f"qhat_{m}"].to_numpy(float)
        # score > 0 when the model's predicted q ordering runs against the L
        # ordering, i.e. the model ranks the pair as a predicted inversion
        scores[m] = (-np.sign(d_ij) * (qh[iu] - qh[ju]))[keep]
    return inv[keep], scores, int(tie.sum()), n


def ap_diagnostics(df: pd.DataFrame, tol: float) -> dict:
    """Inversion-AP diagnostics over (env, rep) cells at tie tolerance tol.

    A cell is eligible when it has at least 3 participants, at least one
    non-tied pair, and mixed inversion labels. Eligibility depends only on
    (q, L), so every model is scored on identical pairs. Returns exclusion
    counts, cell-mean and pair-pooled inversion prevalence, a seeded
    random-ranking AP reference (descriptive only), per-model cell-mean AP,
    and the per-cell records.
    """
    cells, excl = [], {"n_lt_3": 0, "all_tie": 0, "no_positive": 0, "all_positive": 0}
    for (envv, repv), g in df.groupby(["env", "rep"]):
        if len(g) < 3:
            excl["n_lt_3"] += 1
            continue
        inv, scores, n_tie, n = cell_pairs(g, tol)
        if len(inv) == 0:
            excl["all_tie"] += 1
            continue
        if inv.sum() == 0:
            excl["no_positive"] += 1
            continue
        if inv.sum() == len(inv):
            excl["all_positive"] += 1
            continue
        cells.append({"env": envv, "rep": int(repv), "n_participants": n,
                      "n_pairs_after_ties": int(len(inv)), "n_ties": n_tie,
                      "n_inversions": int(inv.sum()),
                      "prevalence": float(inv.mean()),
                      "ap": {m: ap_score(inv, scores[m]) for m in MODELS}})
    if not cells:
        return {"eligible_cells": 0, "exclusions": excl}
    # finite-sample reference: AP of random permutations on each eligible
    # cell's pair labels, one shared RNG stream across cells
    rng = np.random.RandomState(RAND_SEED)
    rand_ref = []
    for c in cells:
        sub = df[(df.env == c["env"]) & (df.rep == c["rep"])]
        inv, _, _, _ = cell_pairs(sub, tol)
        vals = [ap_score(inv, rng.permutation(len(inv)).astype(float))
                for _ in range(N_RAND)]
        rand_ref.append(float(np.mean(vals)))
    pooled_pairs = sum(c["n_pairs_after_ties"] for c in cells)
    pooled_inv = sum(c["n_inversions"] for c in cells)
    return {
        "tie_tol": tol,
        "eligible_cells": len(cells),
        "exclusions": excl,
        "eligible_participants": int(sum(c["n_participants"] for c in cells)),
        "eligible_pairs": pooled_pairs,
        "eligible_inversions": pooled_inv,
        "prevalence_cell_mean": float(np.mean([c["prevalence"] for c in cells])),
        "prevalence_pair_pooled": float(pooled_inv / pooled_pairs),
        "random_ranking_reference_cell_mean": float(np.mean(rand_ref)),
        "ap_cell_mean": {m: float(np.mean([c["ap"][m] for c in cells]))
                         for m in MODELS},
        "per_cell": cells,
    }


def paired_ap_bootstrap(df: pd.DataFrame, tol: float) -> dict:
    """Participant-block bootstrap of mean-cell AP(M3) - AP(M2); pairs rebuilt
    inside each draw; same draws for both models by construction."""
    uids = np.array(sorted(df["uid"].unique()))
    n_u = len(uids)
    rng = np.random.RandomState(BOOT_SEED)
    rows_by_uid = {u: df[df.uid == u] for u in uids}
    deltas, invalid = [], 0
    for _ in range(B_BOOT):
        draw = rng.randint(0, n_u, n_u)
        boot_df = pd.concat([rows_by_uid[uids[i]] for i in draw],
                            ignore_index=True)
        ap2, ap3 = [], []
        for (_, _), g in boot_df.groupby(["env", "rep"]):
            if len(g) < 3:
                continue
            inv, scores, _, _ = cell_pairs(g, tol)
            if len(inv) == 0 or inv.sum() in (0, len(inv)):
                continue
            ap2.append(ap_score(inv, scores["M2"]))
            ap3.append(ap_score(inv, scores["M3"]))
        if not ap2:
            invalid += 1
            continue
        deltas.append(float(np.mean(ap3) - np.mean(ap2)))
    deltas = np.array(deltas)
    return {
        "label": "POST-HOC paired participant-block bootstrap of "
                 "cell-mean AP(M3) - AP(M2); pairs rebuilt per draw",
        "B": B_BOOT, "seed": BOOT_SEED, "invalid_replicates": invalid,
        "delta_mean": float(deltas.mean()),
        "ci95": [float(np.percentile(deltas, 2.5)),
                 float(np.percentile(deltas, 97.5))],
    }


def main() -> None:
    """Regenerate endpoints and AP diagnostics per cohort; write both output JSONs."""
    OUT.mkdir(exist_ok=True)
    summary = json.load(open(CRT / "results/combined_summary.json"))
    verif = {"pair_mse_identity_test": identity_test(), "cohorts": {}}
    apdiag = {}
    for cn in COHORTS:
        df = pd.read_csv(CRT / f"results/{cn}_conf_rows_local.csv")
        rep_counts = df.groupby(["env", "rep"]).size()
        E = {}
        for m in MODELS:
            df[f"res_{m}"] = df["q"] - df[f"qhat_{m}"]
            E[m] = cell_endpoint(df, f"res_{m}")
        df["res_raw"] = df["q"] - (df["L"] + df["a"])
        E["raw_transport"] = cell_endpoint(df, "res_raw")
        ref = summary["cohorts"][cn]["E"]
        gaps = {m: abs(E[m] - ref[m]) for m in E}
        delta_m2 = E["M2"] - E["M3"]
        rel = {m: (E[m] - E["M3"]) / E[m] for m in ("M0", "M1", "M2")}
        ci_hi = summary["cohorts"][cn]["pb_ci"][1]
        # regenerate the frozen participant-block bootstrap CI with the
        # production code and seed (crt_study/code/analysis.py)
        import sys
        sys.path.insert(0, str(CRT / "code"))
        from misfit.crt_study.code import analysis as crt_analysis
        envs = list(dict.fromkeys(df["env"]))
        reps = sorted(df["rep"].unique())
        resid = {m: df[f"res_{m}"].to_numpy(float) for m in MODELS}
        boot = crt_analysis.boot_participants(df, resid, envs, reps)
        ci_regen = boot["M2"]["ci95"]
        ci_frozen = summary["cohorts"][cn]["pb_ci"]
        ci_gap = max(abs(ci_regen[0] - ci_frozen[0]), abs(ci_regen[1] - ci_frozen[1]))
        verif["cohorts"][cn] = {
            "n_rows": int(len(df)),
            "n_conf_participants": int(df["uid"].nunique()),
            "cells": {f"{e}|rep{r}": int(v) for (e, r), v in rep_counts.items()},
            "regenerated_E": E,
            "abs_gap_vs_frozen_summary": gaps,
            "match_within_float32_serialization_5e-9": bool(max(gaps.values()) < 5e-9),
            "pb_ci_M2_regenerated": ci_regen,
            "pb_ci_M2_frozen": ci_frozen,
            "pb_ci_max_abs_gap": ci_gap,
            "delta_M2_regenerated": delta_m2,
            "delta_M2_frozen": summary["cohorts"][cn]["delta_M2"],
            "relative_regenerated": rel,
            "relative_vs_M2_pct": 100 * rel["M2"],
            "ci_upper_as_relative_of_E_M2": float(ci_hi / E["M2"]),
            "note_5pct_not_excluded": bool(ci_hi / E["M2"] > 0.05),
        }
        apdiag[cn] = {
            "frozen_tie_tol": ap_diagnostics(df, TIE_TOL_FROZEN),
            "sensitivity_tie_tol_1e-2": ap_diagnostics(df, TIE_TOL_SENS),
            "paired_M3_minus_M2": paired_ap_bootstrap(df, TIE_TOL_FROZEN),
        }
        print(f"[{cn}] E regenerated, max gap {max(gaps.values()):.2e}; "
              f"AP cells {apdiag[cn]['frozen_tie_tol'].get('eligible_cells')}",
              flush=True)
    (OUT / "crt_recompute_verification.json").write_text(json.dumps(verif, indent=2))
    (OUT / "crt_ap_diagnostics.json").write_text(json.dumps(apdiag, indent=2))
    print("saved", OUT)


if __name__ == "__main__":
    main()
