"""TASK 5: CRT re-derivation from the frozen OASIS-3 confirmation rows.

From crt_study/results/oasis3_conf_rows_local.csv, with verifier-written code:
 (a) E(M2), E(M3), delta = E(M2)-E(M3): equal-weight mean over (env,rep) cells
     of the pair MSE of residuals r = q - qhat_M, computed BOTH via the
     identity (2/(n-1))*sum((r-rbar)^2) and by explicit pair enumeration;
     compare to crt/crt_recompute_verification.json.
 (b) inversion prevalence + AP for M2/M3 at tie tol 1e-6 on (env,rep) cells:
     pairs i<j; r_ij = q_i-q_j; d_ij = L_i-L_j; tie if |r|<=tol or |d|<=tol;
     inversion = sign disagreement among non-tied pairs; cell eligible if
     n>=3, has non-tied pairs, and inversions are neither 0 nor all;
     score = -sign(d_ij)*(qhat_i-qhat_j); AP = sum(precision_at_k * y_k)/n_pos
     over descending score; compare to crt/crt_ap_diagnostics.json.
 (c) the externally recorded relative CI upper bound, including an independent
     re-implementation of the participant-block bootstrap CI
     (per-env participant resampling, RandomState(20261018), B=2000,
     delta draw = mean-cell E(M2) - mean-cell E(M3)).
"""

from misfit.paths import resolve_path as _resolve_path
import csv
import itertools
import json
from collections import defaultdict

import numpy as np

COR = _resolve_path("<REPO_ROOT>/misfit/correction_20261018")
CSV = _resolve_path("<REPO_ROOT>/misfit/crt_study/results/oasis3_conf_rows_local.csv")
TOL = 1e-6
out = {"task": 5}

rows = list(csv.DictReader(open(CSV)))
uid = np.array([r["uid"] for r in rows])
env = np.array([r["env"] for r in rows])
rep = np.array([int(r["rep"]) for r in rows])
q = np.array([float(r["q"]) for r in rows])
L = np.array([float(r["L"]) for r in rows])
qh = {m: np.array([float(r[f"qhat_{m}"]) for r in rows]) for m in ["M2", "M3"]}
out["n_rows"] = len(rows)
out["n_participants"] = int(len(set(uid.tolist())))

cells = sorted(set(zip(env.tolist(), rep.tolist())))
cell_idx = {c: np.where((env == c[0]) & (rep == c[1]))[0] for c in cells}
out["n_cells"] = len(cells)


def mse_identity(u):
    return 2.0 * ((u - u.mean()) ** 2).sum() / (len(u) - 1)


def mse_enum(u):
    return float(np.mean([(u[i] - u[j]) ** 2
                          for i, j in itertools.combinations(range(len(u)), 2)]))


# ---- (a) endpoint
E_id, E_enum = {}, {}
for m in ["M2", "M3"]:
    r = q - qh[m]
    vi = [mse_identity(r[cell_idx[c]]) for c in cells if len(cell_idx[c]) >= 2]
    ve = [mse_enum(r[cell_idx[c]]) for c in cells if len(cell_idx[c]) >= 2]
    E_id[m] = float(np.mean(vi))
    E_enum[m] = float(np.mean(ve))
    out[f"identity_vs_enum_max_gap_{m}"] = float(
        max(abs(a - b) for a, b in zip(vi, ve)))

delta_id = E_id["M2"] - E_id["M3"]
ref = json.load(open(f"{COR}/crt/crt_recompute_verification.json"))["cohorts"]["oasis3"]
out["my_E_M2_identity"] = E_id["M2"]
out["my_E_M3_identity"] = E_id["M3"]
out["my_E_M2_enum"] = E_enum["M2"]
out["my_E_M3_enum"] = E_enum["M3"]
out["my_delta"] = float(delta_id)
out["ref_E_M2"] = ref["regenerated_E"]["M2"]
out["ref_E_M3"] = ref["regenerated_E"]["M3"]
out["ref_delta"] = ref["delta_M2_regenerated"]
out["abs_diff_E_M2"] = abs(E_id["M2"] - ref["regenerated_E"]["M2"])
out["abs_diff_E_M3"] = abs(E_id["M3"] - ref["regenerated_E"]["M3"])
out["abs_diff_delta"] = abs(delta_id - ref["delta_M2_regenerated"])
out["pass_endpoint"] = bool(
    out["abs_diff_E_M2"] < 1e-12 and out["abs_diff_E_M3"] < 1e-12
    and out["abs_diff_delta"] < 1e-12
    and max(out["identity_vs_enum_max_gap_M2"],
            out["identity_vs_enum_max_gap_M3"]) < 1e-10)

# ---- (b) inversion prevalence + AP
def ap(y, s, kind="quicksort"):
    """Average precision over a descending-score sweep; sort kind exposes tie-order sensitivity."""
    order = np.argsort(-s, kind=kind)
    yy = y[order].astype(float)
    prec = np.cumsum(yy) / (np.arange(len(yy)) + 1.0)
    return float((prec * yy).sum() / yy.sum())


elig, prev, aps = 0, [], {"M2": [], "M3": []}
aps_stable = {"M2": [], "M3": []}
excl = {"n_lt_3": 0, "all_tie": 0, "no_positive": 0, "all_positive": 0}
for c in cells:
    ii = cell_idx[c]
    n = len(ii)
    if n < 3:
        excl["n_lt_3"] += 1
        continue
    iu, ju = np.triu_indices(n, 1)
    r_ij = q[ii][iu] - q[ii][ju]
    d_ij = L[ii][iu] - L[ii][ju]
    tie = (np.abs(r_ij) <= TOL) | (np.abs(d_ij) <= TOL)
    keep = ~tie
    if keep.sum() == 0:
        excl["all_tie"] += 1
        continue
    inv = ((np.sign(r_ij) != np.sign(d_ij)) & keep)[keep]
    if inv.sum() == 0:
        excl["no_positive"] += 1
        continue
    if inv.sum() == len(inv):
        excl["all_positive"] += 1
        continue
    elig += 1
    prev.append(float(inv.mean()))
    for m in ["M2", "M3"]:
        s = (-np.sign(d_ij) * (qh[m][ii][iu] - qh[m][ii][ju]))[keep]
        aps[m].append(ap(inv, s))
        aps_stable[m].append(ap(inv, s, kind="stable"))

refap = json.load(open(f"{COR}/crt/crt_ap_diagnostics.json"))["oasis3"]["frozen_tie_tol"]
out["my_eligible_cells"] = elig
out["my_exclusions"] = excl
out["my_prevalence_cell_mean"] = float(np.mean(prev))
out["my_ap_cell_mean_M2"] = float(np.mean(aps["M2"]))
out["my_ap_cell_mean_M3"] = float(np.mean(aps["M3"]))
out["ap_sort_tie_sensitivity_max"] = float(max(
    max(abs(a - b) for a, b in zip(aps["M2"], aps_stable["M2"])),
    max(abs(a - b) for a, b in zip(aps["M3"], aps_stable["M3"]))))
out["ref_eligible_cells"] = refap["eligible_cells"]
out["ref_prevalence_cell_mean"] = refap["prevalence_cell_mean"]
out["ref_ap_cell_mean_M2"] = refap["ap_cell_mean"]["M2"]
out["ref_ap_cell_mean_M3"] = refap["ap_cell_mean"]["M3"]
out["abs_diff_prevalence"] = abs(out["my_prevalence_cell_mean"]
                                 - refap["prevalence_cell_mean"])
out["abs_diff_ap_M2"] = abs(out["my_ap_cell_mean_M2"] - refap["ap_cell_mean"]["M2"])
out["abs_diff_ap_M3"] = abs(out["my_ap_cell_mean_M3"] - refap["ap_cell_mean"]["M3"])
out["pass_ap"] = bool(elig == refap["eligible_cells"]
                      and out["abs_diff_prevalence"] < 1e-12
                      and out["abs_diff_ap_M2"] < 1e-12
                      and out["abs_diff_ap_M3"] < 1e-12)

# ---- (c) CI-upper claim + independent bootstrap re-implementation
frozen = json.load(open(
    _resolve_path("<REPO_ROOT>/misfit/crt_study/results/combined_summary.json")))
pb_ci = frozen["cohorts"]["oasis3"]["pb_ci"]
ratio = pb_ci[1] / E_id["M2"]
out["frozen_pb_ci"] = pb_ci
out["ci_upper"] = pb_ci[1]
out["ci_upper_over_my_E_M2"] = float(ratio)
out["ref_ci_upper_as_relative"] = ref["ci_upper_as_relative_of_E_M2"]
out["pass_ratio_claim"] = bool(abs(ratio - ref["ci_upper_as_relative_of_E_M2"]) < 1e-12)

# independent bootstrap (recipe: per-env participant block resampling,
# RandomState(20261018), B=2000, envs in first-appearance order,
# participants within env in sorted-unique order; draw stat =
# mean-cell E(M2) - mean-cell E(M3) over cells with n>=2)
envs_order = list(dict.fromkeys(env.tolist()))
reps_order = sorted(set(rep.tolist()))
res2 = q - qh["M2"]
res3 = q - qh["M3"]
cell_uids = {g: np.unique(uid[env == g]) for g in envs_order}
row_map = {g: [np.where((env == g) & (uid == u_))[0] for u_ in cell_uids[g]]
           for g in envs_order}
rng = np.random.RandomState(20261018)
B = 2000
deltas = np.empty(B)
for b in range(B):
    rr = []
    for g in envs_order:
        nu = len(cell_uids[g])
        pick = rng.choice(nu, nu, replace=True)
        rr.extend(row_map[g][p] for p in pick)
    rr = np.concatenate(rr)
    se, sr = env[rr], rep[rr]
    def mean_E(u):
        """Equal-weight mean of the pair MSE over (env, rep) cells with n >= 2."""
        vals = [mse_identity(u[(se == g) & (sr == k)])
                for g in envs_order for k in reps_order
                if ((se == g) & (sr == k)).sum() >= 2]
        return float(np.mean(vals))
    deltas[b] = mean_E(res2[rr]) - mean_E(res3[rr])
my_ci = [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]
out["my_bootstrap_ci"] = my_ci
out["bootstrap_ci_max_abs_diff_vs_frozen"] = max(
    abs(my_ci[0] - pb_ci[0]), abs(my_ci[1] - pb_ci[1]))
out["pass_bootstrap"] = bool(out["bootstrap_ci_max_abs_diff_vs_frozen"] < 1e-9)

out["PASS"] = bool(out["pass_endpoint"] and out["pass_ap"]
                   and out["pass_ratio_claim"] and out["pass_bootstrap"])
print(json.dumps(out, indent=2))
json.dump(out, open(f"{COR}/verify/out_task5.json", "w"), indent=2)
