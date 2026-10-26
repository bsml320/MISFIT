"""TASK 4: Independent re-implementation of the pinned paired participant
bootstrap (v2, misfit vs aum_neg) from reruns/purged_v2_dynamics_local.npz.

Pinned recipe (re-implemented from scratch, not imported):
  - tested mask must be identical across the 5 seeds;
  - units = sorted unique ptids of tested scans;
  - draw matrix: np.random.RandomState(0).randint(0, n_units, (2000, n_units));
  - each drawn participant contributes all of its tested scan rows, replicated
    across all 5 seeds (scores and err concatenated seed-major);
  - per draw: dAUROC = AUROC(err, misfit) - AUROC(err, aum_neg) on the SAME
    drawn rows; draws whose pooled err is single-class are dropped;
  - CI = percentile [2.5, 97.5] of the dAUROC draws.
Compare to corrected_v2_summaries.json
  paired_dynamics.participant.comparators.aum_neg.dAUROC_ci (exact match
  expected since the recipe pins the RNG).
"""

from misfit.paths import resolve_path as _resolve_path
import json
import numpy as np
from sklearn.metrics import roc_auc_score

COR = _resolve_path("<REPO_ROOT>/misfit/correction_20261018")
B = 2000
out = {"task": 4}

f = np.load(f"{COR}/reruns/purged_v2_dynamics_local.npz", allow_pickle=False)
tested = f["tested_s0"].astype(bool)
same_tested = all(np.array_equal(tested, f[f"tested_s{s}"].astype(bool))
                  for s in range(5))
out["tested_identical_across_seeds"] = bool(same_tested)
assert same_tested

pt = f["ptids"][tested]
n_rows = int(tested.sum())
mis = np.concatenate([f[f"misfit_s{s}"][tested] for s in range(5)])
aum = np.concatenate([f[f"aum_neg_s{s}"][tested] for s in range(5)])
err = np.concatenate([f[f"err_s{s}"][tested] for s in range(5)])

units = np.array(sorted(set(pt.tolist())))
n_units = len(units)
out["n_units"] = int(n_units)
rows_by_unit = {u: np.where(pt == u)[0] for u in units}

rng = np.random.RandomState(0)
draws = rng.randint(0, n_units, (B, n_units))

# seed-major layout: the tested row at within-seed index i lives at s*n_rows + i
offsets = (np.arange(5) * n_rows)[:, None]
d_auc = []
dropped = 0
for b in range(B):
    ridx = np.concatenate([rows_by_unit[units[i]] for i in draws[b]])
    idx = (offsets + ridx[None, :]).ravel()
    e = err[idx]
    s = e.sum()
    # AUROC undefined on single-class draws; drop and count them
    if s == 0 or s == len(e):
        dropped += 1
        continue
    d_auc.append(roc_auc_score(e, mis[idx]) - roc_auc_score(e, aum[idx]))

d_auc = np.asarray(d_auc)
my_ci = [float(np.percentile(d_auc, 2.5)), float(np.percentile(d_auc, 97.5))]
my_mean = float(d_auc.mean())

ref = json.load(open(f"{COR}/results/corrected_v2_summaries.json"))
r = ref["paired_dynamics"]["participant"]["comparators"]["aum_neg"]
out["my_dAUROC_ci"] = my_ci
out["ref_dAUROC_ci"] = r["dAUROC_ci"]
out["my_dAUROC_mean"] = my_mean
out["ref_dAUROC_mean"] = r["dAUROC_mean"]
out["my_dropped_single_class"] = dropped
out["ref_dropped_single_class"] = r["dropped_single_class"]
out["ref_n_units"] = ref["paired_dynamics"]["participant"]["n_units"]
out["ci_max_abs_diff"] = max(abs(my_ci[0] - r["dAUROC_ci"][0]),
                             abs(my_ci[1] - r["dAUROC_ci"][1]))
out["mean_abs_diff"] = abs(my_mean - r["dAUROC_mean"])
out["ci_contains_zero"] = bool(my_ci[0] <= 0.0 <= my_ci[1])
out["PASS"] = bool(out["ci_max_abs_diff"] == 0.0
                   and out["mean_abs_diff"] == 0.0
                   and dropped == r["dropped_single_class"]
                   and n_units == out["ref_n_units"])
out["PASS_1e-12"] = bool(out["ci_max_abs_diff"] < 1e-12
                         and out["mean_abs_diff"] < 1e-12)

print(json.dumps(out, indent=2))
json.dump(out, open(f"{COR}/verify/out_task4.json", "w"), indent=2)
