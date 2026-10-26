"""TASK 1: Reproduction binding (v0 vs archived historical ADNI numbers).

Independent re-derivation: per-seed pooled Spearman rho(misfit, err) over the
tested scans from reruns/purged_v0_perscan_local.npz, compared to the archived
historical per-seed list in results/honest_estimator_table.json
(estimator_table.structural_misfit.adni.per_seed). Also recompute the per-seed
held-out error rates and compare to results/purged_v0_estimators.json.

All code here is written by the independent verifier; no package summary code
is imported.
"""

from misfit.paths import resolve_path as _resolve_path
import json
import numpy as np
from scipy.stats import spearmanr

COR = _resolve_path("<REPO_ROOT>/misfit/correction_20261018")
HIST = _resolve_path("<REPO_ROOT>/misfit/results/honest_estimator_table.json")

out = {"task": 1}

f = np.load(f"{COR}/reruns/purged_v0_perscan_local.npz", allow_pickle=False)

my_rho, my_err = [], []
for s in range(5):
    t = f[f"tested_s{s}"].astype(bool)
    mis = f[f"misfit_s{s}"][t]
    err = f[f"err_s{s}"][t].astype(np.float64)
    my_rho.append(float(spearmanr(mis, err).statistic))
    my_err.append(float(err.mean()))

hist = json.load(open(HIST))
hist_per_seed = hist["estimator_table"]["structural_misfit"]["adni"]["per_seed"]
hist_mean = hist["estimator_table"]["structural_misfit"]["adni"]["mean"]

v0 = json.load(open(f"{COR}/results/purged_v0_estimators.json"))
v0_per_seed = v0["estimators"]["structural_misfit"]["rho_per_seed"]
v0_err = v0["err_rate_per_seed"]

diff_hist = [abs(a - b) for a, b in zip(my_rho, hist_per_seed)]
diff_v0 = [abs(a - b) for a, b in zip(my_rho, v0_per_seed)]
diff_err = [abs(a - b) for a, b in zip(my_err, v0_err)]

out["my_rho_per_seed"] = my_rho
out["hist_per_seed"] = hist_per_seed
out["v0_package_per_seed"] = v0_per_seed
out["max_abs_diff_vs_historical"] = max(diff_hist)
out["max_abs_diff_vs_v0_package"] = max(diff_v0)
out["my_rho_mean"] = float(np.mean(my_rho))
out["hist_mean"] = hist_mean
out["my_err_rate_per_seed"] = my_err
out["v0_package_err_rate_per_seed"] = v0_err
out["max_abs_diff_err_rate"] = max(diff_err)
# err stored float32 in npz, JSON stores float(np.float32) exactly; demand
# bit-level agreement for rho (same data, same scipy spearman definition) and
# <1e-7 for the float32-serialized error rates.
out["pass_rho_exact"] = bool(max(diff_hist) == 0.0 and max(diff_v0) == 0.0)
out["pass_rho_1e-12"] = bool(max(diff_hist) < 1e-12 and max(diff_v0) < 1e-12)
out["pass_err"] = bool(max(diff_err) < 1e-7)
out["PASS"] = bool(out["pass_rho_1e-12"] and out["pass_err"])

print(json.dumps(out, indent=2))
json.dump(out, open(f"{COR}/verify/out_task1.json", "w"), indent=2)
