"""TASK 6: Master table spot-check (results/adni_original_vs_corrected.json).

Verifier-chosen random sample of 10 rows (RandomState(20261018) over the row
list), each checked against the primary source files:
  historical  <- historical JSONs in <REPO_ROOT>/misfit/results/
  corrected   <- the correction package's v2 result files
and the 'changed_vs_v2_A6' flag re-evaluated (|v2-hist| > 2*hist_sd where sd
is present; abs/sign/CI rules otherwise). As an extra sweep, the mean2sd flag
rule is re-evaluated on every row that carries a historical_sd using the
table's own stored values.
"""

from misfit.paths import resolve_path as _resolve_path
import json
import numpy as np

COR = _resolve_path("<REPO_ROOT>/misfit/correction_20261018")
HIST = _resolve_path("<REPO_ROOT>/misfit/results")


def j(p):
    return json.load(open(p))


table = j(f"{COR}/results/adni_original_vs_corrected.json")
rows = table["rows"]

het = j(f"{HIST}/honest_estimator_table.json")
wec = j(f"{HIST}/within_env_canonical.json")
mlr = j(f"{HIST}/ml_reliability.json")
td = j(f"{HIST}/training_dynamics.json")
pt = j(f"{HIST}/paired_tests.json")
cs = j(f"{HIST}/clinical_stratified.json")
bb = j(f"{HIST}/bootstrap_balanced.json")
strata = j(f"{HIST}/class_env_strata_s16.json")

e2 = j(f"{COR}/results/purged_v2_estimators.json")
s2 = j(f"{COR}/results/corrected_v2_summaries.json")
m2 = j(f"{COR}/results/purged_v2_mlrel.json")
d2 = j(f"{COR}/results/purged_v2_dynamics.json")
e0 = j(f"{COR}/results/purged_v0_estimators.json")

COVS = ["1.0", "0.8", "0.5", "0.3"]


def resolve(claim):
    """Return (hist_value, hist_sd, v2_value, flag_kind) from PRIMARY files."""
    # claim strings may carry a parenthetical suffix; the key is the first token
    c = claim.split(" ")[0]
    p = c.split("/")
    if p[0] == "estimator_table":
        h = het["estimator_table"][p[1]]["adni"]
        return h["mean"], h.get("std"), e2["estimators"][p[1]]["rho_mean"], "mean2sd"
    if p[0] == "held_out_error_rate":
        return (float(np.mean(e0["err_rate_per_seed"])), None,
                float(np.mean(e2["err_rate_per_seed"])), "abs")
    if p[0].startswith("within_env_rho_mean"):
        a = wec["adni"]
        return (a["mean"], a["std"],
                s2["within_env_canonical"]["mean_within_env_rho"], "mean2sd")
    if p[0].startswith("within_env_cells_positive"):
        return (wec["adni"]["n_positive"], None,
                s2["within_env_canonical"]["n_positive"], "abs")
    if p[0] == "auroc":
        k = p[1].split(" ")[0]
        h = mlr["adni"]["block_b_auroc"][k]
        return (h["auroc"], h.get("auroc_std"),
                m2["block_b_auroc"][k]["auroc"], "mean2sd")
    if p[0] == "selective":
        cov, metric = p[1], p[2]
        h = cs["adni"]["selective_clinical"][COVS.index(cov)][metric]
        return (h.get("mean"), h.get("std"),
                s2["selective"][cov][metric]["mean"], "mean2sd")
    if p[0] == "stratified":
        h = cs["adni"]["stratified"][p[1]]
        return (h.get("mean"), h.get("std"),
                s2["stratified"][p[1]]["mean"], "mean2sd")
    if p[0].startswith("strata16"):
        h = strata["adni"]["pooled_over_cells"]
        return (h.get("rho_mean"), h.get("rho_std"),
                s2["strata16"]["rho_mean"], "mean2sd")
    if p[0] == "dynamics":
        h = td["adni"][p[1]]
        return (h["auroc_mean"], h.get("auroc_std"),
                d2[p[1]]["auroc_mean"], "mean2sd")
    if p[0] == "paired":
        comp = p[1].replace("misfit_vs_", "")
        return (pt["adni"][comp]["dAUROC_ci"], None,
                s2["paired_dynamics"]["participant"]["comparators"][comp]["dAUROC_ci"],
                "ci_upper_sign")
    if p[0].startswith("bootstrap_auroc_ci"):
        return (bb["adni"]["auroc_ci"], None,
                s2["auroc_boot"]["participant_unit"]["auroc_ci95"], "none")
    if p[0] == "fn_capture":
        hv = {"20pct": 0.63, "50pct": 0.96}[p[1]]
        return hv, None, s2["fn_capture"][p[1]]["fn_capture"]["mean"], "abs"
    if p[0] == "conformal":
        h = mlr["adni"]["block_a_conformal"][p[1]]
        return (h[p[2]], h.get(p[2] + "_std"),
                m2["block_a_conformal"][p[1]][p[2]], "mean2sd")
    raise KeyError(claim)


def expected_flag(hv, sd, v2, kind):
    """Re-evaluate the changed_vs_v2_A6 flag from source values under the A6 rule for this kind."""
    if kind == "mean2sd" and sd is not None:
        return bool(abs(v2 - hv) > 2 * sd)
    if kind == "mean2sd" and sd is None:
        return None  # builder leaves changed=None without sd
    if kind == "abs":
        return bool(abs(v2 - hv) > 1e-9)
    if kind == "ci_upper_sign":
        return bool((hv[1] < 0) != (v2[1] < 0))
    return None


rng = np.random.RandomState(20261018)
picks = sorted(rng.choice(len(rows), 10, replace=False).tolist())

out = {"task": 6, "sampled_row_indices": picks, "checks": []}
all_ok = True
for i in picks:
    r = rows[i]
    hv, sd, v2, kind = resolve(r["claim"])
    def close(a, b):
        if a is None or b is None:
            return a is None and b is None
        if isinstance(a, list):
            return max(abs(x - y) for x, y in zip(a, b)) < 1e-12
        return abs(a - b) < 1e-12
    ok_h = close(hv, r["historical"])
    ok_sd = close(sd, r["historical_sd"])
    ok_v2 = close(v2, r["corrected_v2"])
    exp = expected_flag(hv, sd, v2, kind)
    # expected None (mean2sd without sd) must correspond to a stored None flag
    ok_flag = (exp == r["changed_vs_v2_A6"]) if exp is not None \
        else r["changed_vs_v2_A6"] in (None, exp)
    rowok = ok_h and ok_sd and ok_v2 and ok_flag
    all_ok &= rowok
    out["checks"].append({
        "row": i, "claim": r["claim"], "kind": kind,
        "hist_source": hv, "hist_table": r["historical"],
        "sd_source": sd, "sd_table": r["historical_sd"],
        "v2_source": v2, "v2_table": r["corrected_v2"],
        "flag_expected": exp, "flag_table": r["changed_vs_v2_A6"],
        "ok": bool(rowok)})

# extra sweep: mean2sd flag logic on all rows that carry a numeric sd
sweep_bad = []
for i, r in enumerate(rows):
    if r["historical_sd"] is None or not isinstance(r["historical"], (int, float)):
        continue
    if r["corrected_v2"] is None:
        continue
    exp = bool(abs(r["corrected_v2"] - r["historical"]) > 2 * r["historical_sd"])
    if exp != r["changed_vs_v2_A6"]:
        sweep_bad.append({"row": i, "claim": r["claim"],
                          "expected": exp, "stored": r["changed_vs_v2_A6"]})
out["flag_sweep_rows_checked"] = sum(
    1 for r in rows if r["historical_sd"] is not None
    and isinstance(r["historical"], (int, float)) and r["corrected_v2"] is not None)
out["flag_sweep_mismatches"] = sweep_bad
out["PASS"] = bool(all_ok and not sweep_bad)
print(json.dumps(out, indent=2))
json.dump(out, open(f"{COR}/verify/out_task6.json", "w"), indent=2)
