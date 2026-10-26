"""Original-versus-corrected ADNI results (deliverable 5).

Builds machine-readable tables comparing every tracked manuscript quantity:
historical archived value -> v0 fresh reproduction (binding) -> V1 (purge,
historical preprocessing) -> V2 (purge + source-only preprocessing; HEADLINE
per addendum A1) -> V1R (size-matched random-removal control) -> V3 (labeled
dedup sensitivity). Change assessment per the frozen A6 criteria, evaluated
against V2.

Outputs: results/adni_original_vs_corrected.json / .csv
"""

from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

import json
from pathlib import Path

import numpy as np

COR = Path(__file__).resolve().parent.parent
RES = COR / "results"
HIST = Path(_resolve_path("<REPO_ROOT>/misfit/results"))
VARIANTS = ["v0", "v1", "v2", "v1r", "v3"]


def jload(p):
    return json.load(open(p)) if Path(p).exists() else None


def est(variant):
    tag = "dedup_v3" if variant == "v3" else f"purged_{variant}"
    return jload(RES / f"{tag}_estimators.json")


def summ(variant):
    return jload(RES / f"corrected_{variant}_summaries.json")


def mlrel(variant):
    tag = "dedup_v3" if variant == "v3" else f"purged_{variant}"
    return jload(RES / f"{tag}_mlrel.json")


def dyn(variant):
    tag = "dedup_v3" if variant == "v3" else f"purged_{variant}"
    return jload(RES / f"{tag}_dynamics.json")


def row(claim, hist_mean, hist_sd, values, kind="mean2sd", note=""):
    """values: {variant: corrected value}; A6 criterion (i) when hist_sd given."""
    v2 = values.get("v2")
    changed = None
    if v2 is not None and hist_mean is not None:
        if kind == "mean2sd" and hist_sd is not None:
            changed = bool(abs(v2 - hist_mean) > 2 * hist_sd)
        elif kind == "sign":
            changed = bool(np.sign(v2) != np.sign(hist_mean))
        elif kind == "abs":
            changed = bool(abs(v2 - hist_mean) > 1e-9)
    return {"claim": claim, "historical": hist_mean, "historical_sd": hist_sd,
            **{f"corrected_{v}": values.get(v) for v in VARIANTS},
            "changed_vs_v2_A6": changed, "note": note}


def main():
    """Assemble every tracked comparison row from the primary result files and
    write the JSON and CSV tables."""
    E = {v: est(v) for v in VARIANTS}
    S = {v: summ(v) for v in VARIANTS}
    M = {v: mlrel(v) for v in VARIANTS}
    D = {v: dyn(v) for v in VARIANTS}
    het = jload(HIST / "honest_estimator_table.json")
    wec = jload(HIST / "within_env_canonical.json")
    mlr = jload(HIST / "ml_reliability.json")
    td = jload(HIST / "training_dynamics.json")
    pt = jload(HIST / "paired_tests.json")
    cs = jload(HIST / "clinical_stratified.json")
    bb = jload(HIST / "bootstrap_balanced.json")
    strata = jload(HIST / "class_env_strata_s16.json")
    cip = jload(HIST / "clinical_insight_panel.json")

    rows = []

    # headline pooled rho (every estimator in the table)
    for name, h in het["estimator_table"].items():
        if not isinstance(h, dict) or "adni" not in h:
            continue
        hm, hs = h["adni"].get("mean"), h["adni"].get("std")
        if hm is None:
            continue
        vals = {v: (E[v]["estimators"].get(name, {}).get("rho_mean")
                    if E[v] else None) for v in VARIANTS}
        rows.append(row(f"estimator_table/{name}/rho", hm, hs, vals))

    # error rate (no archived scalar; the v0 reproduction is the reference)
    vals = {v: float(np.mean(E[v]["err_rate_per_seed"])) if E[v] else None
            for v in VARIANTS}
    rows.append(row("held_out_error_rate", vals.get("v0"), None, vals,
                    kind="abs", note="reference = v0 fresh reproduction"))

    # within-env canonical
    a = wec["adni"]
    vals = {v: (S[v]["within_env_canonical"]["mean_within_env_rho"]
                if S[v] else None) for v in VARIANTS}
    rows.append(row("within_env_rho_mean", a["mean"], a["std"], vals))
    vals = {v: (S[v]["within_env_canonical"]["n_positive"] if S[v] else None)
            for v in VARIANTS}
    rows.append(row("within_env_cells_positive (40/40)", a["n_positive"], None,
                    vals, kind="abs",
                    note="categorical A6(iii): 'all positive' must hold exactly"))

    # AUROC / AUPRC (MLR block B)
    for k in ["misfit", "erm_loss", "conformal_q", "max_softmax", "energy",
              "mc_dropout"]:
        h = mlr["adni"]["block_b_auroc"][k]
        vals = {v: (M[v]["block_b_auroc"][k]["auroc"] if M[v] else None)
                for v in VARIANTS}
        rows.append(row(f"auroc/{k}", h["auroc"],
                        h.get("auroc_std"), vals))

    # selective prediction (clinical table)
    for covkey, hist_row in zip(["1.0", "0.8", "0.5", "0.3"],
                                cs["adni"]["selective_clinical"]):
        for metric in ["accuracy", "sensitivity", "specificity",
                       "fn_among_accepted"]:
            h = hist_row[metric]
            vals = {v: (S[v]["selective"][covkey][metric]["mean"]
                        if S[v] else None) for v in VARIANTS}
            rows.append(row(f"selective/{covkey}/{metric}", h.get("mean"),
                            h.get("std"), vals))

    # stratified
    for k in ["auroc_overall", "auroc_impaired", "auroc_control",
              "rho_impaired", "rho_control", "balanced_accuracy",
              "rho_excl_singleclass", "rho_balanced_only"]:
        h = cs["adni"]["stratified"][k]
        vals = {v: (S[v]["stratified"][k]["mean"] if S[v] else None)
                for v in VARIANTS}
        rows.append(row(f"stratified/{k}", h.get("mean"), h.get("std"), vals))

    # strata S16
    h = strata["adni"]
    vals = {v: (S[v]["strata16"]["rho_mean"] if S[v] else None) for v in VARIANTS}
    rows.append(row("strata16/rho_mean",
                    h["pooled_over_cells"].get("rho_mean"),
                    h["pooled_over_cells"].get("rho_std"), vals))

    # training dynamics
    for k in ["misfit", "aum_neg", "forgetting", "el2n"]:
        h = td["adni"][k]
        vals = {v: (D[v][k]["auroc_mean"] if D[v] else None) for v in VARIANTS}
        rows.append(row(f"dynamics/{k}/auroc", h["auroc_mean"], h["auroc_std"],
                        vals))

    # paired tests (CI-based statements, A6(iv))
    for comp in ["aum_neg", "el2n", "forgetting"]:
        h = pt["adni"][comp]["dAUROC_ci"]
        vals = {}
        for v in VARIANTS:
            if S[v] and S[v].get("paired_dynamics"):
                vals[v] = S[v]["paired_dynamics"]["participant"][
                    "comparators"][comp]["dAUROC_ci"]
        rows.append({"claim": f"paired/misfit_vs_{comp}/dAUROC_ci",
                     "historical": h, "historical_sd": None,
                     **{f"corrected_{v}": vals.get(v) for v in VARIANTS},
                     "changed_vs_v2_A6": (None if "v2" not in vals else bool(
                         (h[1] < 0) != (vals["v2"][1] < 0))),
                     "note": "historical unit = acquisitions mislabeled as "
                             "participants; corrected = participant draws"})

    # bootstrap balanced AUROC CI
    vals = {v: (S[v]["auroc_boot"]["participant_unit"]["auroc_ci95"]
                if S[v] else None) for v in VARIANTS}
    rows.append({"claim": "bootstrap_auroc_ci",
                 "historical": bb["adni"]["auroc_ci"], "historical_sd": None,
                 **{f"corrected_{v}": vals.get(v) for v in VARIANTS},
                 "changed_vs_v2_A6": None,
                 "note": "historical resampled acquisitions; corrected "
                         "participant unit (acquisition unit also stored)"})

    # fn capture (manuscript M47: 63% at 20% budget, 96% at 50%)
    for b, hv in [("20pct", 0.63), ("50pct", 0.96)]:
        vals = {v: (S[v]["fn_capture"][b]["fn_capture"]["mean"]
                    if S[v] else None) for v in VARIANTS}
        rows.append(row(f"fn_capture/{b}", hv, None, vals, kind="abs",
                        note="historical = manuscript-quoted value (CIP json "
                             "stores null for this field)"))

    # conformal realized error at alpha=0.10
    h = mlr["adni"]["block_a_conformal"]["0.10"]
    for k in ["exch_err", "shift_err"]:
        vals = {v: (M[v]["block_a_conformal"]["0.10"][k] if M[v] else None)
                for v in VARIANTS}
        rows.append(row(f"conformal/0.10/{k}", h[k], h.get(k + "_std"), vals))

    out = {
        "headline_variant": "v2 (addendum A1)",
        "a6_criteria": "changed if |v2 - hist| > 2*hist_sd (per-seed SD), or "
                       "sign flip, or categorical statement no longer true, or "
                       "CI-based statement no longer holds",
        "rows": rows,
    }
    (RES / "adni_original_vs_corrected.json").write_text(json.dumps(out, indent=2))
    import csv
    with open(RES / "adni_original_vs_corrected.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
                        for k, v in r.items()})
    n_changed = sum(1 for r in rows if r.get("changed_vs_v2_A6"))
    print(f"rows: {len(rows)}, changed vs V2 per A6: {n_changed}")


if __name__ == "__main__":
    main()
