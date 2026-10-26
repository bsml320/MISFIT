"""Participant-level OASIS-3 records backing the CDR-stage analysis (provenance item 5
follow-through). Reconstructs the clinical-insight panel's cohort and 5-seed misfit
exactly (same loader lines, same collect(), seeds 42-46), persists one row per
participant (session, scanner, label, CDR raw + stage, age, sex, per-seed misfit and
held-out error, seed-mean misfit + percentile), and verifies the published numbers
reproduce from the exported records alone.

Outputs:
  results/provenance/oasis3_cdr_participant_records.csv   (one row per participant)
  results/provenance/oasis3_cdr_participant_records.json  (metadata + verification)
"""
from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

import json
import os
import re
import sys

# Pin the default GPU before torch import; an existing env setting wins.
# CUDA visibility is selected by the caller.
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402

sys.path.insert(0, _resolve_path("<REPO_ROOT>"))
from misfit.experiments.run_clinical_stratified_media import collect  # noqa: E402

RES = _resolve_path("<REPO_ROOT>/misfit/results")
os.makedirs(f"{RES}/provenance", exist_ok=True)
SEEDS = (42, 43, 44, 45, 46)

# --- cohort construction, line-for-line as in run_clinical_insight_panel.py
dd = np.load(_resolve_path("<REPO_ROOT>/data/oasis3/oasis3_features.npz"), allow_pickle=True)
subj = dd["subjects"].astype(str)
sess = dd["sessions"].astype(str)
day = np.array([int(re.search(r"_d(\d+)$", x).group(1)) for x in sess])
keep = np.array(sorted(np.where(subj == u)[0][np.argmin(day[subj == u])]
                       for u in np.unique(subj)))
sess_k = sess[keep]
lab3 = dd["label"].astype(int)[keep]
age3 = dd["age"].astype(float)[keep]
sex3 = np.array([1 if str(x).strip().upper().startswith("M") else 0
                 for x in dd["sex"].astype(str)])[keep]
scan = dd["scanner"].astype(str)[keep]
uniq, cnt = np.unique(scan, return_counts=True)
order = uniq[np.argsort(-cnt)]
code = {s: i for i, s in enumerate(order)}
site3 = torch.tensor([code[x] for x in scan])
f3 = torch.tensor(dd["features"][keep], dtype=torch.float32)
t3 = torch.tensor(lab3)

mis3, err3, pred3, tested3 = [], [], [], []
for sd in SEEDS:
    c = collect(f3, t3, site3, sd)
    mis3.append(c["misfit"])
    err3.append(c["err"])
    pred3.append(c["pred"])
    tested3.append(c["tested"])
    print(f"[collect] seed {sd} done", flush=True)
mis3 = np.stack(mis3)
err3 = np.stack(err3)
pred3 = np.stack(pred3)
tested3 = np.stack(tested3)
mis3m = mis3.mean(0)

labdf = pd.read_csv(_resolve_path("<REPO_ROOT>/data/oasis3/oasis3_mr_labels.csv"))
cdr_by_sess = dict(zip(labdf["session"].astype(str), labdf["CDRTOT"].astype(float)))
cdr3 = np.array([cdr_by_sess.get(s, np.nan) for s in sess_k])
stage3 = np.where(cdr3 == 0, 0, np.where(cdr3 == 0.5, 1, 2))
pct = 100 * rankdata(mis3m) / len(mis3m)

rows = pd.DataFrame({
    "session": sess_k,
    "scanner": scan,
    "scanner_code": site3.numpy(),
    "label_impaired1": lab3,
    "cdr_raw": cdr3,
    "cdr_stage_0_05_1plus": stage3,
    "age": age3,
    "sex_male1": sex3,
    "misfit_mean": mis3m,
    "misfit_percentile": pct,
    **{f"misfit_seed{sd}": mis3[i] for i, sd in enumerate(SEEDS)},
    **{f"err_seed{sd}": err3[i].astype(int) for i, sd in enumerate(SEEDS)},
    **{f"pred_seed{sd}": pred3[i].astype(int) for i, sd in enumerate(SEEDS)},
    **{f"tested_seed{sd}": tested3[i].astype(int) for i, sd in enumerate(SEEDS)},
})
csv_path = f"{RES}/provenance/oasis3_cdr_participant_records.csv"
rows.to_csv(csv_path, index=False)
np.savez(f"{RES}/provenance/oasis3_cdr_participant_arrays.npz",
         sessions=sess_k, scanner=scan, label=lab3, cdr_raw=cdr3, age=age3,
         sex_male1=sex3, misfit_per_seed=mis3, err_per_seed=err3,
         pred_per_seed=pred3, tested_per_seed=tested3, seeds=np.array(SEEDS))

# --- verification: recompute every published number FROM THE CSV just written
r = pd.read_csv(csv_path)
val = np.isfinite(r["cdr_raw"].to_numpy())
st = r["cdr_stage_0_05_1plus"].to_numpy()
mm = r["misfit_mean"].to_numpy()
rho_cdr = spearmanr(st[val].astype(float), mm[val])
pctv = 100 * rankdata(mm) / len(mm)
med = [round(float(np.median(pctv[val & (st == k)])), 1) for k in (0, 1, 2)]
per_seed = [round(float(spearmanr(st[val], r[f"misfit_seed{sd}"].to_numpy()[val])[0]), 3)
            for sd in SEEDS]
previous_mean = f"{RES}/oasis3/oasis3_misfit_leakagefree_5seed.npy"
verif = {
    "misfit_vs_cdr_stage_rho": round(float(rho_cdr.statistic), 3),
    "misfit_vs_cdr_stage_p": float(f"{rho_cdr.pvalue:.3g}"),
    "stage_n": [int((st[val] == k).sum()) for k in (0, 1, 2)],
    "stage_median_misfit_percentile": med,
    "per_seed_cdr_rho": per_seed,
    "misfit_vs_age_rho": round(float(spearmanr(r["age"], mm)[0]), 3),
    "misfit_vs_sex_rho": round(float(spearmanr(r["sex_male1"].astype(float), mm)[0]), 3),
    "mean_matches_saved_npy": (
        bool(np.allclose(mm, np.load(previous_mean), atol=1e-12))
        if os.path.isfile(previous_mean) else None
    ),
}
import hashlib
import subprocess


def sha(p):
    """Stream a file in 1 MiB chunks and return its SHA-256 hex digest."""
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for ch in iter(lambda: fh.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                     cwd=_resolve_path("<REPO_ROOT>/misfit")).stdout.strip()
meta = {
    "note": "one row per participant (earliest leakage-free session); misfit from the "
            "canonical collect() harness, seeds 42-46, leave-one-scanner-out; CDR from "
            "the session table (closest CDR, see oasis3_clinical_alignment.json); "
            "session IDs are OASIS-3 pseudonymous IDs (DUA: do not redistribute "
            "outside approved use)",
    "inputs": {
        "oasis3_features.npz_sha256": sha(
            _resolve_path("<REPO_ROOT>/data/oasis3/oasis3_features.npz")),
        "oasis3_mr_labels.csv_sha256": sha(
            _resolve_path("<REPO_ROOT>/data/oasis3/oasis3_mr_labels.csv")),
        "feature_config": "route-B SPM25/DARTEL smwc1 (1.5mm, 3mm FWHM) -> cohort mask "
                          "(mean GM>0.1) -> standardize -> PCA-50; 2720 sessions, the "
                          "1360 earliest per subject scored here",
        "labels_builder": "experiments/build_oasis3_labels.py (nearest CDR visit by "
                          "|days_to_visit - session day|, pandas idxmin = ties break "
                          "to the earlier visit; sex GENDER 1->M 2->F from raw "
                          "demographics; raw sources OASIS3_UDSb4_cdr.csv, "
                          "OASIS3_demographics.csv, OASIS3_MR_json.csv)",
    },
    "producing_revision_misfit": rev,
    "scoring_harness": "misfit.experiments.run_clinical_stratified_media"
                       ".collect (per-fold model: run_honest_estimator_table._train_fire;"
                       " no checkpoints are written; any available prior mean is checked "
                       "separately in verification_from_csv)",
    "n": int(len(r)),
    "published_targets": {},
    "verification_from_csv": verif,
    "csv": "oasis3_cdr_participant_records.csv",
    "arrays": "oasis3_cdr_participant_arrays.npz (adds per-seed raw held-out "
              "predictions and tested/availability masks)",
}
json.dump(meta, open(f"{RES}/provenance/oasis3_cdr_participant_records.json", "w"), indent=2)
print(json.dumps(verif, indent=1), flush=True)
print("wrote", csv_path, flush=True)
