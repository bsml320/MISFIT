"""Candidate-level binding of the CDR-stage analysis to the ORIGINAL OASIS-3 records.

Independently recomputes, from the raw OASIS-3 distribution files, every selection the
pipeline made, and checks it against the cohort tables actually used:
  (1) earliest-session choice per participant (candidates = that subject's sessions in
      oasis3_features.npz; ties at the minimum day flagged),
  (2) nearest-CDR assessment per chosen session (candidates = all UDS B4 visits of the
      subject; equidistant ties flagged, pandas-idxmin earlier-visit convention),
  (3) CDR value, scan-assessment gap, age, sex and binary label propagated into
      oasis3_mr_labels.csv, oasis3_features.npz and the participant export.
Writes aggregate verification (mismatch counts, tie counts, source hashes) to
results/provenance/oasis3_cdr_candidate_binding.json and the full local-only candidate
table (DUA: not for redistribution) to oasis3_cdr_candidates_local.csv.
"""
from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

import hashlib
import json
import re

import numpy as np
import pandas as pd

ROOT = _resolve_path("<REPO_ROOT>/data/oasis3")
DF = f"{ROOT}/all/OASIS3_data_files/OASIS3_data_files/scans"
RAW_CDR = (f"{DF}/UDSb4-Form_B4__Global_Staging__CDR__Standard_and_Supplemental/"
           "resources/csv/files/OASIS3_UDSb4_cdr.csv")
RAW_DEMO = f"{DF}/demo-demographics/resources/csv/files/OASIS3_demographics.csv"
RAW_MRJ = f"{DF}/MRI-json-MRI_json_information/resources/csv/files/OASIS3_MR_json.csv"
OUT = _resolve_path("<REPO_ROOT>/misfit/results/provenance")


def sha(p):
    """Stream a file in 1 MiB chunks and return its SHA-256 hex digest."""
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for ch in iter(lambda: fh.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


d = np.load(f"{ROOT}/oasis3_features.npz", allow_pickle=True)
subj = d["subjects"].astype(str)
sess = d["sessions"].astype(str)
day = np.array([int(re.search(r"_d(\d+)$", x).group(1)) for x in sess])
# Earliest session per participant; np.argmin takes the first index at the
# minimum day, matching the convention of the analysis pipeline being audited.
keep = np.array(sorted(np.where(subj == u)[0][np.argmin(day[subj == u])]
                       for u in np.unique(subj)))
chosen = dict(zip(subj[keep], sess[keep]))

lab = pd.read_csv(f"{ROOT}/oasis3_mr_labels.csv")
lab_by_sess = lab.set_index("session")
cdr_raw = pd.read_csv(RAW_CDR).dropna(subset=["days_to_visit", "CDRTOT"])
cdr_raw["days_to_visit"] = pd.to_numeric(cdr_raw["days_to_visit"], errors="coerce")
cdr_raw = cdr_raw.dropna(subset=["days_to_visit"])
cdr_by_sub = {s: g.sort_values("days_to_visit") for s, g in cdr_raw.groupby("OASISID")}
demo = pd.read_csv(RAW_DEMO).set_index("OASISID")

export = pd.read_csv(f"{OUT}/oasis3_cdr_participant_records.csv").set_index("session")

mm = {k: 0 for k in ("earliest_choice", "session_day_tie", "cdr_value", "cdr_gap",
                     "cdr_equidistant_tie", "equidistant_value_differs", "age",
                     "sex", "label", "npz_label", "npz_age", "npz_sex")}
cand_rows = []
for u in np.unique(subj):
    rows_u = np.where(subj == u)[0]
    days_u = day[rows_u]
    s_chosen = chosen[u]
    d_chosen = int(days_u.min())
    if int(day[sess == s_chosen][0]) != d_chosen:
        mm["earliest_choice"] += 1
    if (days_u == d_chosen).sum() > 1:
        mm["session_day_tie"] += 1
    g = cdr_by_sub.get(u)
    rec = lab_by_sess.loc[s_chosen]
    exp = export.loc[s_chosen]
    gaps = (g["days_to_visit"] - d_chosen).abs()
    i = gaps.idxmin()
    cdr_i, gap_i = float(g.loc[i, "CDRTOT"]), int(gaps.loc[i])
    tied = g[gaps == gaps.loc[i]]
    if len(tied) > 1:
        mm["cdr_equidistant_tie"] += 1
        if tied["CDRTOT"].nunique() > 1:
            mm["equidistant_value_differs"] += 1
    if cdr_i != float(rec["CDRTOT"]) or cdr_i != float(exp["cdr_raw"]):
        mm["cdr_value"] += 1
    if gap_i != int(rec["cdr_gap_days"]):
        mm["cdr_gap"] += 1
    age_i = float(g.loc[i, "age at visit"])
    if abs(age_i - float(rec["age"])) > 1e-6:
        mm["age"] += 1
    sex_raw = {1: "M", 2: "F"}.get(int(demo.loc[u, "GENDER"]))
    if sex_raw != str(rec["sex"]):
        mm["sex"] += 1
    if int(cdr_i >= 0.5) != int(rec["label_impaired"]):
        mm["label"] += 1
    npz_row = np.where(sess == s_chosen)[0][0]
    if int(d["label"][npz_row]) != int(cdr_i >= 0.5):
        mm["npz_label"] += 1
    if abs(float(d["age"][npz_row]) - age_i) > 1e-6:
        mm["npz_age"] += 1
    if {1: "M", 2: "F"}.get(int(demo.loc[u, "GENDER"])) != str(d["sex"][npz_row]).strip().upper()[0]:
        mm["npz_sex"] += 1
    cand_rows.append({
        "subject": u, "chosen_session": s_chosen, "chosen_day": d_chosen,
        "n_candidate_sessions": int(len(rows_u)),
        "candidate_days": ";".join(map(str, sorted(days_u.tolist()))),
        "n_cdr_visits": int(len(g)),
        "cdr_visit_days": ";".join(str(int(x)) for x in g["days_to_visit"]),
        "chosen_cdr_day": int(g.loc[i, "days_to_visit"]), "chosen_cdrtot": cdr_i,
        "gap_days": gap_i, "equidistant_tie": int(len(tied) > 1),
    })

pd.DataFrame(cand_rows).to_csv(f"{OUT}/oasis3_cdr_candidates_local.csv", index=False)
res = {
    "note": "independent recomputation from the raw OASIS-3 distribution files; "
            "mismatch counts are against oasis3_mr_labels.csv, oasis3_features.npz and "
            "the participant export. Candidate table oasis3_cdr_candidates_local.csv is "
            "LOCAL-ONLY (OASIS-3 DUA), not packaged in the release.",
    "n_participants": int(len(cand_rows)),
    "mismatches": mm,
    "selection_conventions": {
        "session": "earliest day among the subject's sessions in oasis3_features.npz "
                   "(np.argmin = first index at the minimum; day ties counted above)",
        "cdr": "nearest UDS B4 visit by |days_to_visit - session day|; pandas idxmin = "
               "tie breaks to the earlier visit (equidistant ties counted above, and "
               "counted separately when the tied visits disagree on CDRTOT)",
        "sex": "raw demographics GENDER 1->M, 2->F; analysis coding male=1 via "
               "startswith('M')", "label": "CDRTOT>=0.5 impaired",
        "age": "age at the matched CDR visit",
    },
    "source_hashes_sha256": {
        "OASIS3_UDSb4_cdr.csv": sha(RAW_CDR),
        "OASIS3_demographics.csv": sha(RAW_DEMO),
        "OASIS3_MR_json.csv": sha(RAW_MRJ),
        "oasis3_mr_labels.csv": sha(f"{ROOT}/oasis3_mr_labels.csv"),
        "oasis3_features.npz": sha(f"{ROOT}/oasis3_features.npz"),
    },
}
json.dump(res, open(f"{OUT}/oasis3_cdr_candidate_binding.json", "w"), indent=2)
print(json.dumps({k: v for k, v in res.items() if k != "source_hashes_sha256"},
                 indent=1)[:1500])
