"""Build a session-level label table for OASIS-3 MR sessions (MedIA external cohort).

Matches each MR session (OAS3XXXX_MR_dYYYY, day = days-from-entry) to the nearest CDR
clinical visit for the same subject, attaches demographics + scanner/protocol, and derives
a binary impaired-vs-normal label. Pure local CSV processing (no XNAT credentials).

Label convention (matches the paper's CDR-derived OASIS-1 setup): CDRTOT==0 -> control(0),
CDRTOT>=0.5 -> impaired(1). Also records CDRTOT so a CDR>=1 (frank dementia) cut is available.
Output: <REPO_ROOT>/data/oasis3/oasis3_mr_labels.csv  + a printed summary.
"""
from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(_resolve_path("<REPO_ROOT>/data/oasis3/all"))
DF = ROOT / "OASIS3_data_files/OASIS3_data_files/scans"
OUT = Path(_resolve_path("<REPO_ROOT>/data/oasis3/oasis3_mr_labels.csv"))
CDR = DF / "UDSb4-Form_B4__Global_Staging__CDR__Standard_and_Supplemental/resources/csv/files/OASIS3_UDSb4_cdr.csv"
DEMO = DF / "demo-demographics/resources/csv/files/OASIS3_demographics.csv"
MRJ = DF / "MRI-json-MRI_json_information/resources/csv/files/OASIS3_MR_json.csv"
WINDOW = 365  # days: flag matches whose nearest CDR is farther than this


def day_of(label: str) -> int | None:
    """Days-from-entry parsed from a session label suffix (_dNNNN), else None."""
    m = re.search(r"_d(\d+)$", str(label))
    return int(m.group(1)) if m else None


def main() -> None:
    """Build the session table, match CDR visits, attach demographics, write the CSV.

    Nearest-CDR matching uses |days_to_visit - session day|; pandas idxmin
    breaks equidistant ties toward the earlier visit. Sessions whose nearest
    CDR lies beyond WINDOW days are kept in the table but excluded from the
    printed usable counts.
    """
    cdr = pd.read_csv(CDR)
    demo = pd.read_csv(DEMO)
    mrj = pd.read_csv(MRJ, low_memory=False)

    # --- one row per MR SESSION from the scan-level json table ---
    mrj = mrj.rename(columns={"scan category": "scan_category"})
    mrj["session"] = mrj["label"]
    mrj["subject"] = mrj["subject_id"]
    is_t1 = (mrj["scan_category"].astype(str).str.lower().eq("t1w")) | (
        mrj["SeriesDescription"].astype(str).str.contains("mprage|t1", case=False, na=False)
    )
    t1_by_session = is_t1.groupby(mrj["session"]).any()
    sess = mrj.groupby("session").agg(
        subject=("subject", "first"),
        MagneticFieldStrength=("MagneticFieldStrength", "first"),
        Manufacturer=("Manufacturer", "first"),
        Model=("ManufacturersModelName", "first"),
    ).reset_index()
    sess["day"] = sess["session"].map(day_of)
    sess["has_t1w"] = sess["session"].map(t1_by_session).fillna(False)
    sess = sess.dropna(subset=["day"]).astype({"day": int})

    # --- nearest CDR visit per session ---
    cdr = cdr.dropna(subset=["days_to_visit", "CDRTOT"]).copy()
    cdr["days_to_visit"] = pd.to_numeric(cdr["days_to_visit"], errors="coerce")
    cdr = cdr.dropna(subset=["days_to_visit"])
    cdr_by_sub = {sid: g.sort_values("days_to_visit") for sid, g in cdr.groupby("OASISID")}

    rows = []
    for r in sess.itertuples(index=False):
        g = cdr_by_sub.get(r.subject)
        if g is None or g.empty:
            cdrtot = gap = age = np.nan
        else:
            i = (g["days_to_visit"] - r.day).abs().idxmin()
            cdrtot = float(g.loc[i, "CDRTOT"])
            gap = int(abs(g.loc[i, "days_to_visit"] - r.day))
            age = float(g.loc[i, "age at visit"]) if "age at visit" in g.columns else np.nan
        rows.append({**r._asdict(), "cdr_gap_days": gap, "CDRTOT": cdrtot, "age": age})
    out = pd.DataFrame(rows)

    # --- demographics (sex, APOE, education) ---
    demo_small = demo[["OASISID", "GENDER", "EDUC", "APOE"]].rename(columns={"OASISID": "subject"})
    out = out.merge(demo_small, on="subject", how="left")
    out["sex"] = out["GENDER"].map({1: "M", 2: "F"})

    # --- labels ---
    out["label_impaired"] = (out["CDRTOT"] >= 0.5).astype("Int64")   # 0=normal, 1=impaired
    out["label_dementia"] = (out["CDRTOT"] >= 1.0).astype("Int64")   # frank dementia cut
    out.loc[out["CDRTOT"].isna(), ["label_impaired", "label_dementia"]] = pd.NA

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    # --- summary ---
    n = len(out)
    lab = out.dropna(subset=["CDRTOT"])
    close = lab[lab["cdr_gap_days"] <= WINDOW]
    print(f"MR sessions total:            {n}")
    print(f"  with a CDR match:           {len(lab)}")
    print(f"  CDR within +/-{WINDOW}d:        {len(close)}")
    print(f"  with a T1w scan:            {int(out['has_t1w'].sum())}")
    print(f"  T1w AND CDR within {WINDOW}d:   {int((close['has_t1w']).sum())}   <-- usable for the paper")
    print(f"\nLabel balance (CDR within {WINDOW}d, CDR>=0.5 = impaired):")
    print(close["label_impaired"].value_counts(dropna=False).rename({0: "normal", 1: "impaired"}))
    print(f"\ncdr_gap_days distribution (matched): median={lab['cdr_gap_days'].median():.0f} "
          f"p90={lab['cdr_gap_days'].quantile(0.9):.0f} max={lab['cdr_gap_days'].max():.0f}")
    print("\nScanner environments (field strength x manufacturer x model), T1w+CDR-close sessions:")
    env = close[close["has_t1w"]].groupby(
        ["MagneticFieldStrength", "Manufacturer", "Model"]).size().sort_values(ascending=False)
    print(env.head(15).to_string())
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
