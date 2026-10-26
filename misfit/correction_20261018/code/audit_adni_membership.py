"""Audit: actual train/test membership of the historical ADNI MISFIT analysis.

Replicates the eligibility logic of
misfit/data/adni_loader.py::load_adni_smri (git-tracked,
unmodified) WITHOUT loading volumes, reconstructs the realized leave-one-
protocol-out partitions used by run_honest_estimator_table.py /
run_ml_reliability_media.py (both mask by site_ids only), and binds the
reconstruction to the archived per-seed prediction record
results/perseed_misfit_two_phases.npz (site + label sequence must match).

Facts this script establishes, per held-out environment:
  - acquisitions and unique participants in train / test,
  - participants present in BOTH train and test (P_e overlap),
  - test acquisitions whose participant also has >=1 training acquisition,
  - training acquisitions belonging to test participants,
  - binary-label conflicts between a participant's test copy and train copies,
  - repeated acquisitions within each partition.
Plus identity validation (PTID regex, sex consistency, age/phase ordering),
label-table duplicate keys, and the supplement's cohort-flow counts
(790 -> 751 -> 731). There is no validation split (TwoStageTrainer is called
with val_data=None; early stopping monitors training loss), so model-selection
overlap is structurally identical to training overlap. PCA (d=50) and global
z-normalization are fit once on all retained acquisitions, so every test
acquisition participates in learned preprocessing by construction.

Outputs (aggregate, shareable):
  audit/adni_membership_audit.json, audit/adni_fold_audit.csv
Output (participant-level, DUA local-only, never packaged):
  audit/adni_linkage_local.csv
"""

from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(_resolve_path("<REPO_ROOT>"))
DATA = ROOT / "data/adni/smri"
COR = ROOT / "misfit/correction_20261018"
PERSEED = ROOT / "misfit/results/perseed_misfit_two_phases.npz"

PROTOCOLS = [
    "ADNI2-GE-initial", "ADNI2-Phillips-initial", "ADNI2-Siemens-initial",
    "ADNI3-GE-initial", "ADNI3-Phillips-initial", "ADNI3-Siemens-initial",
    "ADNIGO-GE-initial", "ADNIGO-Phillips-initial", "ADNIGO-Siemens-initial",
]
IMPAIRED = {"AD", "MCI", "EMCI", "LMCI"}
MIN_SUBJECTS = 30
PHASE_ORDER = {"ADNIGO": 0, "ADNI2": 1, "ADNI3": 2}
PTID_RE = re.compile(r"^\d{3}_S_\d{4}$")


def main() -> None:
    """Reconstruct the historical cohort and LOSO folds, validate identities, write outputs.

    Stages: label-table first-row dedup, cohort flow over eligible smwc1
    volumes, min-30 protocol eligibility, per-participant identity checks,
    binding against the archived per-seed predictions, the per-fold membership
    table, and the three output files listed in the module docstring.
    """
    labels = pd.read_csv(DATA / "labels/combined.csv")
    # loader keeps the FIRST row per (dataset_id, file_name)
    key = list(zip(labels["dataset_id"].astype(str).str.strip(),
                   labels["file_name"].astype(str).str.strip()))
    labels = labels.assign(_key=key)
    dup_keys = int(labels["_key"].duplicated().sum())
    first = {}
    for rec in labels.to_dict("records"):
        first.setdefault(rec["_key"], rec)
    # conflicting Group among duplicate keys
    grp_nunique = labels.groupby("_key")["Group"].nunique()
    conflict_dup_keys = int((grp_nunique > 1).sum())

    rows = []
    n_smwc1_with_label = 0
    n_smwc1_no_label = 0
    n_label_bad_group = 0
    for proto in PROTOCOLS:
        pdir = DATA / "nii" / proto
        if not pdir.exists():
            continue
        for sd in sorted(pdir.iterdir()):
            if not (sd / "smwc1pT1.nii").exists():
                continue
            k = (proto, sd.name)
            if k not in first:
                n_smwc1_no_label += 1
                continue
            n_smwc1_with_label += 1
            r = first[k]
            group = str(r["Group"]).strip()
            if group not in {"CN"} | IMPAIRED:
                n_label_bad_group += 1
                continue
            rows.append({
                "protocol": proto, "ptid": sd.name, "group": group,
                "label": 0 if group == "CN" else 1,
                "sex": str(r.get("Sex", "")).strip(),
                "age": float(r.get("Age", np.nan)),
                "phase": proto.split("-")[0],
            })
    df = pd.DataFrame(rows)
    per_proto = df.groupby("protocol").size()
    kept = [p for p in PROTOCOLS if per_proto.get(p, 0) >= MIN_SUBJECTS]
    dropped = {p: int(per_proto.get(p, 0)) for p in PROTOCOLS
               if 0 < per_proto.get(p, 0) < MIN_SUBJECTS}
    df = df[df["protocol"].isin(kept)].reset_index(drop=True)
    # loader order: PROTOCOLS order, then sorted dir name within protocol
    df["site_id"] = df["protocol"].map({p: i for i, p in enumerate(kept)})

    # ---- identity validation ----
    bad_ptid = sorted(set(p for p in df["ptid"] if not PTID_RE.match(p)))
    sex_conflicts, age_violations, label_flips_pairs = [], [], []
    for ptid, g in df.groupby("ptid"):
        if g["sex"].nunique() > 1:
            sex_conflicts.append(ptid)
        gg = g.sort_values("phase", key=lambda s: s.map(PHASE_ORDER))
        ages = gg["age"].tolist()
        if any(b < a - 0.5 for a, b in zip(ages, ages[1:])):
            age_violations.append(ptid)
        if g["label"].nunique() > 1:
            label_flips_pairs.append(ptid)

    copies = df.groupby("ptid").size()
    multi = copies[copies >= 2]
    phase_combo = Counter(
        tuple(sorted(g["phase"])) for _, g in df.groupby("ptid") if len(g) >= 2)

    # ---- binding to the archived per-seed prediction record ----
    binding = {}
    if PERSEED.exists():
        d = np.load(PERSEED)
        binding["n_matches"] = bool(len(d["adni_site"]) == len(df))
        binding["site_sequence_identical"] = bool(
            np.array_equal(d["adni_site"], df["site_id"].to_numpy()))
        binding["label_sequence_identical"] = bool(
            np.array_equal(d["adni_label"], df["label"].to_numpy()))
        # descriptive: held-out error rate by overlap status, per seed
        in_multi = df["ptid"].isin(multi.index).to_numpy()
        err_overlap, err_single = [], []
        for s in range(d["adni_err_A"].shape[0]):
            e = d["adni_err_A"][s]
            err_overlap.append(float(np.nanmean(e[in_multi])))
            err_single.append(float(np.nanmean(e[~in_multi])))
        binding["heldout_err_rate_overlap_participants_mean"] = float(np.mean(err_overlap))
        binding["heldout_err_rate_single_protocol_participants_mean"] = float(np.mean(err_single))
        binding["per_seed_err_overlap"] = err_overlap
        binding["per_seed_err_single"] = err_single

    # ---- per-fold actual membership ----
    fold_rows = []
    for e, proto in enumerate(kept):
        te = df[df["site_id"] == e]
        tr = df[df["site_id"] != e]
        te_p, tr_p = set(te["ptid"]), set(tr["ptid"])
        shared = te_p & tr_p
        te_scans_p_in_train = int(te["ptid"].isin(tr_p).sum())
        tr_scans_of_te_p = int(tr["ptid"].isin(te_p).sum())
        # label conflicts across the partition boundary, per overlapping participant
        n_conflict = 0
        for ptid in shared:
            lte = set(te.loc[te["ptid"] == ptid, "label"])
            ltr = set(tr.loc[tr["ptid"] == ptid, "label"])
            if lte != ltr or len(lte | ltr) > 1:
                n_conflict += 1
        fold_rows.append({
            "held_out_protocol": proto,
            "n_test_acq": len(te), "n_test_participants": len(te_p),
            "n_train_acq": len(tr), "n_train_participants": len(tr_p),
            "repeated_acq_within_test": int(len(te) - len(te_p)),
            "repeated_acq_within_train": int(len(tr) - len(tr_p)),
            "participants_in_both_train_and_test": len(shared),
            "pct_test_participants_also_in_train": round(100 * len(shared) / len(te_p), 1),
            "test_acq_with_participant_in_train": te_scans_p_in_train,
            "train_acq_belonging_to_test_participants": tr_scans_of_te_p,
            "overlapping_participants_with_label_conflict": n_conflict,
            "participants_in_val_and_test": "no validation split (val_data=None)",
        })
    folds = pd.DataFrame(fold_rows)

    out = {
        "generated": "2026-10-18",
        "reconstruction_of": "misfit/data/adni_loader.py::load_adni_smri"
                             " + site-mask LOSO in run_honest_estimator_table.py",
        "claims_under_test": {"n_total_claimed": 731, "n_participants_claimed": 547,
                              "n_multi_protocol_claimed": 156},
        "cohort_flow": {
            "smwc1_with_label_record": n_smwc1_with_label,
            "excluded_ineligible_group": n_label_bad_group,
            "smwc1_without_label_record": n_smwc1_no_label,
            "protocols_dropped_lt30": dropped,
            "retained_acquisitions": int(len(df)),
            "retained_protocols": len(kept),
        },
        "identity": {
            "unique_participants": int(df["ptid"].nunique()),
            "participants_in_ge2_protocols": int(len(multi)),
            "acquisitions_from_multi_protocol_participants": int(copies[copies >= 2].sum()),
            "copies_distribution": {str(k): int(v) for k, v in
                                    Counter(copies.values).items()},
            "phase_combinations_multi": {"+".join(k): v for k, v in
                                         sorted(phase_combo.items())},
            "ptid_format_violations": bad_ptid,
            "sex_conflicts_across_copies": len(sex_conflicts),
            "age_decreasing_with_phase_order": len(age_violations),
            "participants_with_binary_label_conflict_across_copies": len(label_flips_pairs),
        },
        "label_table": {
            "duplicate_keys_in_combined_csv": int(dup_keys),
            "duplicate_keys_with_conflicting_group": conflict_dup_keys,
        },
        "preprocessing": {
            "pca": "fit once on all retained acquisitions (d=50, SVD, float64)",
            "normalization": "global z-norm over all retained acquisitions",
            "test_acq_in_learned_preprocessing": int(len(df)),
            "note": "every test acquisition participates in PCA + z-norm by construction",
        },
        "model_selection": "none: TwoStageTrainer called with val_data=None; "
                           "early stopping monitors training loss",
        "seeds": "SEED_BASE=42, N_SEEDS=5 -> seeds 42..46; partitions seed-independent",
        "binding_to_archived_predictions": binding,
        "fold_totals": {
            "total_test_acq_with_participant_in_train":
                int(folds["test_acq_with_participant_in_train"].sum()),
            "pct_of_all_evaluation_acq": round(
                100 * folds["test_acq_with_participant_in_train"].sum() / len(df), 1),
        },
    }
    (COR / "audit").mkdir(exist_ok=True)
    (COR / "audit/adni_membership_audit.json").write_text(json.dumps(out, indent=2))
    folds.to_csv(COR / "audit/adni_fold_audit.csv", index=False)
    df.assign(n_copies=df["ptid"].map(copies)).to_csv(
        COR / "audit/adni_linkage_local.csv", index=False)
    print(json.dumps(out, indent=2))
    print(folds.to_string(index=False))


if __name__ == "__main__":
    main()
