"""TASK 8: Audit internal consistency.

From audit/adni_linkage_local.csv alone, recompute the cohort identity counts
(731 scans, 547 participants, 156 multi-protocol) and every per-fold quantity
in audit/adni_fold_audit.csv under the HISTORICAL (v0) fold construction
(train = all scans from the other 7 protocols, no purge), and compare.
Aggregates only; no PTIDs are emitted."""

from misfit.paths import resolve_path as _resolve_path
import csv
import json
from collections import defaultdict

COR = _resolve_path("<REPO_ROOT>/misfit/correction_20261018")

rows = list(csv.DictReader(open(f"{COR}/audit/adni_linkage_local.csv")))
out = {"task": 8, "n_scans": len(rows)}

by_pt = defaultdict(list)
for r in rows:
    by_pt[r["ptid"]].append(r)
out["n_participants"] = len(by_pt)
out["n_multi_protocol"] = sum(
    1 for v in by_pt.values() if len({x["protocol"] for x in v}) >= 2)
out["n_scans_multi"] = sum(
    len(v) for v in by_pt.values() if len({x["protocol"] for x in v}) >= 2)
out["pass_counts"] = bool(out["n_scans"] == 731 and out["n_participants"] == 547
                          and out["n_multi_protocol"] == 156
                          and out["n_scans_multi"] == 340)

audit = list(csv.DictReader(open(f"{COR}/audit/adni_fold_audit.csv")))
protocols = sorted({r["protocol"] for r in rows})
out["protocols_match_audit"] = bool(
    protocols == sorted(a["held_out_protocol"] for a in audit))

mism = []
derived = {}
sum_overlap_test_acq = 0
# rebuild every per-fold quantity from the linkage rows under the historical
# v0 construction: train = all scans from the other protocols, no purge
for a in audit:
    e = a["held_out_protocol"]
    test = [r for r in rows if r["protocol"] == e]
    train = [r for r in rows if r["protocol"] != e]
    tpt = {r["ptid"] for r in test}
    rpt = {r["ptid"] for r in train}
    both = tpt & rpt
    # label conflict among overlapping participants (binary label column)
    conflict = 0
    for p in both:
        labs = {x["label"] for x in by_pt[p]}
        if len(labs) > 1:
            conflict += 1
    mine = {
        "n_test_acq": len(test),
        "n_test_participants": len(tpt),
        "n_train_acq": len(train),
        "n_train_participants": len(rpt),
        "repeated_acq_within_test": len(test) - len(tpt),
        "repeated_acq_within_train": len(train) - len(rpt),
        "participants_in_both_train_and_test": len(both),
        "pct_test_participants_also_in_train": round(100 * len(both) / len(tpt), 1),
        "test_acq_with_participant_in_train": sum(
            1 for r in test if r["ptid"] in rpt),
        "train_acq_belonging_to_test_participants": sum(
            1 for r in train if r["ptid"] in tpt),
        "overlapping_participants_with_label_conflict": conflict,
    }
    sum_overlap_test_acq += mine["test_acq_with_participant_in_train"]
    derived[e] = mine
    for k, v in mine.items():
        got = a[k]
        ref = float(got) if "." in got else int(got)
        if abs(ref - v) > (0.051 if k.startswith("pct") else 0):  # pct fields stored rounded to 1 decimal
            mism.append({"fold": e, "field": k, "audit": ref, "mine": v})

out["per_fold_derived"] = derived
out["per_fold_mismatches"] = mism
out["sum_test_acq_with_participant_in_train"] = sum_overlap_test_acq
out["pass_340_total_overlap_test_acq"] = bool(sum_overlap_test_acq == 340)
out["PASS"] = bool(out["pass_counts"] and out["protocols_match_audit"]
                   and not mism and out["pass_340_total_overlap_test_acq"])
print(json.dumps(out, indent=2))
json.dump(out, open(f"{COR}/verify/out_task8.json", "w"), indent=2)
