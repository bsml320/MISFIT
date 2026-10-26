"""TASK 2: Purge verification from the audit linkage file.

From audit/adni_linkage_local.csv (one row per scan: protocol, ptid, site_id),
independently reconstruct the v1/v2 masks (train pool for fold e =
site != e AND ptid not in fold-e test ptids) and verify:
  (a) zero participant overlap between every fold's training pool and test set;
  (b) exactly 340 of 731 evaluation scans belong to multi-protocol participants;
  (c) n_refits in the perscan npz: scan of a k-copy participant has 8-k
      contributing refits in v1/v2, and 7 for every scan in v0.
No PTIDs are printed; aggregates only.
"""

from misfit.paths import resolve_path as _resolve_path
import csv
import json
import numpy as np
from collections import defaultdict

COR = _resolve_path("<REPO_ROOT>/misfit/correction_20261018")

rows = list(csv.DictReader(open(f"{COR}/audit/adni_linkage_local.csv")))
n = len(rows)
site = np.array([int(r["site_id"]) for r in rows])
ptid = np.array([r["ptid"] for r in rows])
folds = sorted(set(site.tolist()))

out = {"task": 2, "n_scans": n, "n_folds": len(folds)}

# (a) reconstruct masks and check overlap
overlap_counts = {}
train_pool_sizes = {}
for e in folds:
    test_pt = set(ptid[site == e].tolist())
    train_mask = (site != e) & np.array([p not in test_pt for p in ptid])
    train_pt = set(ptid[train_mask].tolist())
    overlap_counts[e] = len(train_pt & test_pt)
    train_pool_sizes[e] = int(train_mask.sum())
out["per_fold_train_test_participant_overlap_v1v2"] = overlap_counts
out["per_fold_purged_train_pool_size"] = train_pool_sizes
out["pass_zero_overlap"] = bool(all(v == 0 for v in overlap_counts.values()))

# (b) multi-protocol membership
prot_per_pt = defaultdict(set)
scans_per_pt = defaultdict(int)
for r in rows:
    prot_per_pt[r["ptid"]].add(int(r["site_id"]))
    scans_per_pt[r["ptid"]] += 1
k_distinct = {p: len(s) for p, s in prot_per_pt.items()}
# sanity: no participant has two scans in the same protocol
out["any_participant_with_2_scans_same_protocol"] = bool(
    any(scans_per_pt[p] != k_distinct[p] for p in scans_per_pt))
multi_pt = {p for p, k in k_distinct.items() if k >= 2}
n_scans_multi = int(sum(1 for p in ptid if p in multi_pt))
out["n_participants"] = len(k_distinct)
out["n_multi_protocol_participants"] = len(multi_pt)
out["n_scans_from_multi_protocol_participants"] = n_scans_multi
out["pass_340_of_731"] = bool(n_scans_multi == 340 and n == 731)

# n_copies column in the audit CSV should equal my recomputed k
csv_k = np.array([int(r["n_copies"]) for r in rows])
my_k = np.array([k_distinct[p] for p in ptid])
out["pass_csv_n_copies_matches_recomputed"] = bool((csv_k == my_k).all())

# (c) n_refits verification against the perscan npz files
# expected refits for scan i = number of folds whose reconstructed training
# pool contains scan i
expected_v1v2 = np.zeros(n, dtype=int)
for e in folds:
    test_pt = set(ptid[site == e].tolist())
    in_pool = (site != e) & np.array([p not in test_pt for p in ptid])
    expected_v1v2 += in_pool.astype(int)
# closed form: 8 - k(distinct protocols of the participant)
closed_form = 8 - my_k
out["pass_expected_equals_8_minus_k"] = bool((expected_v1v2 == closed_form).all())

res = {}
for tag, exp in [("purged_v0", np.full(n, 7)),
                 ("purged_v1", expected_v1v2),
                 ("purged_v2", expected_v1v2)]:
    f = np.load(f"{COR}/reruns/{tag}_perscan_local.npz", allow_pickle=False)
    # row alignment: npz ptids/site must equal the audit CSV rows elementwise
    aligned = bool((f["ptids"] == ptid).all() and (f["site"] == site).all())
    match = bool((f["n_refits"] == exp).all())
    res[tag] = {"row_aligned_with_audit_csv": aligned,
                "n_refits_matches_expected": match,
                "n_refits_hist": {int(k): int(v) for k, v in
                                  zip(*np.unique(f["n_refits"],
                                                 return_counts=True))}}
out["n_refits_check"] = res
out["pass_n_refits"] = bool(all(r["row_aligned_with_audit_csv"]
                                and r["n_refits_matches_expected"]
                                for r in res.values()))

out["PASS"] = bool(out["pass_zero_overlap"] and out["pass_340_of_731"]
                   and out["pass_n_refits"]
                   and out["pass_expected_equals_8_minus_k"]
                   and not out["any_participant_with_2_scans_same_protocol"])

print(json.dumps(out, indent=2))
json.dump(out, open(f"{COR}/verify/out_task2.json", "w"), indent=2)
