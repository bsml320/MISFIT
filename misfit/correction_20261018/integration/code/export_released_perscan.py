"""Export the DUA-safe released per-scan array for the corrected V2 ADNI analysis.

Reads the server-only reruns/purged_v2_perscan_local.npz (contains ADNI
participant identifiers; never distributed) and writes
results/purged_v2_perscan_released.npz with the identifier column replaced by
an anonymous integer participant index (order of first appearance). Every
other channel is copied unchanged, so the released file supports an exact
recomputation of the corrected pooled headline (rho, AUROC per seed) and the
participant-blocked bootstrap, without any identifier.

Binding check: the recomputed per-seed pooled Spearman rho and failure AUROC
must equal the stratified rho_overall / auroc_overall block of
results/corrected_v2_summaries.json (np.std ddof=0 across seeds).
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "reruns" / "purged_v2_perscan_local.npz"
DST = ROOT / "results" / "purged_v2_perscan_released.npz"
SUMMARIES = ROOT / "results" / "corrected_v2_summaries.json"


def main() -> None:
    """Anonymize the participant column, binding-check, then write the released npz."""
    z = np.load(SRC, allow_pickle=True)
    ptids = z["ptids"]
    # Anonymous partition: integer index by order of first appearance.
    # np.unique's inverse numbers participants in sorted order; the loop below
    # renumbers them by order of first appearance in the scan sequence
    _, participant_index = np.unique(ptids, return_inverse=True)
    first_seen = {}
    remap = np.empty(len(np.unique(participant_index)), dtype=np.int64)
    next_id = 0
    out_idx = np.empty(len(ptids), dtype=np.int64)
    for i, p in enumerate(participant_index):
        if p not in first_seen:
            first_seen[p] = next_id
            next_id += 1
        out_idx[i] = first_seen[p]

    released = {"participant_index": out_idx}
    for key in z.files:
        if key == "ptids":
            continue
        released[key] = z[key]

    # Binding check against the verified corrected summaries, BEFORE writing:
    # the released file is only produced if it reproduces the corrected headline.
    summ = json.load(open(SUMMARIES))["stratified"]
    rhos, aurocs = [], []
    for s in range(5):
        tested = released[f"tested_s{s}"]
        m = released[f"misfit_s{s}"][tested]
        e = released[f"err_s{s}"][tested]
        rhos.append(spearmanr(m, e).statistic)
        aurocs.append(roc_auc_score((e > 0.5).astype(int), m))
    report = {
        "n_scans": int(len(out_idx)),
        "n_participants": int(out_idx.max() + 1),
        "rho_pooled_mean": float(np.mean(rhos)),
        "rho_pooled_std": float(np.std(rhos)),
        "auroc_mean": float(np.mean(aurocs)),
        "auroc_std": float(np.std(aurocs)),
        "binds_rho": bool(np.isclose(np.mean(rhos), summ["rho_overall"]["mean"], atol=1e-12)),
        "binds_auroc": bool(np.isclose(np.mean(aurocs), summ["auroc_overall"]["mean"], atol=1e-12)),
    }
    print(json.dumps(report, indent=2))
    if not (report["binds_rho"] and report["binds_auroc"]):
        raise SystemExit("BINDING FAILED: released array does not reproduce corrected_v2 headline")
    np.savez_compressed(DST, **released)
    print(f"wrote {DST}")


if __name__ == "__main__":
    main()
