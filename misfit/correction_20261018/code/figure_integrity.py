"""Figure-integrity report (deliverable 10): re-hash every figure asset and
figure source file, compare with the pre-correction snapshot, and list the
figures whose displayed content is contradicted by corrected evidence
(directive section 7: those figures are left unchanged and flagged UNRESOLVED).
"""

from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

import hashlib
import json
from pathlib import Path

ROOT = Path(_resolve_path("<REPO_ROOT>"))
MIA = ROOT / "misfit"
COR = MIA / "correction_20261018"

# from report/dependency_inventory.md section 3 (included figures built from
# ADNI LOSO numbers), with the contradicted content spelled out
AFFECTED = {
    "within_fold_reliability.pdf": "ADNI panel bars + title '60/60 folds positive' "
        "(corrected V2 within-environment values differ; see corrected_v2_summaries)",
    "subgroup_robustness.pdf": "ADNI panel AUROCs and n labels (scan counts)",
    "method_comparison.pdf": "ADNI estimator bars (corrected table differs)",
    "selective_prediction.pdf": "ADNI selective-accuracy curve (76->97 lineage)",
    "cross_cohort_forest.pdf": "historical ADNI row",
    "biology_composite.pdf": "panels a-c ADNI halves (misfit-derived stats/maps)",
    "transfer_drift_degradation.pdf": "entire figure (ADNI drift/degradation)",
    "misfit_examples.pdf": "ADNI example cases chosen by contaminated scores",
    "adni_reliability_biology.pdf": "entire figure (ADNI misfit distributions)",
    "oasis3_signature_participant.pdf": "caption-cited ADNI agreement 0.92 "
        "(figure content itself is OASIS-3)",
}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    """Re-hash tracked assets, write the after-manifest, and emit the markdown report."""
    before = json.loads((COR / "provenance/figure_checksums_before.json").read_text())
    after, changed, missing = {}, [], []
    for rel in before:
        p = ROOT / rel
        if not p.exists():
            missing.append(rel)
            continue
        after[rel] = sha(p)
        if after[rel] != before[rel]:
            changed.append(rel)
    (COR / "provenance/figure_checksums_after.json").write_text(
        json.dumps(after, indent=2))

    lines = [
        "# Figure-integrity report (deliverable 10)", "",
        f"Assets tracked: {len(before)} (figure PDFs incl. manuscript/supplement "
        "PDFs + figure-producing scripts).",
        f"CHANGED since the pre-correction snapshot: {len(changed)}"
        + ("" if not changed else " -> " + ", ".join(changed)),
        f"Missing: {missing if missing else 'none'}", "",
        "No figure, panel, caption asset or figure source file was regenerated, "
        "edited, resized, replaced, renumbered or relabeled by this correction.", "",
        "## Figures now contradicted by corrected evidence (UNRESOLVED; author "
        "authorization required before any change)", "",
        "| figure | contradicted content |", "|---|---|",
    ]
    for k, v in AFFECTED.items():
        lines.append(f"| paper/figures/{k} | {v} |")
    lines += ["",
        "Also UNRESOLVED: manuscript.pdf and supplementary.pdf embed the "
        "affected numbers and figures; they are untouched. The correction "
        "cannot be certified complete at the manuscript level while these "
        "remain; report/REPLACEMENT_TEXT.md carries the author-ready fixes.",
    ]
    (COR / "report/figure_integrity.md").write_text("\n".join(lines) + "\n")
    print(f"figures: {len(before)} tracked, {len(changed)} changed, "
          f"{len(missing)} missing")


if __name__ == "__main__":
    main()
