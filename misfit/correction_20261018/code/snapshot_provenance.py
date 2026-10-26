"""Provenance snapshot for the 2026-10-18 correction package.

Records, WITHOUT modifying anything:
  - sha256 of producing code, result artifacts, data inputs, manuscript files,
  - sha256 of every figure asset (PDF) and figure-producing script,
  - git binding: repo HEADs + tracked/dirty status of each producing file,
  - library versions and actual numeric seeds used by the historical runs.
Output: provenance/provenance_snapshot.json, provenance/figure_checksums_before.json
"""

from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(_resolve_path("<REPO_ROOT>"))
MIA = ROOT / "misfit"
COR = MIA / "correction_20261018"

PRODUCING_CODE = [
    "misfit/data/adni_loader.py",
    "misfit/training/two_stage_trainer.py",
    "misfit/experiments/run_honest_estimator_table.py",
    "misfit/experiments/run_ml_reliability_media.py",
    "misfit/experiments/run_training_dynamics.py",
    "misfit/experiments/run_paired_tests.py",
    "misfit/experiments/run_foldwise_pca_sensitivity.py",
    "misfit/experiments/run_clinical_stratified_media.py",
    "misfit/experiments/run_class_env_strata_s16.py",
    "misfit/experiments/run_bootstrap_balanced.py",
    "misfit/experiments/run_perseed_dump_two_phases.py",
    "misfit/experiments/dump_per_subject_misfit_iop.py",
    "misfit/data/mri_multisite.py",
]
RESULT_ARTIFACTS = [
    "misfit/results/honest_estimator_table.json",
    "misfit/results/ml_reliability.json",
    "misfit/results/paired_tests.json",
    "misfit/results/foldwise_pca_sensitivity.json",
    "misfit/results/clinical_stratified.json",
    "misfit/results/class_env_strata_s16.json",
    "misfit/results/bootstrap_balanced.json",
    "misfit/results/perseed_misfit_two_phases.npz",
    "misfit/results/per_subject_misfit_iop.npz",
    "misfit/results/per_subject_misfit.npz",
    "misfit/legacy_inputs/external_robustness/honest_estimator_table.json",
    "misfit/legacy_inputs/external_robustness/ml_reliability.json",
    "data/adni/smri/labels/combined.csv",
]
MANUSCRIPT = [
    "misfit/paper/manuscript.tex", "misfit/paper/manuscript.pdf",
    "misfit/paper/supplementary.tex", "misfit/paper/supplementary.pdf",
]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True).stdout.strip()


def file_record(rel: str) -> dict:
    """Hash one repo-relative file and record its git tracking/dirty state.

    Paths under misfit/ resolve against the nested repo; everything else
    against the root repo.
    """
    p = ROOT / rel
    if not p.exists():
        return {"path": rel, "exists": False}
    repo = MIA if rel.startswith("misfit/") else ROOT
    rrel = rel[len("misfit/"):] if rel.startswith("misfit/") else rel
    tracked = git(repo, "ls-files", "--", rrel) != ""
    dirty = git(repo, "status", "--porcelain", "--", rrel) != "" if tracked else None
    last_commit = git(repo, "log", "-1", "--format=%H %ci", "--", rrel) if tracked else None
    return {"path": rel, "exists": True, "sha256": sha(p),
            "size": p.stat().st_size,
            "mtime": p.stat().st_mtime,
            "git_tracked": tracked, "git_dirty_vs_head": dirty,
            "git_last_commit": last_commit}


def main() -> None:
    """Assemble the provenance snapshot and the pre-correction figure checksum manifest."""
    import numpy, pandas, scipy, sklearn, torch
    snap = {
        "generated": "2026-10-18",
        "purpose": "pre-correction provenance freeze (nothing modified)",
        "repos": {
            "fire_mechanist_head": git(ROOT, "log", "-1", "--format=%H %ci %s"),
            "misfit_head": git(MIA, "log", "-1", "--format=%H %ci %s"),
        },
        "versions": {
            "python": sys.version.split()[0], "torch": torch.__version__,
            "numpy": numpy.__version__, "pandas": pandas.__version__,
            "scipy": scipy.__version__, "sklearn": sklearn.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "historical_seeds": {
            "honest_estimator_table / ml_reliability / paired_tests / "
            "training_dynamics": "torch.manual_seed + seed_everything, seeds 42..46 "
                                 "(SEED_BASE=42, N_SEEDS=5)",
            "brain_mask_sampling": "np.random.RandomState(42), 50 volumes",
            "paired_tests_bootstrap": "np.random.RandomState(0), B=2000",
            "ml_reliability_block_a": "np.random.RandomState(0)",
        },
        "producing_code": [file_record(r) for r in PRODUCING_CODE],
        "result_artifacts": [file_record(r) for r in RESULT_ARTIFACTS],
        "manuscript": [file_record(r) for r in MANUSCRIPT],
    }
    (COR / "provenance/provenance_snapshot.json").write_text(json.dumps(snap, indent=2))

    figures = {}
    for d in [MIA / "paper/figures", MIA / "paper"]:
        for p in sorted(d.glob("*.pdf")):
            figures[str(p.relative_to(ROOT))] = sha(p)
    for p in sorted((MIA / "experiments").glob("*fig*.py")):
        figures[str(p.relative_to(ROOT))] = sha(p)
    (COR / "provenance/figure_checksums_before.json").write_text(
        json.dumps(figures, indent=2))
    print(f"code files: {len(snap['producing_code'])}, "
          f"artifacts: {len(snap['result_artifacts'])}, figures+src: {len(figures)}")
    missing = [r["path"] for r in snap["producing_code"] + snap["result_artifacts"]
               + snap["manuscript"] if not r.get("exists")]
    dirty = [r["path"] for r in snap["producing_code"] if r.get("git_dirty_vs_head")]
    untracked = [r["path"] for r in snap["producing_code"] if not r.get("git_tracked")]
    print("missing:", missing)
    print("producing code dirty vs HEAD:", dirty)
    print("producing code untracked:", untracked)


if __name__ == "__main__":
    main()
