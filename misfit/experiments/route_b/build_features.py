"""Assemble the route-B smwc1 outputs into the directory layout the MedIA ADNI loader
expects, so the ENTIRE existing pipeline refits on the DARTEL/3 mm features with no code
change -- just point ADNI_SMRI_DATA_ROOT at the assembled root.

Run after run_route_b_adni.py (pilot or --full):
  python build_features.py
  export ADNI_SMRI_DATA_ROOT=<REPO_ROOT>/misfit/results/route_b/route_b_smri
  # now re-run any MedIA experiment; load_adni_smri() reads the route-B features and the
  # mask + PCA-50 + z-norm are rebuilt on them automatically. Then re-validate AUROC /
  # selective-prediction and compare to the original-feature numbers.
"""
from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

import sys
from pathlib import Path

sys.path.insert(0, _resolve_path("<REPO_ROOT>"))
from misfit.data.adni_loader import DATA_ROOT as ORIG_ROOT  # noqa: E402

WORK = Path(_resolve_path("<REPO_ROOT>/misfit/experiments/route_b/_adni_work"))
OUT = Path(_resolve_path("<REPO_ROOT>/misfit/results/route_b/route_b_smri"))


def main():
    """Symlink every finished smwc1 into <OUT>/nii/<protocol>/<subject>/ and reuse labels."""
    nii = OUT / "nii"
    nii.mkdir(parents=True, exist_ok=True)
    n = 0
    for wd in sorted(WORK.glob("*__*")):
        smwc1 = wd / "smwc1pT1.nii"
        if not smwc1.exists():
            continue
        proto, subj = wd.name.split("__", 1)
        d = nii / proto / subj
        d.mkdir(parents=True, exist_ok=True)
        link = d / "smwc1pT1.nii"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(smwc1.resolve())    # loader only reads smwc1pT1.nii
        n += 1
    lab = OUT / "labels"
    src = Path(ORIG_ROOT) / "labels"
    if src.exists() and not lab.exists():
        lab.symlink_to(src.resolve())       # reuse the same CN/impaired labels
    print(f"assembled {n} route-B subjects at {OUT}")
    print(f"refit:  export ADNI_SMRI_DATA_ROOT={OUT}")
    print("        then re-run the MedIA ADNI experiments (features are rebuilt on the new smwc1)")
    if n < 700:
        print(f"NOTE: only {n} subjects assembled -- this is a PILOT. Run "
              "`run_route_b_adni.py --full` first for the production refit.")


if __name__ == "__main__":
    main()
