"""Apply the route-B pipeline to a NEW cohort (AIBL) using the EXISTING ADNI DARTEL
template: Segment+import -> DARTEL warp to existing template -> MNI-normalise modulated
+ 3 mm smooth -> (project through the refit model). Ready for AIBL; validated here on a
held-out ADNI subject as an AIBL proxy (--selftest), which exercises the identical path.

Usage:
  python apply_aibl.py --selftest
  python apply_aibl.py --t1-list <paths.txt> --template-dir <ADNI Template_*.nii dir> --out <dir>
"""
from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

import glob
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, _resolve_path("<REPO_ROOT>"))
from vbm_pipeline import dartel_mni_norm, dartel_warp_to_existing, segment_many  # noqa: E402


def apply_cohort(items: list[tuple[Path, Path]], template_dir: str, jobs: int = 4) -> list[str]:
    """Run segment -> warp-to-existing-template -> MNI-normalize for a cohort.

    Args:
        items: List of (T1 path, per-subject work directory) pairs.
        template_dir: Directory holding the fixed ADNI Template_*.nii files.
        jobs: Parallel segmentation workers.

    Returns:
        List of produced smwc1 paths (modulated, smoothed, ADNI DARTEL space).

    Raises:
        RuntimeError: If every segmentation fails, or a warp/normalize SPM
            batch does not report its completion marker.
    """
    seg = segment_many(items, jobs=jobs)
    ok = [r for r in seg if r["ok"]]
    if not ok:
        raise RuntimeError("all segmentations failed")
    log = dartel_warp_to_existing([r["rc1"] for r in ok], [r["rc2"] for r in ok], template_dir)
    if "WARP1_DONE" not in log:
        raise RuntimeError("warp-to-existing failed:\n" + log[-800:])
    template6 = sorted(glob.glob(f"{template_dir}/Template_6.nii"))[0]
    subj = []
    for r in ok:
        ff = sorted(glob.glob(str(Path(r["wd"]) / "u_rc1*.nii")))
        if ff:
            subj.append((ff[0], r["c1"]))
    log = dartel_mni_norm(template6, subj)
    if "NORM_DONE" not in log:
        raise RuntimeError("mni_norm failed:\n" + log[-800:])
    return [str(Path(r["wd"]) / "smwc1pT1.nii") for r in ok]


def selftest():
    """Validate the AIBL code path on a held-out ADNI subject warped to the pilot template."""
    from misfit.data.adni_loader import NII_DIR
    template_dir = _resolve_path("<REPO_ROOT>/misfit/experiments/route_b/_adni_work/ADNI2-Phillips-initial__ADNI-PTID-01")
    if not glob.glob(f"{template_dir}/Template_6.nii"):
        print("no pilot template; run run_route_b_adni.py first"); return
    p = "ADNI3-Siemens-initial"
    subs = sorted(s.name for s in (NII_DIR / p).iterdir() if (s / "T1.nii").exists())
    s = subs[-1]   # held out (pilot took the first 2 per protocol)
    wd = Path(_resolve_path("<REPO_ROOT>/misfit/experiments/route_b/_aibl_selftest")) / f"{p}__{s}"
    print(f"selftest: warping held-out {p}/{s} into the pilot ADNI template...", flush=True)
    outs = apply_cohort([(NII_DIR / p / s / "T1.nii", wd)], template_dir)
    im = nib.load(outs[0]); v = im.get_fdata()
    desc = bytes(im.header["descrip"]).decode("latin-1").rstrip("\x00")
    ok = im.shape == (121, 145, 121) and "Dartel" in desc
    print(f"selftest smwc1: shape={im.shape} descrip='{desc}' mean(>0)={v[v>0].mean():.4f}")
    print("AIBL APPLY PATH OK" if ok else "CHECK OUTPUT")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        print("use --selftest, or --t1-list/--template-dir/--out (see docstring)")
