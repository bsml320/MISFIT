"""Route-B orchestrator: reprocess ADNI through the recovered SPM25 DARTEL recipe,
then extract the model features. Validates on a pilot subset (default) or the full
cohort (--full). The full DARTEL template step is a multi-hour joint optimisation.

Usage:
  python run_route_b_adni.py            # pilot: ~2 subjects/protocol
  python run_route_b_adni.py 12         # pilot of 12 subjects
  python run_route_b_adni.py --full     # full ADNI cohort (long; build the production template)

Stages: (1) Segment+import  (2) DARTEL create Template  (3) MNI-normalise modulated + 3mm smooth
Output: results/route_b/route_b_pilot.json (+ per-subject smwc1 in _adni_work/).
"""
from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

import glob
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, _resolve_path("<REPO_ROOT>"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from misfit.data.adni_loader import (  # noqa: E402
    NII_DIR, PROTOCOLS, _create_brain_mask,
)
from vbm_pipeline import dartel_create_template, dartel_mni_norm, segment_many  # noqa: E402

WORK = Path(_resolve_path("<REPO_ROOT>/misfit/experiments/route_b/_adni_work"))
RES = Path(_resolve_path("<REPO_ROOT>/misfit/results/route_b"))
ORIG_DESCRIP = "Smoothed (3x3x3) Dartel normed"


def select_subjects(full: bool, n_total: int):
    """List (protocol, subject) pairs: the whole cohort, or the pilot subset.

    Pilot mode takes the first 2 subjects per protocol, capped at n_total.
    """
    subs = []
    per = 10 ** 6 if full else 2
    for p in PROTOCOLS:
        pdir = NII_DIR / p
        if not pdir.exists():
            continue
        ss = sorted(s.name for s in pdir.iterdir() if (s / "T1.nii").exists())
        subs += [(p, s) for s in ss[:per]]
    return subs if full else subs[:n_total]


def find_one(patterns):
    """First file matching any of the glob patterns, else None."""
    for pat in patterns:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return hits[0]
    return None


def main():
    """Run the three SPM stages, then verify geometry/header and extract features.

    Verification checks each produced smwc1 against the model grid and the
    original recipe's NIfTI descrip string, and summarizes the PCA spectrum of
    the masked features. Writes results/route_b/route_b_pilot.json.
    """
    full = "--full" in sys.argv
    nums = [a for a in sys.argv[1:] if a.isdigit()]
    n_total = int(nums[0]) if nums else 6
    RES.mkdir(parents=True, exist_ok=True)
    subs = select_subjects(full, n_total)
    print(f"route B ({'FULL' if full else 'pilot'}): {len(subs)} subjects", flush=True)

    # ---- Stage 1: Segment + DARTEL import (parallel) ----
    items = [(NII_DIR / p / s / "T1.nii", WORK / f"{p}__{s}") for p, s in subs]
    seg = segment_many(items, jobs=12 if full else 4)
    ok = [r for r in seg if r["ok"]]
    print(f"[stage 1] segmented {len(ok)}/{len(seg)}", flush=True)
    for r in seg:
        if not r["ok"]:
            print("  SEG FAIL", r["wd"], "|", r["err"][:150])
    if len(ok) < 3:
        print("too few segmented; aborting"); return

    # ---- Stage 2: DARTEL create Template (joint) ----
    print(f"[stage 2] DARTEL template over {len(ok)} subjects (this is the slow step)...", flush=True)
    log = dartel_create_template([r["rc1"] for r in ok], [r["rc2"] for r in ok], str(RES / "lists"))
    if "DARTEL_DONE" not in log:
        print("DARTEL FAILED:\n", log[-1000:]); return
    template6 = find_one([str(WORK / "**" / "Template_6.nii"), str(Path.cwd() / "Template_6.nii")])
    if not template6:
        print("Template_6.nii not found after DARTEL"); return
    print(f"[stage 2] template: {template6}", flush=True)

    # ---- Stage 3: MNI-normalise modulated + 3mm smooth ----
    subj = []
    for r in ok:
        wd = Path(r["wd"])
        ff = find_one([str(wd / "u_rc1*.nii")])
        if ff:
            subj.append((ff, r["c1"]))
    print(f"[stage 3] mni_norm over {len(subj)} subjects...", flush=True)
    log = dartel_mni_norm(template6, subj)
    if "NORM_DONE" not in log:
        print("NORM FAILED:\n", log[-1000:]); return

    # ---- Verify + extract features ----
    mask = _create_brain_mask(NII_DIR, PROTOCOLS)   # model geometry (121x145x121, 1.5mm)
    rows, feats = [], []
    for r in ok:
        f = Path(r["wd"]) / "smwc1pT1.nii"
        if not f.exists():
            continue
        im = nib.load(f)
        desc = bytes(im.header["descrip"]).decode("latin-1").rstrip("\x00")
        v = im.get_fdata()
        rows.append({"subject": Path(r["wd"]).name, "descrip": desc,
                     "shape": list(im.shape), "mean_gt0": float(v[v > 0].mean())})
        if im.shape == mask.shape:
            feats.append(v[mask])
    evr = []
    if len(feats) >= 3:
        X = np.asarray(feats, dtype=np.float64)
        S = np.linalg.svd(X - X.mean(0), compute_uv=False)
        evr = [float(x) for x in (S[:10] ** 2 / (S ** 2).sum())]

    summary = {
        "mode": "full" if full else "pilot", "n_smwc1": len(rows),
        "template": template6, "mask_voxels": int(mask.sum()),
        "geometry_ok": all(tuple(r["shape"]) == mask.shape for r in rows),
        "descrip_matches_original": all(r["descrip"] == ORIG_DESCRIP for r in rows),
        "mean_gt0_mean": float(np.mean([r["mean_gt0"] for r in rows])) if rows else None,
        "pca_top10_evr": evr, "rows": rows,
    }
    (RES / "route_b_pilot.json").write_text(json.dumps(summary, indent=2))
    print("\n=== ROUTE B PILOT SUMMARY ===")
    print(f"smwc1 produced           : {len(rows)}")
    print(f"geometry == model grid   : {summary['geometry_ok']}  (mask {mask.shape})")
    print(f"descrip == original recipe: {summary['descrip_matches_original']}  ('{ORIG_DESCRIP}')")
    print(f"mean(>0)                 : {summary['mean_gt0_mean']:.4f}")
    if evr:
        print(f"top-10 PCA EVR           : {[round(x, 3) for x in evr]}")
    print(f"wrote {RES}/route_b_pilot.json")
    print("\nNext: `--full` builds the production template over all ADNI; then refit_model.py "
          "rebuilds mask+PCA-50 on the new smwc1 and retrains; then apply_aibl.py warps AIBL to "
          "this template and scores.")


if __name__ == "__main__":
    main()
