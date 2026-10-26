"""Portable reconstruction of the OASIS-1 anatomical association map (3D NIfTI).

Self-contained: depends only on numpy, nibabel and nilearn (for the public Schaefer
atlas fetch). No repository imports, no cohort data, no participant-level inputs.

The OASIS-1 map is fully determined by two public pieces:
  1. oasis1_roi_coefficients.json - the 100 raw per-parcel Spearman correlations
     between raw Schaefer-100 gray-matter density and the seed-averaged per-subject
     misfit (field "roi_signature"; all values negative), as distributed with the
     revision kit / response bundle.
  2. The Schaefer-2018 100-parcel (7-network) 1 mm MNI152 atlas, fetched by nilearn
     from its public source.

Reconstruction = assign each parcel's coefficient to that parcel's voxels on the
atlas grid (constant within parcel, exact zero outside the atlas), with the atlas's
own affine and a float32 volume. This is byte-for-byte the construction used by the
producing script (run_image_signature_media.py) for results/oasis_misfit_signature.nii.gz.

Usage:
  python reconstruct_oasis1_map.py coefficients.json out_map.nii.gz [reference.nii.gz]

With a reference (the shipped oasis_misfit_signature.nii.gz), the script verifies the
reconstruction voxel-for-voxel (atol 1e-6 on the float32 arrays) and reports the
result; it exits nonzero on any mismatch. The display transform used by the figures
(z-score across the 100 parcel values, |z| > 0.8 shown, symmetric limit 2.6) is a
documented function of these raw values and is NOT baked into the saved map.
"""
import json
import sys

import nibabel as nib
import numpy as np
from nilearn.datasets import fetch_atlas_schaefer_2018


def main() -> int:
    if len(sys.argv) not in (3, 4):
        print(__doc__)
        return 2
    coef_path, out_path = sys.argv[1], sys.argv[2]
    ref_path = sys.argv[3] if len(sys.argv) == 4 else None

    roi = np.array(json.load(open(coef_path))["roi_signature"], dtype=float)
    assert roi.shape == (100,), f"expected 100 coefficients, got {roi.shape}"

    atlas = fetch_atlas_schaefer_2018(n_rois=100)
    lab_img = nib.load(atlas.maps)
    labvol = np.asarray(lab_img.dataobj).astype(int)

    vol = np.zeros(labvol.shape, dtype=np.float32)
    for j in range(1, 101):
        vol[labvol == j] = roi[j - 1]
    img = nib.Nifti1Image(vol, lab_img.affine)
    nib.save(img, out_path)
    print(f"wrote {out_path}: shape {vol.shape}, "
          f"{int((labvol > 0).sum())} in-atlas voxels, "
          f"coefficient range [{roi.min():.4f}, {roi.max():.4f}]")

    if ref_path:
        ref = np.asarray(nib.load(ref_path).dataobj).astype(np.float32)
        if ref.shape != vol.shape:
            print(f"VERIFICATION FAILED: shape {ref.shape} != {vol.shape}")
            return 1
        if not np.allclose(ref, vol, atol=1e-6):
            n = int((~np.isclose(ref, vol, atol=1e-6)).sum())
            print(f"VERIFICATION FAILED: {n} voxels differ beyond 1e-6")
            return 1
        print("VERIFICATION PASSED: reconstruction matches the reference "
              "voxel-for-voxel (atol 1e-6)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
