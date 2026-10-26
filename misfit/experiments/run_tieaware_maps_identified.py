"""Identified tie-aware rank comparison for BOTH retained anatomical maps.

Supersedes the unidentified provenance/tieaware_rank_check.json (2026-10-04), which did
not name cohort, map, mask or producing script. Here each comparison is bound to the
retained artifact: the recomputed map in the artifact's own rank convention must match
the saved NIfTI voxel-for-voxel before the alternative convention is evaluated.

  (a) ADNI misfit signature  [results/adni_misfit_signature.nii.gz, producing script
      run_image_signature_media.py]: published convention = tie-aware average ranks
      (scipy rankdata); comparison = ordinal double-argsort variant.
  (b) OASIS-3 participant-level signature (main-paper figure)
      [results/oasis3/oasis3_signature_participant.nii.gz, producing script
      run_oasis3_signature_participant.py]: published convention = ordinal double
      argsort; comparison = tie-aware average-rank variant.

Outputs: results/provenance/tieaware_maps_identified.json (+ raw maps npz).
CPU only; misfit vectors from the canonical persisted arrays (bit-verified earlier).
"""
from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

import json
import re
from concurrent.futures import ThreadPoolExecutor

import nibabel as nib
import numpy as np
from scipy.stats import rankdata

RES = _resolve_path("<REPO_ROOT>/misfit/results")
OUT = f"{RES}/provenance"


def zrank_ordinal(X):
    """Column-wise z-scored ordinal ranks (double argsort; ties broken by order)."""
    g = np.argsort(np.argsort(X, axis=0), axis=0).astype(np.float32)
    g -= g.mean(0)
    g /= (g.std(0) + 1e-9)
    return g


def zrank_tieaware(X):
    """Column-wise z-scored tie-aware average ranks (scipy rankdata)."""
    g = rankdata(X, axis=0).astype(np.float32)
    g -= g.mean(0)
    g /= (g.std(0) + 1e-9)
    return g


def corr_map(G, y_rank_z):
    return (G * y_rank_z[:, None]).mean(0)


def tie_census(X, chunk=20000):
    """Fraction of columns with at least one tied value, and mean tied-pair fraction."""
    any_tie = 0
    frac_sum = 0.0
    n = X.shape[1]
    for i in range(0, n, chunk):
        S = np.sort(X[:, i:i + chunk], axis=0)
        eq = (np.diff(S, axis=0) == 0)
        any_tie += int((eq.any(0)).sum())
        frac_sum += float(eq.mean(0).sum())
    return any_tie / n, frac_sum / n


def compare(name, Xm, mis, saved_nii, mask, published):
    """Bind one map to its retained artifact, then compare rank conventions.

    Recomputes the map in the published convention and measures its max abs
    difference to the saved NIfTI (binds_to_saved_map_max_abs_diff), then
    evaluates the alternative convention and a tie census over voxel columns.
    Raw ordinal and tie-aware maps are saved to an npz alongside the mask.

    Returns:
        Dict of binding, agreement and tie statistics.
    """
    yr = rankdata(mis).astype(np.float32)
    yz = (yr - yr.mean()) / yr.std()
    m_ord = corr_map(zrank_ordinal(Xm), yz)
    m_tie = corr_map(zrank_tieaware(Xm), yz)
    repro = m_ord if published == "ordinal" else m_tie
    saved = nib.load(saved_nii).get_fdata(dtype=np.float32).ravel()[mask]
    bind = float(np.abs(repro - saved).max())
    col_tie_frac, pair_tie_frac = tie_census(Xm)
    r = {
        "published_rank_convention": published,
        "binds_to_saved_map_max_abs_diff": bind,
        "corr_ordinal_vs_tieaware": float(np.corrcoef(m_ord, m_tie)[0, 1]),
        "max_abs_map_difference": float(np.abs(m_ord - m_tie).max()),
        "frac_negative_ordinal": float((m_ord < 0).mean()),
        "frac_negative_tieaware": float((m_tie < 0).mean()),
        "voxel_columns_with_any_tie_frac": col_tie_frac,
        "mean_tied_adjacent_pair_frac": pair_tie_frac,
        "misfit_n_tied_values": int(len(mis) - len(np.unique(mis))),
        "n_subjects": int(Xm.shape[0]), "n_voxels": int(Xm.shape[1]),
    }
    np.savez(f"{OUT}/tieaware_maps_{name}.npz", ordinal=m_ord, tieaware=m_tie,
             mask=mask)
    print(name, json.dumps(r, indent=1), flush=True)
    return r


res = {"note": "each comparison is bound to the retained artifact by exact "
               "recomputation in its own convention (binds_to_saved_map_max_abs_diff); "
               "supersedes the unidentified tieaware_rank_check.json of 2026-10-04"}

# ---------- (b) OASIS-3 participant-level map (ordinal published) ----------
WORK = _resolve_path("<REPO_ROOT>/data/oasis3/route_b_work")
d = np.load(_resolve_path("<REPO_ROOT>/data/oasis3/oasis3_features.npz"), allow_pickle=True)
subj = d["subjects"].astype(str)
sess = d["sessions"].astype(str)
day = np.array([int(re.search(r"_d(\d+)$", s).group(1)) for s in sess])
# one earliest session per participant, matching the producing script's cohort
keep = np.array(sorted(np.where(subj == u)[0][np.argmin(day[subj == u])]
                       for u in np.unique(subj)))
sess = sess[keep]
mis = np.load(f"{RES}/oasis3/oasis3_misfit_leakagefree_5seed.npy")
print(f"[oasis3] loading {len(sess)} volumes...", flush=True)
with ThreadPoolExecutor(max_workers=16) as ex:
    X = np.stack(list(ex.map(
        lambda s: nib.load(f"{WORK}/{s}/smwc1pT1.nii").get_fdata(dtype=np.float32).ravel(),
        sess)))
mask = X.mean(0) > 0.1
Xm = X[:, mask]
del X
res["oasis3_participant_signature"] = dict(
    cohort="OASIS-3 leakage-free, one earliest session per participant",
    map="results/oasis3/oasis3_signature_participant.nii.gz (main-paper figure)",
    mask="mean GM > 0.1 over the 1360 route_b_work smwc1 volumes",
    producing_script="experiments/run_oasis3_signature_participant.py",
    misfit="canonical collect() 5-seed mean (oasis3_misfit_leakagefree_5seed.npy, "
           "bit-verified against the 2026-10-05 rerun)",
    **compare("oasis3_participant", Xm, mis,
              f"{RES}/oasis3/oasis3_signature_participant.nii.gz", mask, "ordinal"))
del Xm

# ---------- (a) ADNI misfit signature (tie-aware published) ----------
import sys
sys.path.insert(0, _resolve_path("<REPO_ROOT>"))
sys.path.insert(0, _resolve_path("<REPO_ROOT>/misfit/experiments"))
from run_image_signature_media import adni_raw_voxels  # noqa: E402

print("[adni] loading raw voxels...", flush=True)
Vx, labs, amask, _ = adni_raw_voxels()
pm = np.load(f"{RES}/perseed_misfit_two_phases.npz")
mis_a = pm["adni_mis_A"].mean(0)
assert len(mis_a) == Vx.shape[0], (len(mis_a), Vx.shape)
res["adni_misfit_signature"] = dict(
    cohort="ADNI sMRI, 731 subjects, 8 protocols",
    map="results/adni_misfit_signature.nii.gz",
    mask="_create_brain_mask(NII_DIR, PROTOCOLS) from run_adni_smri (435026 voxels)",
    producing_script="experiments/run_image_signature_media.py",
    misfit="canonical 5-seed mean (perseed_misfit_two_phases.npz adni_mis_A)",
    **compare("adni_signature", Vx, mis_a,
              f"{RES}/adni_misfit_signature.nii.gz", amask.ravel(), "tieaware"))

json.dump(res, open(f"{OUT}/tieaware_maps_identified.json", "w"), indent=2)
print("wrote tieaware_maps_identified.json", flush=True)
