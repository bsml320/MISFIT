"""Example-subject panel for the MIA paper: actual VBM gray-matter scans of
RELIABLE (low misfit, consistently correct) versus UNRELIABLE (high misfit,
consistently misclassified) subjects, for both cohorts.

This is the concrete image-level companion to the anatomical-signature maps: the
signature says lower gray matter associates with higher misfit; these examples
show that the subjects the model is least reliable on are visibly more atrophic.
Subjects are chosen as the misfit extremes within the consistently-correct and
consistently-wrong groups (illustrative, not cherry-picked across the ranking).

GPU 2. Writes paper/figures/misfit_examples.pdf and caches per-subject arrays.
"""
from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

import os
import sys
from pathlib import Path

# CUDA visibility is selected by the caller.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, _resolve_path("<REPO_ROOT>"))
from misfit.experiments.run_clinical_stratified_media import collect  # noqa: E402
from misfit.experiments.run_honest_estimator_table import (  # noqa: E402
    N_SEEDS,
    SEED_BASE,
    load_adni,
    load_oasis_ds,
)

OUT_FIG = Path(_resolve_path("<REPO_ROOT>/misfit/paper/figures"))
OUT_RES = Path(_resolve_path("<REPO_ROOT>/misfit/results"))
plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})


def misfit_err(features, targets, site_ids):
    """Seed-averaged per-subject misfit and misclassification fraction."""
    mis, err = [], []
    for s in range(N_SEEDS):
        d = collect(features, targets, site_ids, SEED_BASE + s)
        mis.append(d["misfit"])
        err.append(d["err"].astype(float))
    return np.array(mis).mean(0), np.array(err).mean(0)


def adni_paths():
    """Rebuild ADNI smwc1 volume paths and labels in loader order.

    Applies the same protocol, label and minimum-site-size filters as the
    Phase 1A ADNI experiment, so indices align with load_adni() features.
    """
    import pandas as pd
    from misfit.data.adni_loader import (
        IMPAIRED_GROUPS,
        LABELS_CSV,
        MIN_SUBJECTS_PER_SITE,
        NII_DIR,
        PROTOCOLS,
    )

    df = pd.read_csv(LABELS_CSV)
    lut = {}
    for _, r in df.iterrows():
        k = (str(r["dataset_id"]).strip(), str(r["file_name"]).strip())
        lut.setdefault(k, str(r["Group"]).strip())
    paths, labs = [], []
    for protocol in PROTOCOLS:
        pdir = NII_DIR / protocol
        if not pdir.exists():
            continue
        pp, pl = [], []
        for sdir in sorted(pdir.iterdir()):
            f = sdir / "smwc1pT1.nii"
            if not f.exists():
                continue
            k = (protocol, sdir.name)
            if k not in lut or lut[k] not in {"CN"} | IMPAIRED_GROUPS:
                continue
            pp.append(str(f))
            pl.append(0 if lut[k] == "CN" else 1)
        if len(pp) < MIN_SUBJECTS_PER_SITE:
            continue
        paths.extend(pp)
        labs.extend(pl)
    return np.array(paths), np.array(labs)


def oasis_paths():
    """Rebuild OASIS-1 gray-matter map paths and binary CDR labels.

    Mirrors the age-quartile environment filter of load_oasis_ds() so indices
    align with the loader's feature order.
    """
    from nilearn.datasets import fetch_oasis_vbm

    o = fetch_oasis_vbm()
    gm = np.asarray(o.gray_matter_maps)
    cdr = o.ext_vars["cdr"].values.astype(float)
    age = o.ext_vars["age"].values.astype(float)
    valid = ~np.isnan(cdr)
    gm, age = gm[valid], age[valid]
    q = np.quantile(age, np.linspace(0, 1, 5))
    env = np.digitize(age, q[1:-1])
    counts = {e: int((env == e).sum()) for e in sorted(set(env))}
    keep = np.isin(env, [e for e, c in counts.items() if c >= 10])
    return gm[keep], (cdr[valid][keep] > 0).astype(int)


def pick(mis, err, labs, want_label):
    """Return (reliable_idx, unreliable_idx) within a diagnostic class."""
    m = labs == want_label
    rel_pool = np.where(m & (err == 0))[0]
    unr_pool = np.where(m & (err == 1))[0]
    rel = rel_pool[np.argmin(mis[rel_pool])] if len(rel_pool) else int(np.argmin(np.where(m, mis, np.inf)))
    unr = unr_pool[np.argmax(mis[unr_pool])] if len(unr_pool) else int(np.argmax(np.where(m, mis, -np.inf)))
    return rel, unr


def axial(path, k):
    vol = nib.load(path).get_fdata(dtype=np.float32)
    sl = vol[:, :, k]
    return np.rot90(sl)


def render(ax, path, k, title, vmax):
    ax.imshow(axial(path, k), cmap="gray", vmin=0, vmax=vmax)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    """Compute misfit, pick example subjects per cohort, render the 4-panel figure."""
    print("ADNI misfit/err ...", flush=True)
    fa, ta, sa = load_adni()
    mis_a, err_a = misfit_err(fa, ta, sa)
    pa, la = adni_paths()
    assert len(pa) == len(ta) and np.array_equal(la, ta.numpy().astype(int)), "ADNI align broken"

    print("OASIS misfit/err ...", flush=True)
    fo, to, so = load_oasis_ds()
    mis_o, err_o = misfit_err(fo, to, so)
    po, lo = oasis_paths()
    assert len(po) == len(to), "OASIS align broken"

    np.savez(OUT_RES / "per_subject_misfit.npz",
             adni_misfit=mis_a, adni_err=err_a, adni_label=la,
             oasis_misfit=mis_o, oasis_err=err_o, oasis_label=lo)

    # ADNI: impaired class (1) reliable vs unreliable; OASIS: demented (1)
    a_rel, a_unr = pick(mis_a, err_a, la, 1)
    o_rel, o_unr = pick(mis_o, err_o, lo, 1)
    print(f"ADNI rel idx={a_rel} mis={mis_a[a_rel]:.3f} | unr idx={a_unr} mis={mis_a[a_unr]:.3f}", flush=True)
    print(f"OASIS rel idx={o_rel} mis={mis_o[o_rel]:.3f} | unr idx={o_unr} mis={mis_o[o_unr]:.3f}", flush=True)

    ka = 60  # ADNI axial slice (121,145,121) ~ ventricles + cortex
    # OASIS map dims differ; pick mid axial
    ko = nib.load(po[o_rel]).shape[2] // 2
    vmax_a = float(np.percentile(nib.load(pa[a_rel]).get_fdata(dtype=np.float32), 99))
    vmax_o = float(np.percentile(nib.load(po[o_rel]).get_fdata(dtype=np.float32), 99))

    fig, axes = plt.subplots(1, 4, figsize=(8.0, 2.5))
    render(axes[0], pa[a_rel], ka, f"ADNI reliable\nmisfit={mis_a[a_rel]:.2f}, impaired (correct)", vmax_a)
    render(axes[1], pa[a_unr], ka, f"ADNI unreliable\nmisfit={mis_a[a_unr]:.2f}, impaired (missed)", vmax_a)
    render(axes[2], po[o_rel], ko, f"OASIS reliable\nmisfit={mis_o[o_rel]:.2f}, demented (correct)", vmax_o)
    render(axes[3], po[o_unr], ko, f"OASIS unreliable\nmisfit={mis_o[o_unr]:.2f}, demented (missed)", vmax_o)
    fig.suptitle("Example subjects: gray-matter maps of reliable vs unreliable cases", fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "misfit_examples.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote paper/figures/misfit_examples.pdf", flush=True)


if __name__ == "__main__":
    main()
