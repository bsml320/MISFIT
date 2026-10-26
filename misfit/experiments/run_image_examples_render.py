"""Render-only example-subject panel from cached per-subject misfit.

The structural misfit is a near-zero training cross-entropy whose RANKING carries
the reliability signal, so subjects are annotated by reliability percentile (rank),
not by the raw loss value. The panel shows the global extremes: the most-reliable
(lowest-misfit, correctly classified) and least-reliable (highest-misfit,
misclassified) subject per cohort. Illustrative, not a quantitative claim.

Input: results/per_subject_misfit.npz cached by run_image_examples_media.py,
plus the volume paths rebuilt via that module's adni_paths()/oasis_paths().
Output: paper/figures/misfit_examples.pdf. Run as a plain script, no arguments.
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
from misfit.experiments.run_image_examples_media import adni_paths, oasis_paths  # noqa: E402

RES = Path(_resolve_path("<REPO_ROOT>/misfit/results"))
FIG = Path(_resolve_path("<REPO_ROOT>/misfit/paper/figures"))
plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})

ADNI_CLS = {0: "CN", 1: "impaired"}
OASIS_CLS = {0: "healthy", 1: "demented"}


def pct(mis, i):
    return 100.0 * (mis < mis[i]).mean()


def pick_global(mis, err):
    """Indices of the lowest-misfit correct subject and highest-misfit wrong subject."""
    rel = np.where(err == 0)[0]
    unr = np.where(err == 1)[0]
    return int(rel[mis[rel].argmin()]), int(unr[mis[unr].argmax()])


def axial(path, k):
    return np.rot90(nib.load(path).get_fdata(dtype=np.float32)[:, :, k])


def render(ax, path, k, title, vmax):
    ax.imshow(axial(path, k), cmap="gray", vmin=0, vmax=vmax)
    ax.set_title(title, fontsize=7.5)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    """Load cached misfit, pick per-cohort extremes, render and save the panel."""
    d = np.load(RES / "per_subject_misfit.npz")
    ma, ea, la = d["adni_misfit"], d["adni_err"], d["adni_label"]
    mo, eo, lo = d["oasis_misfit"], d["oasis_err"], d["oasis_label"]
    pa, _ = adni_paths()
    po, _ = oasis_paths()

    a_rel, a_unr = pick_global(ma, ea)
    o_rel, o_unr = pick_global(mo, eo)

    def cap(cohort, idx, lab, cls, reliable):
        t = cls[int(lab[idx])]
        pred = cls[int(lab[idx])] if reliable else cls[1 - int(lab[idx])]
        tag = "most reliable" if reliable else "least reliable"
        verdict = "correct" if reliable else "wrong"
        return f"{cohort}: {tag}\ntrue {t} $\\to$ pred {pred}\n({verdict})"

    ka = 60
    ko = nib.load(po[o_rel]).shape[2] // 2
    va = float(np.percentile(nib.load(pa[a_rel]).get_fdata(dtype=np.float32), 99))
    vo = float(np.percentile(nib.load(po[o_rel]).get_fdata(dtype=np.float32), 99))

    fig, ax = plt.subplots(1, 4, figsize=(10.4, 3.0))
    render(ax[0], pa[a_rel], ka, cap("ADNI", a_rel, la, ADNI_CLS, True), va)
    render(ax[1], pa[a_unr], ka, cap("ADNI", a_unr, la, ADNI_CLS, False), va)
    render(ax[2], po[o_rel], ko, cap("OASIS", o_rel, lo, OASIS_CLS, True), vo)
    render(ax[3], po[o_unr], ko, cap("OASIS", o_unr, lo, OASIS_CLS, False), vo)
    fig.suptitle("Gray-matter maps at the reliability extremes (most vs least reliable case per cohort)",
                 fontsize=9.5, y=1.06)
    fig.subplots_adjust(wspace=0.06, top=0.80)
    fig.savefig(FIG / "misfit_examples.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"ADNI rel={a_rel}(pct {100-pct(ma,a_rel):.1f}) unr={a_unr}(pct {100-pct(ma,a_unr):.1f})", flush=True)
    print(f"OASIS rel={o_rel}(pct {100-pct(mo,o_rel):.1f}) unr={o_unr}(pct {100-pct(mo,o_unr):.1f})", flush=True)
    print("wrote misfit_examples.pdf", flush=True)


if __name__ == "__main__":
    main()
