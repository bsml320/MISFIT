"""Regenerate the selective-prediction figure panels with the misfit-guided
frontier. Replaces stale shipped PDFs that showed the earlier certificate-guided
variant rather than the misfit-guided analysis the manuscript describes.

Recomputes per-subject misfit and held-out error on ADNI and OASIS-1 via the
validated collect() harness (N_SEEDS seeds starting at SEED_BASE) and plots
accepted-set accuracy vs coverage, mean +/- 1 std over seeds, with the gain at
50% coverage annotated. Writes adni_subj_selective_prediction.pdf and
oasis_subj_selective_prediction.pdf into paper/figures/.
Run as a plain script with no arguments; uses the GPU selected by the
imported training code.
"""

from misfit.paths import resolve_path as _resolve_path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, _resolve_path("<REPO_ROOT>"))
from misfit.experiments.run_clinical_stratified_media import collect  # noqa
from misfit.experiments.run_honest_estimator_table import (  # noqa
    N_SEEDS, SEED_BASE, load_adni, load_oasis_ds,
)
from pathlib import Path
OUT = Path(_resolve_path("<REPO_ROOT>/misfit/paper/figures"))
GRID = np.arange(0.10, 1.001, 0.02)
plt.rcParams.update({"font.size": 11, "pdf.fonttype": 42})


def frontier(features, targets, site_ids):
    """Compute the per-seed selective-prediction frontier on the GRID coverages.

    For each seed, sorts tested subjects by ascending misfit and records
    accepted-set accuracy at each coverage level.

    Returns:
        Tuple (mean, std) of accuracy arrays over seeds, one entry per GRID point.
    """
    per_seed = []
    for s in range(N_SEEDS):
        d = collect(features, targets, site_ids, SEED_BASE + s)
        t = d["tested"]
        mis = d["misfit"][t]; err = d["err"][t]
        order = np.argsort(mis)  # ascending misfit -> most reliable accepted first
        accs = []
        for cov in GRID:
            n = max(1, int(round(cov * len(mis))))
            accs.append(1.0 - err[order[:n]].mean())
        per_seed.append(accs)
    a = np.array(per_seed)
    return a.mean(0), a.std(0)


def panel(features, targets, site_ids, fname, title):
    """Render one accuracy-vs-coverage panel and save it under OUT/fname."""
    mean, std = frontier(features, targets, site_ids)
    full = mean[-1]
    # Annotated gain = accepted-set accuracy at 50% coverage minus full coverage.
    i50 = int(np.argmin(np.abs(GRID - 0.5)))
    gain = (mean[i50] - full) * 100
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    x = GRID * 100
    ax.plot(x, mean, color="#1f5fa6", lw=2.2, label="Misfit-guided (mean)")
    ax.fill_between(x, mean - std, mean + std, color="#1f5fa6", alpha=0.18, label=r"$\pm1$ std")
    ax.axhline(full, color="gray", ls="--", lw=1.2, label=f"Full coverage ({full:.3f})")
    ax.annotate(f"+{gain:.1f} pp", xy=(50, mean[i50]), xytext=(55, mean[i50] - 0.06),
                fontsize=10, color="#1f5fa6", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#1f5fa6", lw=1.2))
    ax.set_xlabel("Coverage (%)"); ax.set_ylabel("Accepted-set accuracy")
    ax.set_title(title, fontsize=11)
    ax.set_xlim(10, 100)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(OUT / fname, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fname}: full={full:.3f} 50%={mean[i50]:.3f} gain=+{gain:.1f}pp", flush=True)


print("ADNI...", flush=True)
panel(*load_adni(), "adni_subj_selective_prediction.pdf", "Selective prediction: ADNI (protocol shift)")
print("OASIS...", flush=True)
panel(*load_oasis_ds(), "oasis_subj_selective_prediction.pdf", "Selective prediction: OASIS-1 (age shift)")
