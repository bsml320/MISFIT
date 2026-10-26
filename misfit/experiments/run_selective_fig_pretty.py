"""Polished selective-prediction frontier (single combined figure), restyled to
match the other result figures (seaborn whitegrid, dataset palette, shaded
+/-1 std band, accent gain annotation). Recomputes the per-seed misfit-guided
coverage--accuracy frontier via the validated collector. GPU 2.

Writes paper/figures/selective_prediction.pdf (replaces the two separate panels).
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
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

sys.path.insert(0, _resolve_path("<REPO_ROOT>"))
from misfit.experiments.run_clinical_stratified_media import collect  # noqa: E402
from misfit.experiments.run_honest_estimator_table import (  # noqa: E402
    N_SEEDS,
    SEED_BASE,
    load_adni,
    load_oasis_ds,
)

OUT = Path(_resolve_path("<REPO_ROOT>/misfit/paper/figures"))
sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
plt.rcParams.update({
    "pdf.fonttype": 42, "savefig.dpi": 300,
    "axes.edgecolor": "#43474d", "axes.labelcolor": "#22252a",
    "text.color": "#22252a", "xtick.color": "#43474d", "ytick.color": "#43474d",
    "grid.color": "#dfe3e8", "grid.linewidth": 0.7,
})
ADNI_C, OASIS_C, ACCENT, GRAY = "#2b6cb0", "#2f8f6b", "#c0392b", "#9aa0a6"
GRID = np.arange(0.10, 1.001, 0.02)


def frontier(features, targets, site_ids):
    """Per-seed coverage-accuracy frontier from the misfit ordering.

    For each seed, tested subjects are ranked by misfit (ascending) and
    accepted-set accuracy is evaluated at each coverage level in GRID.

    Returns:
        (mean, std) arrays over seeds, one entry per coverage level.
    """
    per_seed = []
    for s in range(N_SEEDS):
        d = collect(features, targets, site_ids, SEED_BASE + s)
        t = d["tested"]
        mis = d["misfit"][t]
        err = d["err"][t]
        order = np.argsort(mis)
        accs = [1.0 - err[order[:max(1, int(round(c * len(mis))))]].mean() for c in GRID]
        per_seed.append(accs)
    a = np.array(per_seed)
    return a.mean(0), a.std(0)


def panel(ax, features, targets, site_ids, color, title):
    """Draw one dataset's frontier: +/-1 std band, mean curve, full-coverage
    reference line, and the accent annotation at 50% coverage."""
    mean, std = frontier(features, targets, site_ids)
    full = mean[-1]
    x = GRID * 100
    i50 = int(np.argmin(np.abs(GRID - 0.5)))
    gain = (mean[i50] - full) * 100
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.16, lw=0,
                    label=r"$\pm1$ std")
    ax.plot(x, mean, color=color, lw=2.5, zorder=4, label="Misfit-guided")
    ax.axhline(full, color=GRAY, ls=(0, (4, 3)), lw=1.2, zorder=2)
    ax.text(99, full, f"full coverage\n{full:.3f}", color="#6b7178", fontsize=7.4,
            ha="right", va="bottom")
    ax.scatter([50], [mean[i50]], s=60, color=ACCENT, zorder=6, edgecolor="white", lw=1.2)
    span = mean.max() - full
    ax.annotate(f"+{gain:.1f} pp\nat 50%", xy=(50, mean[i50]),
                xytext=(57, mean[i50] - 0.55 * span - 0.02), fontsize=9,
                color=ACCENT, fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.3))
    ax.set_xlabel("Coverage retained (%)")
    ax.set_ylabel("Accepted-set accuracy")
    ax.set_title(title, fontsize=9.8, fontweight="bold")
    ax.set_xlim(10, 100)
    ax.legend(loc="lower left", fontsize=8, frameon=True)
    sns.despine(ax=ax)
    print(f"  {title}: full={full:.3f} 50%={mean[i50]:.3f} +{gain:.1f}pp", flush=True)


def main():
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.5))
    fig.suptitle("Retained-set accuracy as a function of coverage (retrospective, leave-one-environment-out)",
                 fontsize=10.5, fontweight="bold", y=1.02)
    print("ADNI...", flush=True)
    panel(ax[0], *load_adni(), ADNI_C, "ADNI (scanner-protocol shift)")
    print("OASIS...", flush=True)
    panel(ax[1], *load_oasis_ds(), OASIS_C, "OASIS-1 (age-quartile shift)")
    fig.tight_layout()
    fig.savefig(OUT / "selective_prediction.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote selective_prediction.pdf", flush=True)


if __name__ == "__main__":
    main()
