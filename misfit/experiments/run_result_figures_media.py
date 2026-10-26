"""Result figures for the MIA paper (polished), to replace tables (no result
shown in both a figure and a table) and to add a conceptual overview.

  1. overview.pdf                  conceptual method schematic (Fig. 1)
  2. within_fold_reliability.pdf   forest-style dot-and-whisker of per-environment
                                   within-fold Spearman rho (replaces the two
                                   within-fold tables + the OASIS per-quartile figure)
  3. transfer_drift_degradation.pdf  reconstruction-residual drift vs ranking
                                     degradation across the 8 ADNI protocols
                                     (visualizes the transfer Proposition test)

All numbers are the canonical reported values (within-fold tables; transfer
rescue JSON). No model re-training. Writes into paper/figures/.
"""
from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

FIG = Path(_resolve_path("<REPO_ROOT>/misfit/paper/figures"))
RES = Path(_resolve_path("<REPO_ROOT>/misfit/results"))

sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
plt.rcParams.update({
    "pdf.fonttype": 42, "savefig.dpi": 300,
    "axes.edgecolor": "#43474d", "axes.labelcolor": "#22252a",
    "text.color": "#22252a", "xtick.color": "#43474d", "ytick.color": "#43474d",
    "grid.color": "#dfe3e8", "grid.linewidth": 0.7,
})
ADNI_C, OASIS_C, ACCENT = "#2b6cb0", "#2f8f6b", "#c0392b"

# canonical cross-refit-score within-environment values (results/within_env_canonical.json;
# replaces the withdrawn test-CE lineage per AT-S05)
_wec = json.load(open(RES / "within_env_canonical.json"))
ADNI = [(f"P{e}", _wec["adni"]["per_env"][str(e)]["mean"], _wec["adni"]["per_env"][str(e)]["std"],
         _wec["adni"]["per_env"][str(e)]["n_test"]) for e in range(8)]
OASIS = [(f"Q{e+1}", _wec["oasis"]["per_env"][str(e)]["mean"], _wec["oasis"]["per_env"][str(e)]["std"],
          _wec["oasis"]["per_env"][str(e)]["n_test"]) for e in range(4)]


# ----------------------------------------------------------------- within-fold
def fig_withinfold():
    """Forest-style per-environment within-fold rho panel for both cohorts,
    dot size proportional to test-cohort size, mean band per block."""
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    blocks = [("ADNI  (scanner-protocol shift)", ADNI, ADNI_C, round(_wec["adni"]["mean"], 3)),
              ("OASIS-1  (age-quartile shift)", OASIS, OASIS_C, round(_wec["oasis"]["mean"], 3))]
    y = 0
    yticks, ylabels = [], []
    smax = max(n for _, _, _, n in ADNI + OASIS)
    for title, rows, color, mean in blocks:
        y -= 1.7
        ax.text(0.02, y + 0.05, title, fontsize=10, fontweight="bold", color=color)
        ys = []
        for name, rho, std, n in rows:
            y -= 1
            ys.append(y)
            yticks.append(y)
            ylabels.append(name)
            ax.hlines(y, rho - std, rho + std, color=color, lw=1.6, alpha=0.55, zorder=2)
            ax.scatter(rho, y, s=70 + 360 * (n / smax), color=color, alpha=0.92,
                       edgecolor="white", lw=1.3, zorder=3)
            ax.text(0.035, y, f"$N{{=}}{n}$", fontsize=7.3, color="#6b7178",
                    va="center", ha="left")
        # mean band over this block
        ax.fill_betweenx([min(ys) - 0.45, max(ys) + 0.45], mean - 0.004, mean + 0.004,
                         color=color, alpha=0.18, zorder=1)
        ax.vlines(mean, min(ys) - 0.45, max(ys) + 0.45, color=color, ls=(0, (4, 3)),
                  lw=1.3, zorder=1)
        ax.text(mean, max(ys) + 0.62, f"mean {mean:.3f}", fontsize=7.6, color=color,
                ha="center", style="italic")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(y - 0.6, 0.1)
    ax.set_xlabel(r"Within-fold Spearman $\rho$  (per-subject misfit vs. error)")
    ax.set_title("Reliability holds in every held-out environment  (60/60 folds positive)",
                 fontsize=10.5, fontweight="bold", pad=10)
    ax.text(0.98, y - 0.1, "dot size $\\propto$ test-cohort size", fontsize=7.3,
            color="#9aa0a6", va="center", ha="right", style="italic")
    ax.grid(axis="y", visible=False)
    sns.despine(ax=ax, left=True)
    fig.tight_layout()
    fig.savefig(FIG / "within_fold_reliability.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote within_fold_reliability.pdf", flush=True)


# ----------------------------------------------------------------- transfer
def fig_transfer():
    """Scatter of reconstruction-residual drift vs canonical ranking
    degradation across the ADNI protocols, colored by protocol error rate;
    drift and degradation come from transfer_canonical.json, error rates from
    the legacy transfer_rescue.json rows."""
    r = json.load(open(RES / "transfer_canonical.json"))
    legacy = json.load(open(RES / "transfer_rescue.json"))["adni"]["per_protocol"]
    legacy = sorted(legacy, key=lambda v: v["protocol"])
    d = np.array(r["drift"])
    deg = np.array(r["canonical_abs_delta"])
    err = np.array([p["err_rate"] for p in legacy])
    rho = r["spearman_drift_vs_degradation"]["rho"]
    pval = r["spearman_drift_vs_degradation"]["p"]

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    norm = Normalize(err.min(), err.max())
    cmap = plt.get_cmap("viridis")
    ax.scatter(d, deg, c=err, cmap=cmap, norm=norm, s=70, edgecolor="#43474d",
               lw=1.2, zorder=4)
    for i, (xi, yi) in enumerate(zip(d, deg)):
        ax.annotate(f"P{i}", (xi, yi), textcoords="offset points", xytext=(0, 11),
                    fontsize=7.6, color="#43474d", ha="center", zorder=5)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02, fraction=0.046)
    cb.set_label("Protocol error rate", fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)
    ax.set_xlabel(r"Reconstruction-residual drift $d_e$  (atypicality of held-out protocol)")
    ax.set_ylabel(r"Canonical ranking degradation $|\rho^{\mathrm{tr}}-\rho^{\mathrm{te}}|$")
    ax.set_title("Drift vs. canonical ranking degradation:\nno significant rank association",
                 fontsize=10, fontweight="bold")
    ax.text(0.04, 0.93, f"Spearman $\\rho = {rho:.2f}$\n$p = {pval:.2f}$  (8 protocols)",
            transform=ax.transAxes, fontsize=9.5, color="#43474d",
            va="top", bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#8a96a3", lw=1.0, alpha=0.95))
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(FIG / "transfer_drift_degradation.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote transfer_drift_degradation.pdf (rho={rho:.3f}, p={pval:.3f})", flush=True)


# ----------------------------------------------------------------- overview
def _shadow_box(ax, x, y, w, h, fc, ec):
    ax.add_patch(FancyBboxPatch((x + 0.03, y - 0.04), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.04",
                                fc="#d7dbe0", ec="none", alpha=0.6, zorder=1))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.04",
                                fc=fc, ec=ec, lw=1.4, zorder=2))


def _arrow(ax, x0, x1, yy):
    ax.add_patch(FancyArrowPatch((x0, yy), (x1, yy), arrowstyle="-|>",
                                 mutation_scale=16, lw=1.7, color="#5a6068", zorder=1))


def fig_overview():
    """Conceptual schematic: multi-site data -> trained classifier ->
    per-subject misfit -> accept/abstain reliability audit."""
    plt.rcParams.update({"axes.grid": False})
    fig, ax = plt.subplots(figsize=(9.6, 2.7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.7)
    ax.axis("off")

    _shadow_box(ax, 0.1, 0.45, 1.95, 1.65, "#eaf1fb", ADNI_C)
    ax.text(1.07, 1.92, "Multi-site MRI", ha="center", fontsize=9.5, fontweight="bold", color=ADNI_C)
    for cx, col, lab in [(0.58, ADNI_C, "Site A"), (1.07, OASIS_C, "Site B"), (1.57, ACCENT, "Site C")]:
        ax.add_patch(plt.Circle((cx, 1.28), 0.18, fc=col, ec="white", lw=1.2, alpha=0.9, zorder=3))
        ax.text(cx, 0.72, lab, ha="center", fontsize=6.6, color=col)
    ax.text(1.07, 0.55, "distribution shift", ha="center", fontsize=6.6, style="italic", color="#6b7178")

    _shadow_box(ax, 2.55, 0.75, 1.7, 1.05, "#f3f4f6", "#5a6068")
    ax.text(3.4, 1.46, "Trained", ha="center", fontsize=9.5)
    ax.text(3.4, 1.16, r"classifier $f(x)$", ha="center", fontsize=9.5)

    _shadow_box(ax, 4.75, 0.45, 2.05, 1.65, "#fdf6ec", "#b9770e")
    ax.text(5.78, 1.92, "Structural misfit", ha="center", fontsize=9.5, fontweight="bold", color="#b9770e")
    ax.text(5.78, 1.64, r"$L_i =$ per-subject loss", ha="center", fontsize=7.4)
    grad = np.linspace(0, 1, 100).reshape(-1, 1)
    ax.imshow(grad, extent=[5.0, 5.32, 0.66, 1.44], aspect="auto", cmap="RdYlGn", zorder=3)
    ax.add_patch(plt.Rectangle((5.0, 0.66), 0.32, 0.78, fill=False, ec="#b9770e", lw=0.8, zorder=4))
    ax.text(5.42, 1.34, "high $L_i$: unreliable", fontsize=6.5, va="center", color=ACCENT)
    ax.text(5.42, 0.76, "low $L_i$: reliable", fontsize=6.5, va="center", color=OASIS_C)

    _shadow_box(ax, 7.3, 0.45, 2.55, 1.65, "#ecf6f0", OASIS_C)
    ax.text(8.57, 1.92, "Reliability audit", ha="center", fontsize=9.5, fontweight="bold", color=OASIS_C)
    ax.text(8.57, 1.62, "rank subjects, accept reliable,", ha="center", fontsize=6.7)
    ax.text(8.57, 1.42, "abstain on unreliable", ha="center", fontsize=6.7)
    ax.text(7.62, 1.04, "✓", fontsize=13, color=OASIS_C, fontweight="bold")
    ax.text(7.95, 1.02, "accepted (accurate)", fontsize=6.5, va="center")
    ax.text(7.62, 0.7, "✗", fontsize=13, color=ACCENT, fontweight="bold")
    ax.text(7.95, 0.68, "abstained (would fail)", fontsize=6.5, va="center")

    for x0, x1 in [(2.05, 2.55), (4.25, 4.75), (6.8, 7.3)]:
        _arrow(ax, x0, x1, 1.28)
    fig.savefig(FIG / "overview.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote overview.pdf", flush=True)


# ----------------------------------------------------------------- method comparison
def fig_method_comp():
    """Horizontal bar chart of estimator-vs-error Spearman rho (mean +/- std)
    per cohort, read from honest_estimator_table.json."""
    d = json.load(open(RES / "honest_estimator_table.json"))["estimator_table"]
    methods = [("structural_misfit", "Structural misfit"), ("erm_loss", "ERM training loss"),
               ("pac_bayes_full", "PAC-inspired diagnostic"), ("deep_ensemble5", "Deep ensemble"),
               ("max_softmax", "Max softmax"), ("mc_dropout", "MC dropout")]
    a_m = [d[k]["adni"]["mean"] for k, _ in methods]
    a_s = [d[k]["adni"]["std"] for k, _ in methods]
    o_m = [d[k]["oasis"]["mean"] for k, _ in methods]
    o_s = [d[k]["oasis"]["std"] for k, _ in methods]
    labels = [n for _, n in methods]
    y = np.arange(len(methods))[::-1]
    h = 0.38

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    for i in range(len(methods)):
        hot = i == 0
        ax.barh(y[i] + h / 2 + 0.02, a_m[i], height=h, xerr=a_s[i],
                color=ACCENT if hot else ADNI_C, alpha=0.95 if hot else 0.85,
                error_kw={"lw": 0.9, "ecolor": "#43474d"}, zorder=3,
                label="ADNI" if i == 1 else None)
        ax.barh(y[i] - h / 2 - 0.02, o_m[i], height=h, xerr=o_s[i],
                color="#e8a33d" if hot else OASIS_C, alpha=0.95 if hot else 0.8,
                error_kw={"lw": 0.9, "ecolor": "#43474d"}, zorder=3,
                label="OASIS-1" if i == 1 else None)
        ax.text(a_m[i] + a_s[i] + 0.012, y[i] + h / 2 + 0.02, f"{a_m[i]:.2f}",
                va="center", fontsize=7, color="#43474d")
        ox = o_m[i] + o_s[i] + 0.012 if o_m[i] >= 0 else o_m[i] - o_s[i] - 0.012
        ax.text(ox, y[i] - h / 2 - 0.02, f"{o_m[i]:.2f}", va="center",
                fontsize=7, color="#43474d", ha="left" if o_m[i] >= 0 else "right")
    ax.axvline(0, color="#43474d", lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.set_xlabel(r"Spearman $\rho$ (estimator vs. per-subject error), 5-seed mean $\pm$ std")
    ax.set_xlim(-0.12, 0.72)
    ax.set_title("Reliability estimators vs. per-subject error under environment shift",
                 fontsize=10.5, fontweight="bold", pad=8)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=ADNI_C, label="ADNI (protocol shift)"),
                       Patch(color=OASIS_C, label="OASIS-1 (age shift)")],
              loc="lower right", fontsize=8, frameon=True)
    ax.grid(axis="y", visible=False)
    sns.despine(ax=ax, left=True)
    fig.tight_layout()
    fig.savefig(FIG / "method_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote method_comparison.pdf", flush=True)


if __name__ == "__main__":
    fig_withinfold()
    fig_transfer()
    fig_overview()
    fig_method_comp()
