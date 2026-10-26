"""Regenerate the figures contradicted by the 2026-10-18 ADNI correction.

Authorized manuscript/figure update (2026-10-19). Each figure keeps the
historical script's layout and styling (sources listed per function) and
replaces ONLY the contaminated ADNI inputs with the verified corrected V2
outputs (participant-purged + per-fold source-only preprocessing). OASIS-1,
OASIS-3, ABIDE-I and UK Biobank inputs are byte-identical to the historical
analysis and re-enter unchanged.

Originals are preserved in correction_20261018/integration/figures_backup/.

Usage: python3 regen_figures_corrected.py fig1 fig2 ...
  figs: withinfold method forest subgroup reliability examples selective biology
"""
from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path, reporting_reference

import csv
import json
import os
import sys
from pathlib import Path

# CUDA visibility is selected by the caller.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

sys.path.insert(0, _resolve_path("<REPO_ROOT>"))

ROOT = Path(_resolve_path("<REPO_ROOT>/misfit"))
COR = ROOT / "correction_20261018"
RES = ROOT / "results"
FIG = ROOT / "paper" / "figures"
INT = COR / "integration" / "results"

sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
plt.rcParams.update({
    "pdf.fonttype": 42, "savefig.dpi": 300,
    "axes.edgecolor": "#43474d", "axes.labelcolor": "#22252a",
    "text.color": "#22252a", "xtick.color": "#43474d", "ytick.color": "#43474d",
    "grid.color": "#dfe3e8", "grid.linewidth": 0.7,
})
ADNI_C, OASIS_C, ACCENT, GRAY = "#2b6cb0", "#2f8f6b", "#c0392b", "#9aa0a6"
N_SEEDS = 5


def v2():
    """Verified corrected V2 summary JSON."""
    return json.load(open(COR / "results" / "corrected_v2_summaries.json"))


def perscan():
    """Stacked (seed, scan) misfit and error arrays plus labels and protocol ids."""
    z = np.load(COR / "reruns" / "purged_v2_perscan_local.npz", allow_pickle=True)
    mis = np.stack([z[f"misfit_s{s}"] for s in range(N_SEEDS)])
    err = np.stack([z[f"err_s{s}"].astype(int) for s in range(N_SEEDS)])
    return mis, err, z["label"].astype(int), z["site"].astype(int)


# ------------------------------------------------- within-fold forest
def fig_withinfold():
    """Source layout: run_result_figures_media.fig_withinfold()."""
    wec = json.load(open(RES / "within_env_canonical.json"))
    we = v2()["within_env_canonical"]
    ADNI = [(f"P{e}", we["per_protocol"][str(e)]["rho_mean"],
             we["per_protocol"][str(e)]["rho_std"],
             we["per_protocol"][str(e)]["n_test"]) for e in range(8)]
    OASIS = [(f"Q{e+1}", wec["oasis"]["per_env"][str(e)]["mean"],
              wec["oasis"]["per_env"][str(e)]["std"],
              wec["oasis"]["per_env"][str(e)]["n_test"]) for e in range(4)]
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    blocks = [("ADNI  (scanner-protocol shift; participant-purged)", ADNI, ADNI_C,
               round(we["mean_within_env_rho"], 3)),
              ("OASIS-1  (age-quartile shift)", OASIS, OASIS_C,
               round(wec["oasis"]["mean"], 3))]
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
    ax.set_title("The reliability direction holds in every held-out environment  (60/60 folds positive)",
                 fontsize=10.5, fontweight="bold", pad=10)
    ax.text(0.98, y - 0.1, "dot size $\\propto$ test-cohort size", fontsize=7.3,
            color="#9aa0a6", va="center", ha="right", style="italic")
    ax.grid(axis="y", visible=False)
    sns.despine(ax=ax, left=True)
    fig.tight_layout()
    fig.savefig(FIG / "within_fold_reliability.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote within_fold_reliability.pdf", flush=True)


# ------------------------------------------------- method comparison
def fig_method():
    """Source layout: run_result_figures_media.fig_method_comp()."""
    est = json.load(open(COR / "results" / "purged_v2_estimators.json"))["estimators"]
    oas = json.load(open(RES / "honest_estimator_table.json"))["estimator_table"]
    methods = [("structural_misfit", "Structural misfit"), ("erm_loss", "ERM training loss"),
               ("pac_bayes_full", "PAC-inspired diagnostic"), ("deep_ensemble5", "Deep ensemble"),
               ("max_softmax", "Max softmax"), ("mc_dropout", "MC dropout")]
    a_m = [est[k]["rho_mean"] for k, _ in methods]
    a_s = [est[k]["rho_std"] for k, _ in methods]
    o_m = [oas[k]["oasis"]["mean"] for k, _ in methods]
    o_s = [oas[k]["oasis"]["std"] for k, _ in methods]
    labels = [n for _, n in methods]
    y = np.arange(len(methods))[::-1]
    h = 0.38
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    for i in range(len(methods)):
        hot = i == 0
        ax.barh(y[i] + h / 2 + 0.02, a_m[i], height=h, xerr=a_s[i],
                color=ACCENT if hot else ADNI_C, alpha=0.95 if hot else 0.85,
                error_kw={"lw": 0.9, "ecolor": "#43474d"}, zorder=3)
        ax.barh(y[i] - h / 2 - 0.02, o_m[i], height=h, xerr=o_s[i],
                color="#e8a33d" if hot else OASIS_C, alpha=0.95 if hot else 0.8,
                error_kw={"lw": 0.9, "ecolor": "#43474d"}, zorder=3)
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
    ax.legend(handles=[Patch(color=ADNI_C, label="ADNI (protocol shift, participant-purged)"),
                       Patch(color=OASIS_C, label="OASIS-1 (age shift)")],
              loc="lower right", fontsize=8, frameon=True)
    ax.grid(axis="y", visible=False)
    sns.despine(ax=ax, left=True)
    fig.tight_layout()
    fig.savefig(FIG / "method_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote method_comparison.pdf", flush=True)


# ------------------------------------------------- cross-cohort forest
def fig_forest():
    """Source layout: fancy_figs/fig_cross_cohort_forest.py."""
    s = v2()
    rho_a = round(s["within_env_canonical"]["mean_within_env_rho"], 3)
    auc_a = s["pooled_auroc"]["mean"]
    auc_ci = 1.96 * s["pooled_auroc"]["std"]
    C = [row for row in reporting_reference("cross_cohort_forest") if row[0] != "ADNI"] + [("ADNI", "development; 8 protocols; participant-purged", rho_a, None,
          round(auc_a, 3), round(auc_ci, 3), True, False)]
    ADc, GENc, HL = "#8e44ad", "#2c7fb8", "#c0392b"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=True)
    y = np.arange(len(C))
    for ax, (vi, ci_i), xl, ref, reflab, xlim in [
            (a1, (2, 3), "Within-environment Spearman $\\rho$", 0.0, "no signal", (-0.05, 1.0)),
            (a2, (4, 5), "Failure-detection AUROC", 0.5, "chance", (0.5, 1.0))]:
        ax.axvline(ref, color="#b0b4b8", ls=(0, (4, 3)), lw=1.2, zorder=0)
        ax.text(ref, len(C) - 0.35, reflab, fontsize=7, color="#7a7f85", ha="center", va="bottom")
        for i, row in enumerate(C):
            val = row[vi]; ci = row[ci_i]
            col = HL if row[7] else (ADc if row[6] else GENc)
            ms = 120 if row[7] else 62
            if ci:
                ax.errorbar(val, i, xerr=ci, fmt="none", ecolor=col, elinewidth=1.8,
                            capsize=3, zorder=2, alpha=0.9)
            ax.scatter(val, i, s=ms, color=col, edgecolor="white", lw=1.4, zorder=3,
                       marker="D" if row[7] else "o")
            ax.text(val, i + 0.28, f"{val:.3f}", fontsize=7.6, ha="center", color=col,
                    fontweight="bold" if row[7] else "normal")
        ax.set_xlim(*xlim); ax.set_xlabel(xl, fontsize=9.5)
    a1.set_yticks(y); a1.set_yticklabels([f"{c[0]}" for c in C], fontsize=9.5)
    for i, c in enumerate(C):
        a1.text(-0.05, i - 0.30, c[1], fontsize=6.7, color="#8a8f95", ha="left",
                va="top", transform=a1.transData)
    for ax in (a1, a2):
        sns.despine(ax=ax, left=True); ax.tick_params(left=False)
    a1.set_ylim(-0.6, len(C) - 0.1)
    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], marker="D", color="w", markerfacecolor=HL, markersize=9,
                  label="OASIS-3 disease-matched replication"),
           Line2D([0], [0], marker="o", color="w", markerfacecolor=ADc, markersize=8,
                  label="Alzheimer cohorts"),
           Line2D([0], [0], marker="o", color="w", markerfacecolor=GENc, markersize=8,
                  label="Generalization cohorts")]
    fig.legend(handles=leg, loc="lower center", ncol=3, fontsize=7.6, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Retrospective per-subject reliability across five cohorts",
                 fontsize=11, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(FIG / "cross_cohort_forest.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("wrote cross_cohort_forest.pdf", flush=True)


# ------------------------------------------------- subgroup robustness
def fig_subgroup():
    """Source layout: run_subgroup_media.main(); ADNI panel from the corrected
    participant-blocked integration stats, OASIS-1 panel unchanged
    (results/subgroup.json)."""
    stats = json.load(open(INT / "integration_stats.json"))
    adni = [{"group": g["group"], "n": g["n_acq"], "auroc": g["auroc"],
             "lo": g["lo"], "hi": g["hi"]} for g in stats["subgroup"]["groups"]]
    oasis = json.load(open(RES / "subgroup.json"))["oasis"]
    fig, ax = plt.subplots(2, 1, figsize=(3.5, 4.6), sharex=True)
    for k, (res, color, title) in enumerate([
            (adni, ADNI_C, "ADNI (scanner shift,\nparticipant-purged)"),
            (oasis, OASIS_C, "OASIS-1 (age shift)")]):
        y = np.arange(len(res))[::-1]
        overall = res[0]["auroc"]
        ax[k].axvline(overall, color=GRAY, ls=(0, (4, 3)), lw=1.2, zorder=1)
        for i, r in enumerate(res):
            ax[k].hlines(y[i], r["lo"], r["hi"], color=color, lw=1.8, alpha=0.6, zorder=2)
            ax[k].scatter(r["auroc"], y[i], s=58, color=color, edgecolor="white", lw=1.1, zorder=3)
            ax[k].text(0.50, y[i] + 0.30, f"n={r['n']}", fontsize=6.2, color="#6b7178")
        ax[k].set_yticks(y)
        ax[k].set_yticklabels([r["group"] for r in res], fontsize=8.5)
        ax[k].set_title(title, fontsize=9.2, fontweight="bold")
        ax[k].axvline(0.5, color="#c0392b", lw=0.8, alpha=0.5)
        ax[k].set_xlim(0.48, 1.0)
        ax[k].set_ylim(-0.6, len(res) - 0.4)
        sns.despine(ax=ax[k], left=True)
        ax[k].grid(axis="y", visible=False)
    ax[1].set_xlabel("Failure-detection AUROC")
    fig.suptitle("Reliability detection across\nsex and age subgroups",
                 fontsize=9.8, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "subgroup_robustness.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote subgroup_robustness.pdf", flush=True)


# ------------------------------------------------- reliability biology (ADNI)
def _violin(ax, groups, labels, colors, ylabel):
    """Violin bodies with jittered points and median bars on a 0-100 percentile axis."""
    parts = ax.violinplot(list(groups), showextrema=False, widths=0.8)
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c)
        pc.set_alpha(0.35)
    rng = np.random.RandomState(0)
    for i, (g, c) in enumerate(zip(groups, colors), 1):
        x = i + rng.uniform(-0.13, 0.13, len(g))
        ax.scatter(x, g, s=3, color=c, alpha=0.45, lw=0)
        ax.hlines(np.median(g), i - 0.28, i + 0.28, color=c, lw=2.2, zorder=5)
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylim(-3, 103)
    ax.set_ylabel(ylabel, fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_reliability():
    """Source layout: run_reliability_biology_reexport.py (ADNI panel)."""
    TEAL, CORAL = "#0f6b74", "#c0392b"
    plt.rcParams.update({"font.size": 8.5})
    mis, err, label, site = perscan()
    ma = mis.mean(0)
    pa = 100.0 * rankdata(ma) / len(ma)
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
    _violin(ax[0], [pa[label == 0], pa[label == 1]],
            [f"CN (n={(label==0).sum()})", f"impaired (n={(label==1).sum()})"],
            [TEAL, CORAL], "misfit percentile")
    ax[0].set_title("ADNI (corrected): training loss by group", fontsize=8.5)
    groups = [pa[site == e] for e in np.unique(site)]
    _violin(ax[1], groups, [f"P{e}" for e in np.unique(site)],
            ["#5a8f9b"] * len(groups), "misfit percentile")
    ax[1].set_title("ADNI (corrected): by scanner protocol", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / "adni_reliability_biology.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("wrote adni_reliability_biology.pdf", flush=True)


# ------------------------------------------------- examples
def fig_examples():
    """Source layout: run_image_examples_render.py; corrected ADNI picks from
    integration stats (always-correct / always-wrong across the 5 corrected
    seeds), OASIS-1 picks unchanged."""
    import nibabel as nib
    from misfit.experiments.run_image_examples_media import adni_paths, oasis_paths
    plt.rcParams.update({"font.size": 9})
    ADNI_CLS = {0: "CN", 1: "impaired"}
    OASIS_CLS = {0: "healthy", 1: "demented"}
    stats = json.load(open(INT / "integration_stats.json"))["examples"]
    a_rel, a_unr = stats["most_reliable"]["index"], stats["least_reliable"]["index"]
    la = {a_rel: stats["most_reliable"]["label"], a_unr: stats["least_reliable"]["label"]}
    d = np.load(RES / "per_subject_misfit.npz")
    mo, eo, lo = d["oasis_misfit"], d["oasis_err"], d["oasis_label"]
    rel_pool = np.where(eo == 0)[0]
    unr_pool = np.where(eo == 1)[0]
    o_rel = int(rel_pool[mo[rel_pool].argmin()])
    o_unr = int(unr_pool[mo[unr_pool].argmax()])
    pa, _ = adni_paths()
    po, _ = oasis_paths()

    def cap(cohort, lab, cls, reliable):
        t = cls[int(lab)]
        pred = cls[int(lab)] if reliable else cls[1 - int(lab)]
        tag = "most reliable" if reliable else "least reliable"
        verdict = "correct" if reliable else "wrong"
        return f"{cohort}: {tag}\ntrue {t} $\\to$ pred {pred}\n({verdict})"

    def render(ax, path, k, title, vmax):
        ax.imshow(np.rot90(nib.load(path).get_fdata(dtype=np.float32)[:, :, k]),
                  cmap="gray", vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=7.5)
        ax.set_xticks([])
        ax.set_yticks([])

    ka = 60
    ko = nib.load(po[o_rel]).shape[2] // 2
    va = float(np.percentile(nib.load(pa[a_rel]).get_fdata(dtype=np.float32), 99))
    vo = float(np.percentile(nib.load(po[o_rel]).get_fdata(dtype=np.float32), 99))
    fig, ax = plt.subplots(1, 4, figsize=(10.4, 3.0))
    render(ax[0], pa[a_rel], ka, cap("ADNI", la[a_rel], ADNI_CLS, True), va)
    render(ax[1], pa[a_unr], ka, cap("ADNI", la[a_unr], ADNI_CLS, False), va)
    render(ax[2], po[o_rel], ko, cap("OASIS", lo[o_rel], OASIS_CLS, True), vo)
    render(ax[3], po[o_unr], ko, cap("OASIS", lo[o_unr], OASIS_CLS, False), vo)
    fig.suptitle("Gray-matter maps at the reliability extremes (most vs least reliable case per cohort)",
                 fontsize=9.5, y=1.06)
    fig.subplots_adjust(wspace=0.06, top=0.80)
    fig.savefig(FIG / "misfit_examples.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"ADNI rel idx={a_rel} (label {la[a_rel]}), unr idx={a_unr} (label {la[a_unr]})", flush=True)
    print("wrote misfit_examples.pdf", flush=True)


# ------------------------------------------------- selective frontier
GRID = np.arange(0.10, 1.001, 0.02)


def _frontier_from_arrays(mis, err):
    """Per-seed accepted-set accuracy over the coverage GRID (accept lowest
    misfit first); returns mean and std across seeds."""
    per_seed = []
    for s in range(mis.shape[0]):
        order = np.argsort(mis[s])
        accs = [1.0 - err[s][order[:max(1, int(round(c * mis.shape[1])))]].mean()
                for c in GRID]
        per_seed.append(accs)
    a = np.array(per_seed)
    return a.mean(0), a.std(0)


def fig_selective():
    """Source layout: run_selective_fig_pretty.py; ADNI frontier from the
    corrected V2 per-scan arrays, OASIS-1 recomputed live via the unchanged
    historical collector (identical to the original script)."""
    from misfit.experiments.run_clinical_stratified_media import collect
    from misfit.experiments.run_honest_estimator_table import (
        N_SEEDS as NS, SEED_BASE, load_oasis_ds,
    )
    mis, err, _, _ = perscan()
    adni_mean, adni_std = _frontier_from_arrays(mis, err.astype(float))

    feats, targets, sites = load_oasis_ds()
    per_seed = []
    for s in range(NS):
        d = collect(feats, targets, sites, SEED_BASE + s)
        t = d["tested"]
        m_, e_ = d["misfit"][t], d["err"][t]
        order = np.argsort(m_)
        per_seed.append([1.0 - e_[order[:max(1, int(round(c * len(m_))))]].mean()
                         for c in GRID])
    oas = np.array(per_seed)
    oasis_mean, oasis_std = oas.mean(0), oas.std(0)

    def panel(ax, mean, std, color, title):
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
        span = max(mean.max() - full, 0.04)
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

    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.5))
    fig.suptitle("Retained-set accuracy as a function of coverage (retrospective, leave-one-environment-out)",
                 fontsize=10.5, fontweight="bold", y=1.02)
    panel(ax[0], adni_mean, adni_std, ADNI_C, "ADNI (scanner-protocol shift, participant-purged)")
    panel(ax[1], oasis_mean, oasis_std, OASIS_C, "OASIS-1 (age-quartile shift)")
    fig.tight_layout()
    fig.savefig(FIG / "selective_prediction.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote selective_prediction.pdf", flush=True)


# ------------------------------------------------- biology composite
def fig_biology():
    """Source layout: run_biology_composite.py; ADNI inputs corrected (V2
    misfit, corrected signature map), OASIS-1 inputs unchanged."""
    import nibabel as nib
    from matplotlib import gridspec
    from nilearn import plotting
    from scipy.stats import ks_2samp, spearmanr
    from misfit.experiments.run_biology_composite import oasis_paths_cdr
    from misfit.experiments.run_image_examples_media import adni_paths

    TEAL, CORAL = "#0f6b74", "#c0392b"
    plt.rcParams.update({"font.size": 8.5})
    mis, err, la, site = perscan()
    ma = mis.mean(0)
    dx = np.load(RES / "per_subject_misfit.npz")
    xmo, xeo, xlo = dx["oasis_misfit"], dx["oasis_err"], dx["oasis_label"]
    stats = json.load(open(INT / "integration_stats.json"))

    fig = plt.figure(figsize=(7.2, 8.2))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1.0, 0.80, 0.85], hspace=0.42)

    # (a) stratification
    gsa = gridspec.GridSpecFromSubplotSpec(1, 2, gs[0], wspace=0.3)
    a1 = fig.add_subplot(gsa[0])
    pca = 100.0 * rankdata(ma) / len(ma)
    cn, imp = pca[la == 0], pca[la == 1]
    ks = ks_2samp(ma[la == 0], ma[la == 1])
    ratio = ma[la == 0].mean() / ma[la == 1].mean()
    _violin(a1, [cn, imp], [f"CN\n(n={len(cn)})", f"impaired\n(n={len(imp)})"],
            [TEAL, CORAL], "misfit percentile (cohort rank)")
    a1.set_title(f"ADNI (corrected): impaired subjects are fit more reliably\n"
                 f"mean misfit ratio {ratio:.1f}$\\times$; KS $D$={ks.statistic:.2f}, "
                 f"$p<10^{{-29}}$", fontsize=8.5)
    _, cdr = oasis_paths_cdr()
    assert len(cdr) == len(xmo)
    pco = 100.0 * rankdata(xmo) / len(xmo)
    stages = [pco[cdr == 0], pco[cdr == 0.5], pco[cdr >= 1]]
    stage_idx = np.where(cdr == 0, 0, np.where(cdr == 0.5, 1, 2))
    rho_s = spearmanr(stage_idx, xmo)
    a2 = fig.add_subplot(gsa[1])
    _violin(a2, stages, [f"CDR 0\n(n={len(stages[0])})", f"CDR 0.5\n(n={len(stages[1])})",
                         f"CDR$\\geq$1\n(n={len(stages[2])})"],
            [TEAL, "#b07fd6", CORAL], "misfit percentile (cohort rank)")
    a2.set_title(f"OASIS-1: higher CDR, less reliable fit\nSpearman(stage, misfit) "
                 f"$\\rho$={rho_s.statistic:.2f}, $p$={rho_s.pvalue:.2g}", fontsize=8.5)

    # (b) failure cases
    pa, _ = adni_paths()
    po, _ = oasis_paths_cdr()
    ACLS, OCLS = {0: "CN", 1: "impaired"}, {0: "healthy", 1: "demented"}
    ex = stats["examples"]
    a_rel, a_unr = ex["most_reliable"]["index"], ex["least_reliable"]["index"]
    a_lab = {a_rel: ex["most_reliable"]["label"], a_unr: ex["least_reliable"]["label"]}
    rel_pool = np.where(xeo == 0)[0]
    unr_pool = np.where(xeo == 1)[0]
    o_rel = int(rel_pool[xmo[rel_pool].argmin()])
    o_unr = int(unr_pool[xmo[unr_pool].argmax()])
    gsb = gridspec.GridSpecFromSubplotSpec(1, 4, gs[1], wspace=0.05)

    def render(pos, path, k, title, vmax, edge):
        ax = fig.add_subplot(gsb[pos])
        ax.imshow(np.rot90(nib.load(path).get_fdata(dtype=np.float32)[:, :, k]),
                  cmap="gray", vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=7.2, color=edge)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(edge)
            s.set_linewidth(1.4)

    ka = 60
    ko = nib.load(po[o_rel]).shape[2] // 2
    va = float(np.percentile(nib.load(pa[a_rel]).get_fdata(dtype=np.float32), 99))
    vo = float(np.percentile(nib.load(po[o_rel]).get_fdata(dtype=np.float32), 99))
    wrong_tag = "FN" if a_lab[a_unr] == 1 else "FP"
    render(0, pa[a_rel], ka, f"ADNI most reliable\ntrue {ACLS[a_lab[a_rel]]} "
           f"$\\to$ correct", va, TEAL)
    render(1, pa[a_unr], ka, f"ADNI least reliable\ntrue {ACLS[a_lab[a_unr]]} "
           f"$\\to$ wrong ({wrong_tag})", va, CORAL)
    render(2, po[o_rel], ko, f"OASIS-1 most reliable\ntrue {OCLS[int(xlo[o_rel])]} "
           f"$\\to$ correct", vo, TEAL)
    render(3, po[o_unr], ko, f"OASIS-1 least reliable\ntrue {OCLS[int(xlo[o_unr])]} "
           f"$\\to$ wrong (FP)", vo, CORAL)

    # (c) anatomical maps
    def zscore_brain(volm):
        m = volm != 0
        out = np.zeros_like(volm, dtype=np.float32)
        v = volm[m]
        out[m] = (v - v.mean()) / (v.std() + 1e-12)
        return out

    sig = json.load(open(RES / "image_signature.json"))
    gsc = gridspec.GridSpecFromSubplotSpec(1, 2, gs[2], wspace=0.12)
    c1 = fig.add_subplot(gsc[0])
    img = nib.load(INT / "adni_misfit_signature_corrected.nii.gz")
    zimg = nib.Nifti1Image(zscore_brain(np.asarray(img.dataobj)), img.affine)
    plotting.plot_stat_map(zimg, axes=c1, colorbar=True, display_mode="z",
                           cut_coords=[-20, 10, 45], threshold=0.8, vmax=2.6,
                           cmap="RdBu_r", annotate=False)
    c1.set_title("ADNI (corrected): voxelwise GM association ($z$)", fontsize=8.5)
    c2 = fig.add_subplot(gsc[1])
    roi = np.array(sig["oasis"]["roi_signature"], float)
    roi_z = (roi - roi.mean()) / (roi.std() + 1e-12)
    from nilearn.datasets import fetch_atlas_schaefer_2018
    atlas = fetch_atlas_schaefer_2018(n_rois=100)
    lab_img = nib.load(atlas.maps)
    labvol = np.asarray(lab_img.dataobj).astype(int)
    statvol = np.zeros(labvol.shape, np.float32)
    for j in range(1, 101):
        statvol[labvol == j] = roi_z[j - 1]
    plotting.plot_stat_map(nib.Nifti1Image(statvol, lab_img.affine), axes=c2, colorbar=True,
                           display_mode="z", cut_coords=[-20, 10, 45], threshold=0.8,
                           vmax=2.6, cmap="RdBu_r", annotate=False)
    c2.set_title("OASIS-1: Schaefer-100 GM association ($z$)", fontsize=8.5)

    for lab_, y_ in (("(a)", 0.985), ("(b)", 0.615), ("(c)", 0.36)):
        fig.text(0.015, y_, lab_, fontsize=11, fontweight="bold")
    fig.savefig(FIG / "biology_composite.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"ADNI ratio {ratio:.2f} KS D={ks.statistic:.3f} p={ks.pvalue:.2g}", flush=True)
    print("wrote biology_composite.pdf", flush=True)


ALL = {"withinfold": fig_withinfold, "method": fig_method, "forest": fig_forest,
       "subgroup": fig_subgroup, "reliability": fig_reliability,
       "examples": fig_examples, "selective": fig_selective, "biology": fig_biology}

if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    for name in (sys.argv[1:] or list(ALL)):
        print(f"== {name} ==", flush=True)
        ALL[name]()
