"""Full external multi-site validation on ABIDE-I (20 sites, leave-one-site-out,
5 seeds). Computes, with the same validated collector used for ADNI/OASIS:
  - pooled failure-detection AUROC and rho(misfit, error),
  - WITHIN-SITE rho (per-subject test: does misfit rank error inside each held-out
    site, ruling out between-site difficulty inflation),
  - the misfit-guided selective-prediction frontier.
Writes results/abide_validation.json and paper/figures/abide_selective.pdf. GPU 2.
"""
from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

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
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

sys.path.insert(0, _resolve_path("<REPO_ROOT>"))
from misfit.data.abide_loader import load_abide  # noqa: E402
from misfit.experiments.run_clinical_stratified_media import collect  # noqa: E402

RES = Path(_resolve_path("<REPO_ROOT>/misfit/results"))
FIG = Path(_resolve_path("<REPO_ROOT>/misfit/paper/figures"))
N_SEEDS, SEED_BASE = 5, 42
GRID = np.arange(0.10, 1.001, 0.02)
sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
plt.rcParams.update({"pdf.fonttype": 42, "savefig.dpi": 300, "grid.color": "#dfe3e8"})
ABIDE_C, ACCENT, GRAY = "#7b3fa0", "#c0392b", "#9aa0a6"


def within_site_rho(mis, err, site, tested):
    """Average Spearman rho(misfit, error) inside each held-out site.

    Sites are skipped when fewer than 8 tested subjects or when all
    subjects share one outcome (rho undefined).

    Returns:
        Tuple (mean rho, fraction of sites with rho > 0, number of sites scored).
    """
    rhos = []
    for s in np.unique(site[tested]):
        m = tested & (site == s)
        e, mi = err[m], mis[m]
        if len(e) >= 8 and 0 < e.sum() < len(e):
            r = spearmanr(mi, e)[0]
            if not np.isnan(r):
                rhos.append(r)
    rhos = np.array(rhos)
    return float(rhos.mean()), float((rhos > 0).mean()), len(rhos)


def main():
    """Run the 5-seed ABIDE-I validation, write the JSON summary and figure."""
    d = load_abide()
    f, t, s = d.features, d.targets, d.site_ids
    print(f"ABIDE-I: N={len(f)} dim={f.shape[1]} sites={int(s.max())+1} "
          f"balance={float(t.float().mean()):.3f}", flush=True)
    aurocs, pooled_rho, ws_rho, ws_frac, ws_n, frontiers, base_acc = [], [], [], [], [], [], []
    for seed in range(N_SEEDS):
        c = collect(f, t, s, SEED_BASE + seed)
        m = c["tested"]
        mis, err, site = c["misfit"], c["err"], c["site"]
        mt, et = mis[m], err[m]
        aurocs.append(roc_auc_score(et, mt))
        pooled_rho.append(spearmanr(mt, et)[0])
        r, frac, n = within_site_rho(mis, err, site, m)
        ws_rho.append(r); ws_frac.append(frac); ws_n.append(n)
        order = np.argsort(mt)
        accs = [1.0 - et[order[:max(1, int(round(cov * len(mt))))]].mean() for cov in GRID]
        frontiers.append(accs)
        base_acc.append(1.0 - et.mean())
        print(f"  seed {seed}: AUROC={aurocs[-1]:.3f} pooled_rho={pooled_rho[-1]:.3f} "
              f"within-site_rho={r:.3f} ({frac*100:.0f}% of {n} sites +)", flush=True)

    fr = np.array(frontiers)
    i50 = int(np.argmin(np.abs(GRID - 0.5)))
    summary = {
        "n": int(len(f)), "n_sites": int(s.max()) + 1, "dim": int(f.shape[1]),
        "label_balance": float(t.float().mean()),
        "auroc_mean": float(np.mean(aurocs)), "auroc_std": float(np.std(aurocs)),
        "pooled_rho_mean": float(np.mean(pooled_rho)), "pooled_rho_std": float(np.std(pooled_rho)),
        "within_site_rho_mean": float(np.mean(ws_rho)), "within_site_rho_std": float(np.std(ws_rho)),
        "within_site_frac_positive": float(np.mean(ws_frac)), "sites_tested": int(np.mean(ws_n)),
        "base_acc_mean": float(np.mean(base_acc)),
        "selective_acc_50cov_mean": float(fr[:, i50].mean()),
        "selective_acc_full_mean": float(fr[:, -1].mean()),
    }
    json.dump(summary, open(RES / "abide_validation.json", "w"), indent=2)
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # selective-prediction figure
    mean, std = fr.mean(0), fr.std(0)
    full, gain = mean[-1], (mean[i50] - mean[-1]) * 100
    fig, ax = plt.subplots(figsize=(4.4, 3.3))
    x = GRID * 100
    ax.fill_between(x, mean - std, mean + std, color=ABIDE_C, alpha=0.16, lw=0, label=r"$\pm1$ std")
    ax.plot(x, mean, color=ABIDE_C, lw=2.5, zorder=4, label="Misfit-guided")
    ax.axhline(full, color=GRAY, ls=(0, (4, 3)), lw=1.2)
    ax.text(99, full, f"full coverage\n{full:.3f}", color="#6b7178", fontsize=7.4, ha="right", va="bottom")
    ax.scatter([50], [mean[i50]], s=60, color=ACCENT, zorder=6, edgecolor="white", lw=1.2)
    ax.annotate(f"+{gain:.1f} pp\nat 50%", xy=(50, mean[i50]), xytext=(57, mean[i50] - 0.13),
                fontsize=9, color=ACCENT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.3))
    ax.set_xlabel("Coverage retained (%)"); ax.set_ylabel("Accepted-set accuracy")
    ax.set_title("ABIDE-I generalization cohort (20 sites, autism fMRI):\nretained-set accuracy vs. coverage (near-chance base classifier)",
                 fontsize=8.8, fontweight="bold")
    ax.set_xlim(10, 100)
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(FIG / "abide_selective.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote abide_validation.json + abide_selective.pdf", flush=True)


if __name__ == "__main__":
    main()
