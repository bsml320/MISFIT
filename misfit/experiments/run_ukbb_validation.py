"""UK Biobank reliability validation: does structural misfit rank per-subject
prediction error at population scale, on a different task (sex) and shift axis (age quartiles)?

Same validated IOP collector as ADNI/OASIS/ABIDE (leave-one-environment-out, deployable
training-side misfit, binary per-subject error target). Cohort size defaults to the full
extract and can be capped with argv[1]. GPU 2. 5 seeds.
Output: misfit/results/ukbb/ukbb_validation.json + ukbb_selective.pdf.

ATTRIBUTION: UKBB data use requires Anton Orlichenko + Dr. Ray Su (Tulane) as co-authors.
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
from misfit.data.ukbb_loader import load_ukbb  # noqa: E402
from misfit.experiments.run_clinical_stratified_media import collect  # noqa: E402

RES = Path(_resolve_path("<REPO_ROOT>/misfit/results/ukbb"))
N_SEEDS, SEED_BASE = 5, 42
N_SUBJECTS = int(sys.argv[1]) if len(sys.argv) > 1 else 45000   # default: full cohort (caps at ~42,779)
GRID = np.arange(0.10, 1.001, 0.02)
sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
plt.rcParams.update({"pdf.fonttype": 42, "savefig.dpi": 300, "grid.color": "#dfe3e8"})
UKBB_C, ACCENT, GRAY = "#0a7d5a", "#c0392b", "#9aa0a6"


def within_env_rho(mis, err, site, tested):
    """Mean within-environment Spearman rho over scoreable environments.

    An environment is scored if it has at least 8 tested subjects and a
    non-degenerate error vector.

    Returns:
        (mean rho, fraction of environments with rho > 0, n scored).
    """
    rhos = []
    for e in np.unique(site[tested]):
        m = tested & (site == e)
        ee, mm = err[m], mis[m]
        if len(ee) >= 8 and 0 < ee.sum() < len(ee):
            r = spearmanr(mm, ee)[0]
            if not np.isnan(r):
                rhos.append(r)
    rhos = np.array(rhos)
    return float(rhos.mean()), float((rhos > 0).mean()), len(rhos)


def main():
    """Run the collector over seeds; write the summary JSON and the
    selective-prediction figure."""
    RES.mkdir(parents=True, exist_ok=True)
    d = load_ukbb(n_subjects=N_SUBJECTS)
    f, t, s = d.features, d.targets, d.site_ids
    print(f"UKBB: N={len(f)} dim={f.shape[1]} envs={int(s.max())+1} "
          f"sex-balance={float(t.float().mean()):.3f} age[{d.age.min()}-{d.age.max()}] "
          f"voxels={d.n_voxels}", flush=True)

    aurocs, pooled, we_rho, we_frac, frontiers, base_acc = [], [], [], [], [], []
    for seed in range(N_SEEDS):
        c = collect(f, t, s, SEED_BASE + seed)
        m = c["tested"]
        mis, err, site = c["misfit"], c["err"], c["site"]
        mt, et = mis[m], err[m]
        aurocs.append(roc_auc_score(et, mt))
        pooled.append(spearmanr(mt, et)[0])
        r, frac, n = within_env_rho(mis, err, site, m)
        we_rho.append(r); we_frac.append(frac)
        order = np.argsort(mt)
        frontiers.append([1.0 - et[order[:max(1, int(round(cov * len(mt))))]].mean() for cov in GRID])
        base_acc.append(1.0 - et.mean())
        print(f"  seed {seed}: AUROC={aurocs[-1]:.3f} pooled_rho={pooled[-1]:.3f} "
              f"within-env_rho={r:.3f} ({frac*100:.0f}% of {n} envs +) base_acc={base_acc[-1]:.3f}", flush=True)

    fr = np.array(frontiers)
    i50 = int(np.argmin(np.abs(GRID - 0.5)))
    summary = {
        "n": int(len(f)), "n_envs": int(s.max()) + 1, "dim": int(f.shape[1]),
        "task": "sex", "shift_axis": "age_quartile", "voxels": d.n_voxels,
        "sex_balance": float(t.float().mean()),
        "auroc_mean": float(np.mean(aurocs)), "auroc_std": float(np.std(aurocs)),
        "pooled_rho_mean": float(np.mean(pooled)), "pooled_rho_std": float(np.std(pooled)),
        "within_env_rho_mean": float(np.mean(we_rho)), "within_env_rho_std": float(np.std(we_rho)),
        "within_env_frac_positive": float(np.mean(we_frac)),
        "base_acc_mean": float(np.mean(base_acc)),
        "selective_acc_50cov_mean": float(fr[:, i50].mean()),
        "selective_acc_50cov_std": float(fr[:, i50].std()),
    }
    (RES / "ukbb_validation.json").write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    mean, std = fr.mean(0), fr.std(0)
    full, gain = mean[-1], (mean[i50] - mean[-1]) * 100
    fig, ax = plt.subplots(figsize=(4.4, 3.3))
    x = GRID * 100
    ax.fill_between(x, mean - std, mean + std, color=UKBB_C, alpha=0.16, lw=0, label=r"$\pm1$ std")
    ax.plot(x, mean, color=UKBB_C, lw=2.5, zorder=4, label="Misfit-guided")
    ax.axhline(full, color=GRAY, ls=(0, (4, 3)), lw=1.2)
    ax.text(99, full, f"full coverage\n{full:.3f}", color="#6b7178", fontsize=7.4, ha="right", va="bottom")
    ax.scatter([50], [mean[i50]], s=60, color=ACCENT, zorder=6, edgecolor="white", lw=1.2)
    ax.annotate(f"+{gain:.1f} pp\nat 50%", xy=(50, mean[i50]), xytext=(57, mean[i50] - 0.06),
                fontsize=9, color=ACCENT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.3))
    ax.set_xlabel("Coverage retained (%)"); ax.set_ylabel("Accepted-set accuracy")
    ax.set_title(f"UK Biobank ({len(f):,} subjects, sex): abstaining on high\n"
                 "structural misfit raises accepted-set accuracy", fontsize=8.8, fontweight="bold")
    ax.set_xlim(10, 100)
    ax.legend(loc="lower left", fontsize=8, frameon=True)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(RES / "ukbb_selective.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {RES}/ukbb_validation.json + ukbb_selective.pdf", flush=True)


if __name__ == "__main__":
    main()
