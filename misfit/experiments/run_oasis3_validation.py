"""OASIS-3 disease-matched external validation (MedIA): does structural misfit rank per-subject
AD-classification error under leave-one-SCANNER-out shift, on a 4x-ADNI-size independent cohort?

Uses the same validated IOP collector as ADNI/OASIS-1/ABIDE/UKBB. Environment = scanner model.
Features: route-B SPM-DARTEL VBM smwc1 -> PCA-50 (oasis3_features.npz).
Reports AUROC, pooled + within-scanner rho, selective prediction, per-scanner breakdown, and a
SUBJECT-LEAKAGE diagnostic (OASIS-3 is longitudinal; some subjects appear on multiple scanners).
GPU 2. 5 seeds. Output: results/oasis3/oasis3_validation.json + oasis3_selective.pdf.
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
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

sys.path.insert(0, _resolve_path("<REPO_ROOT>"))
from misfit.experiments.run_clinical_stratified_media import collect  # noqa: E402

NPZ = os.environ.get("OASIS3_NPZ", _resolve_path("<REPO_ROOT>/data/oasis3/oasis3_features.npz"))
FEATSET = os.environ.get("OASIS3_FEATSET", "vbm")   # 'vbm' or 'freesurfer' -> output filename
RES = Path(_resolve_path("<REPO_ROOT>/misfit/results/oasis3"))
N_SEEDS, SEED_BASE = 5, 42
GRID = np.arange(0.10, 1.001, 0.02)
sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
plt.rcParams.update({"pdf.fonttype": 42, "savefig.dpi": 300})
C, ACCENT, GRAY = "#6a1b9a", "#c0392b", "#9aa0a6"


def load(unique: bool = False):
    """Load OASIS-3 features; unique=True keeps one earliest session per subject.

    Returns:
        (features, targets, site_ids, scanner name per row, subject id per row,
        scanner names ordered by descending frequency).
    """
    d = np.load(NPZ, allow_pickle=True)
    feats = d["features"]; labels = d["label"].astype(int)
    scan = d["scanner"].astype(str); subj = d["subjects"].astype(str)
    sess = d["sessions"].astype(str)
    if unique:
        # one session per subject (earliest day) -> removes cross-scanner longitudinal leakage
        import re
        day = np.array([int(re.search(r"_d(\d+)$", s).group(1)) for s in sess])
        keep_idx = []
        for u in np.unique(subj):
            idx = np.where(subj == u)[0]
            keep_idx.append(idx[np.argmin(day[idx])])
        keep = np.array(sorted(keep_idx))
        feats, labels, scan, subj, sess = feats[keep], labels[keep], scan[keep], subj[keep], sess[keep]
    f = torch.tensor(feats, dtype=torch.float32)
    t = torch.tensor(labels, dtype=torch.long)
    scanners = scan
    subjects = subj
    # scanner -> contiguous int codes, ordered by frequency (stable)
    uniq, counts = np.unique(scanners, return_counts=True)
    order = uniq[np.argsort(-counts)]
    code = {s: i for i, s in enumerate(order)}
    site = torch.tensor([code[s] for s in scanners], dtype=torch.long)
    return f, t, site, scanners, subjects, order


def within_env_rho(mis, err, site, tested):
    """Mean within-scanner Spearman rho over scoreable scanners.

    A scanner is scored if it has at least 8 tested subjects and a
    non-degenerate error vector.

    Returns:
        (mean rho, fraction of scanners with rho > 0, n scored).
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
    """Run the collector over seeds; write summary JSON and the selective
    figure. Pass 'unique' on the command line for the leakage-free cohort."""
    unique = "unique" in sys.argv
    tag = "_unique" if unique else ""
    base = f"oasis3_validation{'' if FEATSET=='vbm' else '_'+FEATSET}{tag}"
    RES.mkdir(parents=True, exist_ok=True)
    f, t, site, scanners, subjects, order = load(unique=unique)
    if unique:
        print("[MODE] one session per subject (leakage-free leave-one-scanner-out)", flush=True)
    sid = site.numpy()
    # subject-leakage diagnostic: for each scanner, fraction of its subjects also scanned elsewhere
    leak = []
    for e in np.unique(sid):
        te_subj = set(subjects[sid == e]); tr_subj = set(subjects[sid != e])
        leak.append(len(te_subj & tr_subj) / max(1, len(te_subj)))
    print(f"OASIS-3: N={len(f)} dim={f.shape[1]} scanners={len(order)} "
          f"impaired={int(t.sum())}/{len(t)} ({100*t.float().mean():.1f}%)", flush=True)
    print(f"scanners (by n): {list(order)}", flush=True)
    print(f"subject-leakage across scanners (mean frac of held-out subjects also in train): "
          f"{np.mean(leak):.3f}", flush=True)

    aurocs, pooled, we_rho, we_frac, frontiers, base_acc = [], [], [], [], [], []
    for seed in range(N_SEEDS):
        c = collect(f, t, site, SEED_BASE + seed)
        m = c["tested"]
        mis, err, st = c["misfit"], c["err"], c["site"]
        mt, et = mis[m], err[m]
        aurocs.append(roc_auc_score(et, mt))
        pooled.append(spearmanr(mt, et)[0])
        r, frac, n = within_env_rho(mis, err, st, m)
        we_rho.append(r); we_frac.append(frac)
        o = np.argsort(mt)
        frontiers.append([1.0 - et[o[:max(1, int(round(cov * len(mt))))]].mean() for cov in GRID])
        base_acc.append(1.0 - et.mean())
        print(f"  seed {seed}: AUROC={aurocs[-1]:.3f} pooled_rho={pooled[-1]:.3f} "
              f"within-scanner_rho={r:.3f} ({frac*100:.0f}% of {n} scanners +) "
              f"base_acc={base_acc[-1]:.3f}", flush=True)

    fr = np.array(frontiers); i50 = int(np.argmin(np.abs(GRID - 0.5)))
    summary = {
        "n": int(len(f)), "n_scanners": int(len(order)), "dim": int(f.shape[1]),
        "task": "AD_vs_control_CDR", "shift_axis": "scanner_model",
        "impaired_frac": float(t.float().mean()), "subject_leakage_mean": float(np.mean(leak)),
        "auroc_mean": float(np.mean(aurocs)), "auroc_std": float(np.std(aurocs)),
        "pooled_rho_mean": float(np.mean(pooled)), "pooled_rho_std": float(np.std(pooled)),
        "within_scanner_rho_mean": float(np.mean(we_rho)), "within_scanner_rho_std": float(np.std(we_rho)),
        "within_scanner_frac_positive": float(np.mean(we_frac)),
        "base_acc_mean": float(np.mean(base_acc)),
        "selective_acc_50cov_mean": float(fr[:, i50].mean()),
        "selective_acc_50cov_std": float(fr[:, i50].std()),
        "scanners": list(order),
    }
    (RES / f"{base}.json").write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    mean, std = fr.mean(0), fr.std(0)
    full, gain = mean[-1], (mean[i50] - mean[-1]) * 100
    fig, ax = plt.subplots(figsize=(4.4, 3.3)); x = GRID * 100
    ax.fill_between(x, mean - std, mean + std, color=C, alpha=0.16, lw=0, label=r"$\pm1$ std")
    ax.plot(x, mean, color=C, lw=2.5, zorder=4, label="Misfit-guided")
    ax.axhline(full, color=GRAY, ls=(0, (4, 3)), lw=1.2)
    ax.text(99, full, f"full coverage\n{full:.3f}", color="#6b7178", fontsize=7.4, ha="right", va="bottom")
    ax.scatter([50], [mean[i50]], s=60, color=ACCENT, zorder=6, edgecolor="white", lw=1.2)
    ax.annotate(f"+{gain:.1f} pp\nat 50%", xy=(50, mean[i50]), xytext=(57, mean[i50] - 0.06),
                fontsize=9, color=ACCENT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.3))
    ax.set_xlabel("Coverage retained (%)"); ax.set_ylabel("Accepted-set accuracy")
    ax.set_title(f"OASIS-3 ({len(f):,} earliest-session participants; six scanner\n"
                 "models, five scored): retained-set accuracy vs. coverage", fontsize=8.8, fontweight="bold")
    ax.set_xlim(10, 100); ax.legend(loc="lower left", fontsize=8, frameon=True)
    sns.despine(ax=ax); fig.tight_layout()
    fig.savefig(RES / f"{base.replace('validation','selective')}.pdf", bbox_inches="tight"); plt.close(fig)
    print(f"wrote {RES}/{base}.json (+ selective pdf)", flush=True)


if __name__ == "__main__":
    main()
