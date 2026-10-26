"""Subgroup robustness / equity of the reliability score: does structural misfit
detect failures equally well across demographic subgroups (sex, age), not only
on average? Computed from the cached seed-averaged per-subject misfit/error (no
model recompute) plus demographics aligned to the loader order. Failure-detection
AUROC per subgroup with bootstrap 95% CIs, both cohorts.

Writes paper/figures/subgroup_robustness.pdf and results/subgroup.json.
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
from sklearn.metrics import roc_auc_score  # noqa: E402

sys.path.insert(0, _resolve_path("<REPO_ROOT>"))
RES = Path(_resolve_path("<REPO_ROOT>/misfit/results"))
FIG = Path(_resolve_path("<REPO_ROOT>/misfit/paper/figures"))
sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
plt.rcParams.update({"pdf.fonttype": 42, "savefig.dpi": 300, "grid.color": "#dfe3e8"})
ADNI_C, OASIS_C, GRAY = "#2b6cb0", "#2f8f6b", "#9aa0a6"


def adni_demo():
    """ADNI labels, sex and age in the exact order of the ADNI loader.

    Replays run_adni_smri's directory walk and filters (smwc1pT1.nii present,
    CN/impaired groups, MIN_SUBJECTS_PER_SITE) so the returned arrays align
    row-for-row with the cached per-subject misfit arrays.
    """
    import pandas as pd
    from misfit.data.adni_loader import (
        IMPAIRED_GROUPS, LABELS_CSV, MIN_SUBJECTS_PER_SITE, NII_DIR, PROTOCOLS,
    )
    df = pd.read_csv(LABELS_CSV)
    lut = {}
    for _, r in df.iterrows():
        k = (str(r["dataset_id"]).strip(), str(r["file_name"]).strip())
        lut.setdefault(k, (str(r["Group"]).strip(), str(r["Sex"]).strip(), float(r["Age"])))
    labs, sex, age = [], [], []
    for protocol in PROTOCOLS:
        pdir = NII_DIR / protocol
        if not pdir.exists():
            continue
        pl, ps, pa = [], [], []
        for sdir in sorted(pdir.iterdir()):
            if not (sdir / "smwc1pT1.nii").exists():
                continue
            k = (protocol, sdir.name)
            if k not in lut or lut[k][0] not in {"CN"} | IMPAIRED_GROUPS:
                continue
            g, s, a = lut[k]
            pl.append(0 if g == "CN" else 1); ps.append(s); pa.append(a)
        if len(pl) < MIN_SUBJECTS_PER_SITE:
            continue
        labs += pl; sex += ps; age += pa
    return np.array(labs), np.array(sex), np.array(age, float)


def oasis_demo():
    """OASIS-1 labels (CDR > 0), sex codes and age, loader-aligned.

    Mirrors the OASIS-1 loader: valid-CDR subjects, age-quartile environments,
    environments with fewer than 10 subjects dropped.
    """
    from nilearn.datasets import fetch_oasis_vbm
    ev = fetch_oasis_vbm().ext_vars
    cdr = ev["cdr"].values.astype(float)
    age = ev["age"].values.astype(float)
    mf = np.array([str(x).strip().upper()[:1] for x in ev["mf"].values])
    valid = ~np.isnan(cdr)
    cdr, age, mf = cdr[valid], age[valid], mf[valid]
    q = np.quantile(age, np.linspace(0, 1, 5))
    env = np.digitize(age, q[1:-1])
    counts = {e: int((env == e).sum()) for e in sorted(set(env))}
    keep = np.isin(env, [e for e, c in counts.items() if c >= 10])
    return (cdr[keep] > 0).astype(int), mf[keep], age[keep]


def auroc_ci(score, err, n_boot=2000, seed=0):
    """AUROC with a bootstrap percentile 95% CI.

    Returns:
        (auroc, lo, hi); NaN triple when err contains a single class.
    """
    err = err.astype(int)
    if len(np.unique(err)) < 2:
        return np.nan, np.nan, np.nan
    a = roc_auc_score(err, score)
    rng = np.random.RandomState(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(err), len(err))
        if len(np.unique(err[idx])) < 2:
            continue
        boots.append(roc_auc_score(err[idx], score[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(a), float(lo), float(hi)


def subgroups(mis, err, sex, age):
    """Failure-detection AUROC with CI for All/Female/Male/Younger/Older."""
    med = np.median(age)
    defs = [("All", np.ones(len(err), bool)),
            ("Female", sex == "F"), ("Male", sex == "M"),
            ("Younger", age < med), ("Older", age >= med)]
    out = []
    for name, m in defs:
        a, lo, hi = auroc_ci(mis[m], err[m])
        out.append({"group": name, "n": int(m.sum()), "auroc": a, "lo": lo, "hi": hi})
        print(f"    {name:14s} n={int(m.sum()):4d}  AUROC={a:.3f} [{lo:.3f},{hi:.3f}]", flush=True)
    return out


def main():
    """Score subgroups on both cohorts, write JSON and the forest figure.

    The label-equality asserts bind the independently loaded demographics to
    the loader order of the cached misfit arrays before any subgroup is scored.
    """
    d = np.load(RES / "per_subject_misfit.npz")
    print("ADNI subgroups:", flush=True)
    la, sa, aa = adni_demo()
    assert len(la) == 731 and np.array_equal(la, d["adni_label"].astype(int)), "ADNI demo align broken"
    adni = subgroups(d["adni_misfit"], d["adni_err"], sa, aa)
    print("OASIS subgroups:", flush=True)
    lo_, mo, ao = oasis_demo()
    assert len(lo_) == 226 and np.array_equal(lo_, d["oasis_label"].astype(int)), "OASIS demo align broken"
    oasis = subgroups(d["oasis_misfit"], d["oasis_err"], mo, ao)
    json.dump({"adni": adni, "oasis": oasis}, open(RES / "subgroup.json", "w"), indent=2)

    # forest figure (single-column, stacked panels)
    fig, ax = plt.subplots(2, 1, figsize=(3.5, 4.6), sharex=True)
    for k, (res, color, title) in enumerate([(adni, ADNI_C, "ADNI (scanner shift)"),
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
    print("wrote subgroup_robustness.pdf + subgroup.json", flush=True)


if __name__ == "__main__":
    main()
