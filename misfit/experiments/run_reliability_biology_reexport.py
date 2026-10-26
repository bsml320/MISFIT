"""AT-S18: re-export the reliability-biology supplement panels from the VERIFIED
canonical score, and recompute the clinical-scale associations.

The previous oasis_reliability_biology.pdf displayed the legacy PAC-Bayes certificate
(its group means match runs/clinical_insights/clinical_insights.json), and the main
text quoted that certificate's age/MMSE/CDR correlations as misfit correlations.
Here both panels are regenerated from results/per_subject_misfit_iop.npz
(canonical collect(), 5 seeds): misfit percentile by diagnostic group and by held-out
environment, and Spearman associations with age (ADNI, OASIS-1), MMSE and CDR (OASIS-1)
are recomputed against the actual misfit. CPU.
Outputs: paper/figures/{adni,oasis}_reliability_biology.pdf (300 ppi, Type-42)
         results/provenance/reliability_biology_recompute.json
"""
from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402

sys.path.insert(0, _resolve_path("<REPO_ROOT>"))

RES = Path(_resolve_path("<REPO_ROOT>/misfit/results"))
FIG = Path(_resolve_path("<REPO_ROOT>/misfit/paper/figures"))
plt.rcParams.update({"pdf.fonttype": 42, "font.size": 8.5})
TEAL, CORAL = "#0f6b74", "#c0392b"


def adni_age():
    """Ages and protocol-site ids in the exact order of the ADNI loader.

    Replays the run_adni_smri directory walk (protocol -> subject dirs with
    smwc1pT1.nii, CN/impaired groups, MIN_SUBJECTS_PER_SITE filter) so the
    returned arrays align row-for-row with the cached misfit vector.

    Returns:
        (ages, site_ids) numpy arrays; unparsable ages are NaN.
    """
    import pandas as pd
    from misfit.data.adni_loader import (
        IMPAIRED_GROUPS, LABELS_CSV, MIN_SUBJECTS_PER_SITE, NII_DIR, PROTOCOLS)
    df = pd.read_csv(LABELS_CSV)
    lut = {}
    for _, r in df.iterrows():
        lut.setdefault((str(r["dataset_id"]).strip(), str(r["file_name"]).strip()), r)
    ages, sites = [], []
    si = 0
    for protocol in PROTOCOLS:
        pdir = NII_DIR / protocol
        if not pdir.exists():
            continue
        rows = []
        for sdir in sorted(pdir.iterdir()):
            k = (protocol, sdir.name)
            if not (sdir / "smwc1pT1.nii").exists() or k not in lut:
                continue
            r = lut[k]
            if str(r["Group"]).strip() not in {"CN"} | IMPAIRED_GROUPS:
                continue
            rows.append(float(r["Age"]) if str(r["Age"]).replace(".", "").isdigit() else np.nan)
        if len(rows) >= MIN_SUBJECTS_PER_SITE:
            ages += rows
            sites += [si] * len(rows)
            si += 1
    return np.array(ages), np.array(sites)


def oasis_vars():
    """OASIS-1 age, CDR, MMSE and age-quartile environment, loader-aligned.

    Mirrors the OASIS-1 loader: valid-CDR subjects, age-quartile environments,
    environments with fewer than 10 subjects dropped.
    """
    from nilearn.datasets import fetch_oasis_vbm
    o = fetch_oasis_vbm()
    cdr = o.ext_vars["cdr"].values.astype(float)
    age = o.ext_vars["age"].values.astype(float)
    mmse = o.ext_vars["mmse"].values.astype(float)
    valid = ~np.isnan(cdr)
    age, cdr, mmse = age[valid], cdr[valid], mmse[valid]
    q = np.quantile(age, np.linspace(0, 1, 5))
    env = np.digitize(age, q[1:-1])
    counts = {e: int((env == e).sum()) for e in sorted(set(env))}
    keep = np.isin(env, [e for e, c in counts.items() if c >= 10])
    return age[keep], cdr[keep], mmse[keep], env[keep]


def violin(ax, groups, labels, colors, ylabel):
    """Violin bodies + jittered points + median bars for percentile groups."""
    parts = ax.violinplot(list(groups), showextrema=False, widths=0.8)
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c)
        pc.set_alpha(0.35)
    rng = np.random.RandomState(0)
    for i, (g, c) in enumerate(zip(groups, colors), 1):
        ax.scatter(i + rng.uniform(-0.13, 0.13, len(g)), g, s=3, color=c, alpha=0.4, lw=0)
        ax.hlines(np.median(g), i - 0.28, i + 0.28, color=c, lw=2.0, zorder=5)
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylim(-3, 103)
    ax.set_ylabel(ylabel, fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def sp(a, b):
    """Spearman rho, p and n over the finite pairs of (a, b)."""
    m = np.isfinite(a) & np.isfinite(b)
    r = spearmanr(a[m], b[m])
    return {"rho": round(float(r.statistic), 3), "p": float(f"{r.pvalue:.3g}"), "n": int(m.sum())}


def main():
    """Regenerate both panels and write the recompute provenance record."""
    d = np.load(RES / "per_subject_misfit_iop.npz")
    rec = {"score_identity": "cross-refit training loss (canonical collect(), 5-seed mean); "
                             "replaces legacy PAC-Bayes-certificate panels/associations"}

    # ---------------- ADNI panel
    ma, la = d["adni_misfit"], d["adni_label"]
    age_a, site_a = adni_age()
    assert len(age_a) == len(ma)
    pa = 100.0 * rankdata(ma) / len(ma)
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
    violin(ax[0], [pa[la == 0], pa[la == 1]],
           [f"CN (n={(la==0).sum()})", f"impaired (n={(la==1).sum()})"],
           [TEAL, CORAL], "misfit percentile")
    ax[0].set_title("ADNI: cross-refit training loss by group", fontsize=8.5)
    groups = [pa[site_a == e] for e in np.unique(site_a)]
    violin(ax[1], groups, [f"P{e}" for e in np.unique(site_a)],
           ["#5a8f9b"] * len(groups), "misfit percentile")
    ax[1].set_title("ADNI: by scanner protocol", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / "adni_reliability_biology.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    rec["adni"] = {"misfit_vs_age": sp(ma, age_a)}

    # ---------------- OASIS-1 panel
    mo, lo = d["oasis_misfit"], d["oasis_label"]
    age_o, cdr_o, mmse_o, env_o = oasis_vars()
    assert len(age_o) == len(mo)
    po = 100.0 * rankdata(mo) / len(mo)
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
    stages = [po[cdr_o == 0], po[cdr_o == 0.5], po[cdr_o >= 1]]
    violin(ax[0], stages, [f"CDR 0\n(n={len(stages[0])})", f"CDR 0.5\n(n={len(stages[1])})",
                           f"CDR$\\geq$1\n(n={len(stages[2])})"],
           [TEAL, "#b07fd6", CORAL], "misfit percentile")
    ax[0].set_title("OASIS-1: cross-refit training loss by CDR", fontsize=8.5)
    groups = [po[env_o == e] for e in np.unique(env_o)]
    violin(ax[1], groups, [f"Q{e+1}" for e in np.unique(env_o)],
           ["#5a8f9b"] * len(groups), "misfit percentile")
    ax[1].set_title("OASIS-1: by age quartile", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / "oasis_reliability_biology.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    rec["oasis1"] = {"misfit_vs_age": sp(mo, age_o), "misfit_vs_mmse": sp(mo, mmse_o),
                     "misfit_vs_cdr": sp(mo, cdr_o)}
    # Historical numerical values are archived separately from the source release.
    rec["legacy_certificate_values_replaced"] = {"status": "historical_values_not_bundled"}

    (RES / "provenance").mkdir(exist_ok=True)
    json.dump(rec, open(RES / "provenance" / "reliability_biology_recompute.json", "w"), indent=2)
    print(json.dumps(rec, indent=2))
    print("wrote both panels + reliability_biology_recompute.json")


if __name__ == "__main__":
    main()
