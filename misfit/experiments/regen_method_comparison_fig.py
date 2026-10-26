"""Regenerate the per-subject method-comparison figure (Fig. 1) with the honest
estimator values. Replaces stale shipped PDFs that still showed the pre-fix IOP
loss numbers. Reads results/honest_estimator_table.json; writes the two panel
PDFs (ADNI and OASIS-1) into paper/figures/. Vector PDF, colorblind-safe,
publication style. Run as a plain script with no arguments.
"""

from misfit.paths import resolve_path as _resolve_path
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(_resolve_path("<REPO_ROOT>/misfit"))
D = json.load(open(ROOT / "results/honest_estimator_table.json"))["estimator_table"]

# misfit vs independent uncertainty/complexity baselines (caption set)
ROWS = [
    ("Structural misfit", "structural_misfit"),
    ("ERM training loss", "erm_loss"),
    ("PAC-Bayes certificate", "pac_bayes_full"),
    ("Deep ensemble", "deep_ensemble5"),
    ("Max softmax", "max_softmax"),
    ("MC dropout", "mc_dropout"),
]
plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8, "pdf.fonttype": 42})


def panel(ds, fname, title):
    """Render one horizontal-bar panel (rho mean with std bars per method) for dataset ds."""
    labels = [r[0] for r in ROWS]
    means = [D[r[1]][ds]["mean"] for r in ROWS]
    stds = [D[r[1]][ds]["std"] for r in ROWS]
    y = range(len(ROWS))
    colors = ["#1b7837" if r[1] == "structural_misfit" else "#888888" for r in ROWS]
    fig, ax = plt.subplots(figsize=(3.2, 2.5))
    ax.barh(list(y), means, xerr=stds, color=colors, height=0.65,
            error_kw={"elinewidth": 0.8, "capsize": 2})
    ax.axvline(0, color="k", lw=0.6)
    ax.set_yticks(list(y)); ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(r"Spearman $\rho$ (misfit vs. per-subject error)")
    ax.set_title(title, fontsize=10)
    ax.set_xlim(min(-0.1, min(means) - 0.1), 0.7)
    ax.spines[["top", "right"]].set_visible(False)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(m + (0.02 if m >= 0 else -0.02) + (s if m >= 0 else -s),
                i, f"{m:.2f}", va="center",
                ha="left" if m >= 0 else "right", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(ROOT / "paper/figures" / fname, bbox_inches="tight")
    plt.close(fig)
    print("wrote", fname, [round(m, 3) for m in means])


panel("adni", "adni_subj_method_comparison.pdf", "ADNI (scanner-protocol shift)")
panel("oasis", "oasis_subj_method_comparison.pdf", "OASIS-1 (age shift)")
