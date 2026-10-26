"""Brain-network-level functional-connectivity signature of the reliability score
on ABIDE-I, visualized as a chord diagram.

Using the AAL atlas grouped into 7 anatomical systems (frontal, temporal, parietal,
occipital, limbic, subcortical, cerebellum), the script computes each subject's
network-to-network functional connectivity and asks which connections characterize
the subjects the model finds unreliable: for each network pair, the mean (Fisher-z)
connectivity is correlated with the per-subject structural misfit across subjects.
The misfit is recomputed on this atlas with the same leave-one-site-out harness,
and its within-site rho is reported as a sanity check that the reliability signal
holds here.

Writes results/abide_network.json and paper/figures/abide_network_chord.pdf. GPU 2.
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
import torch  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, _resolve_path("<REPO_ROOT>"))
from misfit.experiments.run_clinical_stratified_media import collect  # noqa: E402

RES = Path(_resolve_path("<REPO_ROOT>/misfit/results"))
FIG = Path(_resolve_path("<REPO_ROOT>/misfit/paper/figures"))
N_SEEDS, SEED_BASE, MIN_SITE = 2, 42, 10
NET_ORDER = ["Frontal", "Temporal", "Parietal", "Occipital",
             "Limbic", "Subcortical", "Cerebellum"]
NET_SHORT = {n: n for n in NET_ORDER}



def aal_to_system(label):
    """Map an AAL region name to one of the 7 anatomical systems (Frontal is the fallback)."""
    n = label.lower()
    if "cerebel" in n or "vermis" in n:
        return "Cerebellum"
    if any(k in n for k in ["caudate", "putamen", "pallidum", "thalamus"]):
        return "Subcortical"
    if any(k in n for k in ["cingul", "hippocamp", "amygdala", "parahippocampal"]):
        return "Limbic"
    if any(k in n for k in ["occipital", "calcarine", "cuneus", "lingual", "fusiform"]):
        return "Occipital"
    if any(k in n for k in ["parietal", "postcentral", "supramarginal", "angular", "precuneus"]):
        return "Parietal"
    if any(k in n for k in ["temporal", "heschl"]):
        return "Temporal"
    return "Frontal"


def aal116_labels():
    """Standard AAL1 116-region names in label-value order (the C-PAC ROI order)."""
    cer = ['Precentral','Frontal_Sup','Frontal_Sup_Orb','Frontal_Mid','Frontal_Mid_Orb',
           'Frontal_Inf_Oper','Frontal_Inf_Tri','Frontal_Inf_Orb','Rolandic_Oper','Supp_Motor_Area',
           'Olfactory','Frontal_Sup_Medial','Frontal_Med_Orb','Rectus','Insula',
           'Cingulum_Ant','Cingulum_Mid','Cingulum_Post','Hippocampus','ParaHippocampal',
           'Amygdala','Calcarine','Cuneus','Lingual','Occipital_Sup','Occipital_Mid','Occipital_Inf',
           'Fusiform','Postcentral','Parietal_Sup','Parietal_Inf','SupraMarginal','Angular','Precuneus',
           'Paracentral_Lobule','Caudate','Putamen','Pallidum','Thalamus','Heschl',
           'Temporal_Sup','Temporal_Pole_Sup','Temporal_Mid','Temporal_Pole_Mid','Temporal_Inf']
    cb = ['Cerebelum_Crus1','Cerebelum_Crus2','Cerebelum_3','Cerebelum_4_5','Cerebelum_6',
          'Cerebelum_7b','Cerebelum_8','Cerebelum_9','Cerebelum_10']
    verm = ['Vermis_1_2','Vermis_3','Vermis_4_5','Vermis_6','Vermis_7','Vermis_8','Vermis_9','Vermis_10']
    labs = []
    for t in cer: labs += [t+'_L', t+'_R']
    for t in cb:  labs += [t+'_L', t+'_R']
    labs += verm
    return labs


def load_aligned():
    """AAL ABIDE: aligned per-subject mean features (for misfit), FC, target, site;
    each ROI mapped to an anatomical system by AAL region name."""
    from nilearn.datasets import fetch_abide_pcp
    ab = fetch_abide_pcp(data_dir=_resolve_path("<REPO_ROOT>/data/abide"),
                         derivatives=["rois_aal"], pipeline="cpac",
                         quality_checked=True, verbose=0)
    ts_list = ab.rois_aal
    pheno = ab.phenotypic
    labels = aal116_labels()
    assert len(labels) == 116
    nets = np.array([aal_to_system(l) for l in labels])
    assert "Other" not in nets, "unmapped AAL region"

    means, fcs, idx = [], [], []
    for i, ts in enumerate(ts_list):
        if ts is None or len(ts) < 30 or ts.shape[1] != len(nets):
            continue
        m = ts.mean(axis=0)
        if not np.isfinite(m).all():
            continue
        std = ts.std(axis=0)
        tc = ts.copy()
        tc[:, std < 1e-10] = np.random.RandomState(0).randn(ts.shape[0], int((std < 1e-10).sum())) * 1e-6
        fc = np.corrcoef(tc.T)
        if not np.isfinite(fc).all():
            continue
        feat = fc[np.triu_indices(fc.shape[0], 1)]  # connectivity features
        means.append(feat); fcs.append(fc); idx.append(i)
    pv = pheno.iloc[idx].reset_index(drop=True)
    site = pv["SITE_ID"].values.astype(str)
    targ = (pv["DX_GROUP"].values == 1).astype(np.float32)
    means = np.stack(means); fcs = np.stack(fcs)
    # site filter
    counts = {s: int((site == s).sum()) for s in set(site)}
    keep = np.array([counts[s] >= MIN_SITE for s in site])
    means, fcs, site, targ = means[keep], fcs[keep], site[keep], targ[keep]
    senc = {s: k for k, s in enumerate(sorted(set(site)))}
    sid = np.array([senc[s] for s in site])
    return means, fcs, targ, sid, nets


def misfit_dosenbach(means, targ, sid):
    """Per-site z-norm mean features -> collect() misfit over seeds; within-site rho sanity."""
    from sklearn.decomposition import PCA
    X = PCA(n_components=min(150, means.shape[0] - 1, means.shape[1]),
            random_state=0).fit_transform(means)
    f = torch.tensor(X, dtype=torch.float32)
    s = torch.tensor(sid, dtype=torch.long)
    for k in range(int(s.max()) + 1):
        m = s == k
        if m.sum() > 1:
            f[m] = (f[m] - f[m].mean(0)) / f[m].std(0).clamp(min=1e-8)
    t = torch.tensor(targ, dtype=torch.float32)
    per = []
    ws = []
    for seed in range(N_SEEDS):
        c = collect(f, t, s, SEED_BASE + seed)
        per.append(c["misfit"])
        msk = c["tested"]
        rs = []
        for site_i in np.unique(c["site"][msk]):
            mm = msk & (c["site"] == site_i)
            e = c["err"][mm]
            if len(e) >= 8 and 0 < e.sum() < len(e):
                r = spearmanr(c["misfit"][mm], e)[0]
                if np.isfinite(r):
                    rs.append(r)
        ws.append(np.mean(rs))
    return np.mean(per, 0), float(np.mean(ws))


def network_fc(fcs, nets):
    """Per-subject 6x6 network FC (mean Fisher-z connectivity), networks in NET_ORDER."""
    groups = [np.where(nets == n)[0] for n in NET_ORDER]
    z = np.arctanh(np.clip(fcs, -0.999, 0.999))
    K = len(NET_ORDER)
    out = np.zeros((len(fcs), K, K))
    for a in range(K):
        for b in range(a, K):
            blk = z[:, groups[a]][:, :, groups[b]]  # (N, |a|, |b|)
            if a == b:
                iu = np.triu_indices(len(groups[a]), 1)
                val = blk[:, iu[0], iu[1]].mean(1)
            else:
                val = blk.reshape(len(fcs), -1).mean(1)
            out[:, a, b] = out[:, b, a] = val
    return out


def assoc(netfc, misfit):
    """Symmetric matrix of Spearman rho between each network-pair FC and misfit."""
    K = netfc.shape[1]
    A = np.zeros((K, K))
    for a in range(K):
        for b in range(a, K):
            A[a, b] = A[b, a] = spearmanr(netfc[:, a, b], misfit)[0]
    return A


def chord(A, node_within, fname):
    """Draw the chord diagram: edges are between-network associations, nodes within-network.

    Values are z-scored jointly over edges and nodes for the diverging colormap;
    edge width and alpha scale with the standardized magnitude.
    """
    K = len(NET_ORDER)
    ang = np.linspace(0, 2 * np.pi, K, endpoint=False) + np.pi / 2
    pos = np.c_[np.cos(ang), np.sin(ang)]
    off = np.triu_indices(K, 1)
    # standardize (z-score) across all system connections so the diverging
    # colormap reveals which connections stand out; raw associations are
    # predominantly positive (see caption and abide_network.json).
    vals = np.concatenate([A[off], node_within])
    mu, sd = vals.mean(), vals.std() + 1e-12
    Az = (A - mu) / sd
    nz = (node_within - mu) / sd
    vmax = max(np.abs(Az[off]).max(), np.abs(nz).max())
    norm = Normalize(-vmax, vmax)
    cmap = plt.get_cmap("RdBu_r")
    plt.rcParams.update({"pdf.fonttype": 42, "savefig.dpi": 300})
    fig, ax = plt.subplots(figsize=(6.0, 5.8))
    for a in range(K):
        for b in range(a + 1, K):
            v = Az[a, b]
            verts = [pos[a], [0, 0], pos[b]]
            ax.add_patch(PathPatch(MPath(verts, [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3]),
                                   fc="none", ec=cmap(norm(v)), lw=0.5 + 4.8 * abs(v) / vmax,
                                   alpha=0.40 + 0.55 * abs(v) / vmax, zorder=2))
    for i in range(K):
        ax.scatter(*pos[i], s=640, c=[nz[i]], cmap=cmap, norm=norm,
                   edgecolor="#333", lw=1.3, zorder=5)
        lx, ly = pos[i] * 1.30
        ha = "left" if lx > 0.15 else ("right" if lx < -0.15 else "center")
        ax.annotate(NET_SHORT[NET_ORDER[i]], (lx, ly), ha=ha, va="center",
                    fontsize=10, fontweight="bold")
    ax.set_xlim(-1.85, 1.85); ax.set_ylim(-1.5, 1.6); ax.axis("off"); ax.set_aspect("equal")
    ax.set_title("Network connectivity signature of the reliability score\n(ABIDE-I, AAL 7 systems; standardized corr of connectivity with misfit)",
                 fontsize=9.2, fontweight="bold")
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("Standardized association (z): connectivity vs. misfit", fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)
    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight", dpi=300, pad_inches=0.25)
    plt.close(fig)

def main():
    """Load data, compute (or reuse cached) misfit, build association maps, render chord."""
    print("loading AAL ...", flush=True)
    means, fcs, targ, sid, nets = load_aligned()
    print(f"aligned: N={len(means)} ROIs={fcs.shape[1]} sites={int(sid.max())+1} "
          f"balance={targ.mean():.3f}", flush=True)
    print("computing misfit ...", flush=True)
    cache = RES / "abide_network_misfit.npz"
    if cache.exists():
        z = np.load(cache); misfit, ws_rho = z["misfit"], float(z["ws_rho"])
    else:
        misfit, ws_rho = misfit_dosenbach(means, targ, sid)
        np.savez(cache, misfit=misfit, ws_rho=ws_rho)
    print(f"within-site rho (AAL) = {ws_rho:.3f}", flush=True)
    netfc = network_fc(fcs, nets)
    A = assoc(netfc, misfit)
    # split-half stability of the association map
    rng = np.random.RandomState(0); perm = rng.permutation(len(misfit)); h = len(misfit) // 2
    A1 = assoc(netfc[perm[:h]], misfit[perm[:h]]); A2 = assoc(netfc[perm[h:]], misfit[perm[h:]])
    off = np.triu_indices(len(NET_ORDER), 1)
    stab = float(np.corrcoef(A1[off], A2[off])[0, 1])
    within = np.array([A[i, i] for i in range(len(NET_ORDER))])

    pairs = [(NET_SHORT[NET_ORDER[a]] + "-" + NET_SHORT[NET_ORDER[b]], float(A[a, b]))
             for a in range(len(NET_ORDER)) for b in range(a + 1, len(NET_ORDER))]
    pairs.sort(key=lambda x: -abs(x[1]))
    summary = {"n": int(len(means)), "n_sites": int(sid.max()) + 1, "atlas": "AAL-7systems",
               "within_site_rho_misfit": ws_rho, "between_pair_stability_split_half": stab,
               "top_between_pairs": pairs[:6],
               "within_network": {NET_SHORT[NET_ORDER[i]]: float(within[i]) for i in range(len(NET_ORDER))}}
    json.dump(summary, open(RES / "abide_network.json", "w"), indent=2)
    print(json.dumps(summary, indent=1), flush=True)
    chord(A, within, FIG / "abide_network_chord.pdf")
    print("wrote abide_network_chord.pdf + abide_network.json", flush=True)


if __name__ == "__main__":
    main()
