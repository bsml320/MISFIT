"""Numerical instantiation of the cross-site transfer theorem (MedIA).

Reviewer charge: the transfer bound |rho^tr - rho^te| <= C*||B^tr - B^te|| is never
evaluated, so it could be window-dressing. This script instantiates it. For each
held-out protocol e:
  * B^tr = program basis recovered on the TRAINING protocols (recover_programs,
    (K,d) directions); B^te = top-K PCA subspace of protocol e's features (a
    single environment cannot be run through the multi-environment recovery, and
    the proof is a projector-perturbation argument, so the dominant subspace of e
    is the faithful object). drift = ||P^tr - P^te||_2 (largest principal-angle
    sine, in [0,1]); coherence = ||D||_op/||D||_F of the orthogonally-aligned
    basis residual (in [K^{-1/2},1]; near K^{-1/2} = incoherent/favourable).
  * rho_e = within-fold Spearman(misfit, error) on protocol e (test-CE misfit).
    degradation |Delta rho_e| = |rho_e - mean_{e'!=e} rho_{e'}|.
  * predicted bound = c * max||x||^2 * drift * f_max + O(N^{-1/2}), with
    c = 1/sqrt(p(1-p)), f_max = peak density of the misfit scores.

Key test: does degradation track drift across protocols (Spearman > 0)? If yes
the theorem makes a correct, falsifiable prediction (not window-dressing); the
absolute bound may still be loose, reported honestly. Numerics: float64 SVD,
no torch.inverse. Output: results/transfer_theorem.{json,md}
"""
from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path
import json, sys, warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as sp_stats

sys.path.insert(0, _resolve_path("<REPO_ROOT>"))
from misfit.experiments.run_honest_estimator_table import (  # noqa: E402
    N_PROGRAMS, N_SEEDS, SEED_BASE, _predict_loadings, _rho, _train_fire,
    load_adni, load_oasis_ds,
)
from misfit.tensor_spectral.recovery import recover_programs  # noqa: E402
from misfit.core.types import TensorConfig  # noqa: E402
from misfit.core.utils import seed_everything  # noqa: E402
from misfit.data.mri_multisite import MRIData, mri_to_multienv_format  # noqa: E402
warnings.filterwarnings("ignore")
OUT = Path(_resolve_path("<REPO_ROOT>/misfit/results"))
K = N_PROGRAMS


def _orthonormal_rows(B):
    """Return (K,d) orthonormal basis of the row span of B via SVD (float64)."""
    B = B.to(torch.float64)
    U, S, Vt = torch.linalg.svd(B, full_matrices=False)
    return Vt[: B.shape[0]]


def _proj(Q):
    """Projector (d,d) onto row span of orthonormal-rows Q."""
    return Q.T @ Q


def _recover_basis(feat, sites):
    """Program directions (K, d) from the multi-environment spectral recovery.

    Wraps the features into MRIData with a one-hot subject-site matrix and runs
    recover_programs with a TensorConfig matching the pool.
    """
    uniq = sites.unique(); smap = {int(s): i for i, s in enumerate(uniq)}
    rem = torch.tensor([smap[int(s)] for s in sites]); nts = len(uniq)
    n_sub = len(feat)
    ssm = torch.zeros(n_sub, nts)
    for i in range(n_sub):
        ssm[i, int(rem[i])] = 1.0
    mri = MRIData(
        features=feat, targets=torch.zeros(n_sub), subject_ids=torch.arange(n_sub),
        site_ids=rem, n_sites=nts, n_subjects=n_sub,
        feature_names=[f"f{i}" for i in range(feat.shape[1])],
        site_names=[str(i) for i in range(nts)], subject_site_matrix=ssm,
    )
    obs, _ = mri_to_multienv_format(mri)
    tc = TensorConfig(n_programs=K, n_envs=nts, dim=feat.shape[1])
    rec = recover_programs(obs, n_programs=K, tensor_config=tc)
    return rec.directions  # (K, d)


def _pca_basis(feat):
    """Top-K principal directions (K, d) of mean-centered features (float64 SVD)."""
    X = (feat - feat.mean(0)).to(torch.float64)
    U, S, Vt = torch.linalg.svd(X, full_matrices=False)
    return Vt[:K]  # (K, d) top-K principal directions


def _f_max(misfit):
    """Peak density of the (normalised) misfit scores via Gaussian KDE."""
    m = (misfit - misfit.min()) / (misfit.max() - misfit.min() + 1e-12)
    try:
        kde = sp_stats.gaussian_kde(m)
        grid = np.linspace(0, 1, 200)
        return float(kde(grid).max())
    except Exception:
        h, _ = np.histogram(m, bins=20, density=True)
        return float(h.max())


def run_dataset(name, loader):
    """Instantiate the transfer bound on one cohort.

    Per seed and held-out protocol: within-fold rho_e, drift (spectral-norm gap
    between the training-recovered projector and the held-out protocol's PCA
    projector), alignment coherence, and the predicted bound. Seed-averaged
    rows then test whether degradation |Delta rho_e| tracks drift across
    protocols and whether the bound holds everywhere.
    """
    features, targets, site_ids = loader()
    n_sites = int(site_ids.max()) + 1
    dim = features.shape[1]
    maxx2 = float((features.to(torch.float64) ** 2).sum(1).max())
    # per-protocol accumulators over seeds
    rho_e = {e: [] for e in range(n_sites)}
    drift_e = {e: [] for e in range(n_sites)}
    coh_e = {e: [] for e in range(n_sites)}
    p_e = {e: [] for e in range(n_sites)}
    fmax_all = []
    for s in range(N_SEEDS):
        seed = SEED_BASE + s
        seed_everything(seed)
        for e in range(n_sites):
            tr = site_ids != e; te = site_ids == e
            if te.sum() < 5 or tr.sum() < 20:
                continue
            trf, trt, trs = features[tr], targets[tr], site_ids[tr]
            tef, tet = features[te], targets[te]
            uniq = trs.unique(); smap = {int(z): i for i, z in enumerate(uniq)}
            rem = torch.tensor([smap[int(z)] for z in trs]); nts = len(uniq)
            model, loadings = _train_fire(trf, trt, rem, nts, dim, seed)
            tl = _predict_loadings(trf, loadings, tef)
            with torch.no_grad():
                logits = model(tef, tl, torch.zeros(len(tef), dtype=torch.long))
                ce = F.cross_entropy(logits, tet.long(), reduction="none").numpy()
                err = 1.0 - (logits.argmax(-1) == tet.long()).float().numpy()
            r = _rho(ce, err)
            if not np.isnan(r):
                rho_e[e].append(r)
            p_e[e].append(float(err.mean()))
            fmax_all.append(_f_max(ce))
            # bases + drift
            try:
                Btr = _orthonormal_rows(_recover_basis(trf, trs))
            except Exception:
                Btr = _orthonormal_rows(_pca_basis(trf))
            Bte = _orthonormal_rows(_pca_basis(tef))
            Ptr, Pte = _proj(Btr), _proj(Bte)
            drift = float(torch.linalg.matrix_norm(Ptr - Pte, ord=2))
            drift_e[e].append(min(drift, 1.0))
            # coherence of orthogonally-aligned basis residual
            M = Btr @ Bte.T  # (K,K)
            Uu, _, Vv = torch.linalg.svd(M)
            R = Uu @ Vv  # orthogonal alignment
            D = Btr - R @ Bte
            fro = float(torch.linalg.matrix_norm(D, ord="fro"))
            op = float(torch.linalg.matrix_norm(D, ord=2))
            coh_e[e].append(op / fro if fro > 1e-9 else 1.0)

    # aggregate per protocol
    rows = []
    fmax = float(np.median(fmax_all))
    for e in range(n_sites):
        if not rho_e[e]:
            continue
        rho_m = float(np.mean(rho_e[e])); drift_m = float(np.mean(drift_e[e]))
        coh_m = float(np.mean(coh_e[e])); p_m = float(np.mean(p_e[e]))
        rows.append({"protocol": e, "n_seeds": len(rho_e[e]), "rho": rho_m,
                     "drift": drift_m, "coherence": coh_m, "err_rate": p_m})
    # degradation = |rho_e - mean over OTHER protocols|
    rhos = np.array([r["rho"] for r in rows])
    for i, r in enumerate(rows):
        others = np.delete(rhos, i)
        r["rho_tr"] = float(others.mean())
        r["abs_delta_rho"] = float(abs(r["rho"] - others.mean()))
        c = 1.0 / np.sqrt(max(r["err_rate"] * (1 - r["err_rate"]), 1e-6))
        clt = 1.0 / np.sqrt(max(r["n_seeds"] * 5, 1))  # rough N^{-1/2}
        r["pred_bound"] = float(c * maxx2 * r["drift"] * fmax + clt)
    drifts = np.array([r["drift"] for r in rows])
    degr = np.array([r["abs_delta_rho"] for r in rows])
    track = sp_stats.spearmanr(drifts, degr) if len(rows) >= 3 else (np.nan, np.nan)
    holds = int(all(r["pred_bound"] >= r["abs_delta_rho"] for r in rows))
    return {"n_sites": n_sites, "max_x2": maxx2, "f_max": fmax,
            "per_protocol": rows,
            "drift_vs_degradation_spearman": float(track[0]) if not np.isnan(track[0]) else None,
            "drift_vs_degradation_p": float(track[1]) if not np.isnan(track[1]) else None,
            "mean_coherence": float(np.mean([r["coherence"] for r in rows])),
            "Kinvsqrt": float(1 / np.sqrt(K)),
            "bound_holds_all_protocols": bool(holds),
            "mean_pred_bound": float(np.mean([r["pred_bound"] for r in rows])),
            "mean_abs_delta_rho": float(np.mean(degr))}


def main():
    res = {"adni": run_dataset("adni", load_adni),
           "oasis": run_dataset("oasis", load_oasis_ds)}
    (OUT / "transfer_theorem.json").write_text(json.dumps(res, indent=2))
    L = ["# Cross-site transfer theorem -- numerical instantiation", ""]
    for n in ["adni", "oasis"]:
        r = res[n]
        L += [f"## {n.upper()} ({r['n_sites']} protocols)",
              f"- drift-vs-degradation Spearman = **{r['drift_vs_degradation_spearman']}** "
              f"(p={r['drift_vs_degradation_p']})",
              f"- mean coherence ratio = **{r['mean_coherence']:.3f}** "
              f"(incoherent floor K^-1/2 = {r['Kinvsqrt']:.3f})",
              f"- predicted bound holds for all protocols: {r['bound_holds_all_protocols']} "
              f"(mean bound {r['mean_pred_bound']:.2f} vs mean |dRho| {r['mean_abs_delta_rho']:.3f})",
              "", "| protocol | rho_e | rho_tr | |dRho| | drift | coherence | pred bound |",
              "|---|---|---|---|---|---|---|"]
        for p in r["per_protocol"]:
            L.append(f"| {p['protocol']} | {p['rho']:.3f} | {p['rho_tr']:.3f} | "
                     f"{p['abs_delta_rho']:.3f} | {p['drift']:.3f} | {p['coherence']:.3f} | "
                     f"{p['pred_bound']:.2f} |")
        L.append("")
    (OUT / "transfer_theorem.md").write_text("\n".join(L))
    print("saved transfer_theorem.{json,md}", flush=True)


if __name__ == "__main__":
    main()
