"""Rescue attempt for the cross-site transfer bound (MedIA).

The first instantiation used B^te = PCA of the held-out protocol, which saturated
(drift=1 everywhere on ADNI) because the tensor-recovered program subspace is not
the data's PCA subspace. Here two non-degenerate, proof-faithful drift measures
are used, both built from the multi-environment recovery (no single-protocol
recovery needed):

  drift_recon[e] = mean_{i in e} ||x_i - P^{(-e)} x_i||^2 / ||x_i||^2
      reconstruction residual of protocol e under the program subspace recovered
      on the OTHER protocols. This is exactly the structural-reconstruction
      quantity the proof bounds (Step 2, L_i(B)=||x-P_B x||^2). Larger = protocol
      e sits further outside the training program subspace.

  drift_leverage[e] = ||P^{(all)} - P^{(-e)}||_2
      how much the recovered program subspace shifts when protocol e is added.
      Larger = e is more structurally distinctive.

Targets: per-protocol within-fold rho_e (test-CE misfit vs error) and the
degradation |Delta rho_e| = |rho_e - mean_{e'!=e} rho_{e'}|. The script tests
whether either drift measure tracks rho_e and |Delta rho_e| (Spearman across
protocols) and reports the per-protocol error rate so the class-composition
confound is visible. Numerics: float64 SVD, no torch.inverse.
Output: results/transfer_rescue.{json,md}
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
from misfit.experiments.run_transfer_theorem_check import (  # noqa: E402
    _orthonormal_rows, _proj, _recover_basis,
)
from misfit.core.utils import seed_everything  # noqa: E402
warnings.filterwarnings("ignore")
OUT = Path(_resolve_path("<REPO_ROOT>/misfit/results"))
K = N_PROGRAMS


def run_dataset(name, loader):
    """Per-protocol drift and reliability on one cohort.

    Per seed and held-out protocol e: within-fold rho_e (test-CE misfit vs
    error), reconstruction-residual drift under the source-recovered program
    subspace, and projector-leverage drift vs the full-pool basis. Seed-averaged
    per-protocol rows are then rank-correlated (each drift measure vs rho_e and
    |Delta rho_e|).
    """
    features, targets, site_ids = loader()
    n_sites = int(site_ids.max()) + 1
    dim = features.shape[1]
    rho_e = {e: [] for e in range(n_sites)}
    recon_e = {e: [] for e in range(n_sites)}
    lev_e = {e: [] for e in range(n_sites)}
    p_e = {e: [] for e in range(n_sites)}
    for s in range(N_SEEDS):
        seed = SEED_BASE + s
        seed_everything(seed)
        # full-pool basis (once per seed)
        try:
            P_all = _proj(_orthonormal_rows(_recover_basis(features, site_ids)))
        except Exception:
            P_all = None
        for e in range(n_sites):
            tr = site_ids != e; te = site_ids == e
            if te.sum() < 5 or tr.sum() < 20:
                continue
            trf, trt, trs = features[tr], targets[tr], site_ids[tr]
            tef, tet = features[te], targets[te]
            uniq = trs.unique(); smap = {int(z): i for i, z in enumerate(uniq)}
            rem = torch.tensor([smap[int(z)] for z in trs]); nts = len(uniq)
            # within-fold reliability rho_e
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
            # training-pool basis (e excluded)
            try:
                Q = _orthonormal_rows(_recover_basis(trf, trs))
            except Exception:
                continue
            P = _proj(Q)  # (d,d)
            X = tef.to(torch.float64)
            proj = X @ P
            resid = ((X - proj) ** 2).sum(1) / ((X ** 2).sum(1) + 1e-12)
            recon_e[e].append(float(resid.mean()))
            if P_all is not None:
                lev_e[e].append(float(torch.linalg.matrix_norm(P_all - P, ord=2)))

    rows = []
    for e in range(n_sites):
        if not rho_e[e]:
            continue
        rows.append({"protocol": e, "rho": float(np.mean(rho_e[e])),
                     "drift_recon": float(np.mean(recon_e[e])) if recon_e[e] else None,
                     "drift_leverage": float(np.mean(lev_e[e])) if lev_e[e] else None,
                     "err_rate": float(np.mean(p_e[e]))})
    rhos = np.array([r["rho"] for r in rows])
    for i, r in enumerate(rows):
        r["abs_delta_rho"] = float(abs(r["rho"] - np.delete(rhos, i).mean()))

    def corr(key, tgt):
        xs = np.array([r[key] for r in rows], dtype=float)
        ys = np.array([r[tgt] for r in rows], dtype=float)
        if len(rows) < 3 or np.isnan(xs).any():
            return None, None
        c = sp_stats.spearmanr(xs, ys)
        return float(c[0]), float(c[1])

    out = {"n_sites": n_sites, "per_protocol": rows}
    for key in ["drift_recon", "drift_leverage"]:
        for tgt in ["rho", "abs_delta_rho"]:
            rr, pp = corr(key, tgt)
            out[f"{key}_vs_{tgt}_spearman"] = rr
            out[f"{key}_vs_{tgt}_p"] = pp
    # a rank-partial correlation controlling for err_rate is not computed here;
    # err_rate is reported per protocol so the confound stays visible
    return out


def main():
    res = {"adni": run_dataset("adni", load_adni),
           "oasis": run_dataset("oasis", load_oasis_ds)}
    (OUT / "transfer_rescue.json").write_text(json.dumps(res, indent=2))
    L = ["# Cross-site transfer bound -- rescue attempt (better drift measures)", ""]
    for n in ["adni", "oasis"]:
        r = res[n]
        L += [f"## {n.upper()} ({r['n_sites']} protocols)",
              f"- drift_recon vs rho_e: Spearman {r['drift_recon_vs_rho_spearman']} "
              f"(p={r['drift_recon_vs_rho_p']})",
              f"- drift_recon vs |dRho|: Spearman {r['drift_recon_vs_abs_delta_rho_spearman']} "
              f"(p={r['drift_recon_vs_abs_delta_rho_p']})",
              f"- drift_leverage vs rho_e: Spearman {r['drift_leverage_vs_rho_spearman']} "
              f"(p={r['drift_leverage_vs_rho_p']})",
              f"- drift_leverage vs |dRho|: Spearman {r['drift_leverage_vs_abs_delta_rho_spearman']} "
              f"(p={r['drift_leverage_vs_abs_delta_rho_p']})", "",
              "| protocol | rho_e | |dRho| | drift_recon | drift_leverage | err_rate |",
              "|---|---|---|---|---|---|"]
        for p in r["per_protocol"]:
            dr = f"{p['drift_recon']:.4f}" if p['drift_recon'] is not None else "--"
            dl = f"{p['drift_leverage']:.3f}" if p['drift_leverage'] is not None else "--"
            L.append(f"| {p['protocol']} | {p['rho']:.3f} | {p['abs_delta_rho']:.3f} | "
                     f"{dr} | {dl} | {p['err_rate']:.3f} |")
        L.append("")
    (OUT / "transfer_rescue.md").write_text("\n".join(L))
    print("saved transfer_rescue.{json,md}", flush=True)


if __name__ == "__main__":
    main()
