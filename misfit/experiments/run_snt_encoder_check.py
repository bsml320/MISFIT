"""Recompute the encoder-space SNT proxy (paper eq. 6) to get a current,
in-repo number for the deployment-gap discussion. SNT_j = mean training misfit
of the k=10 nearest TRAIN subjects to j in encoder representation space; novel
subject j is held out. Reports Spearman rho(SNT, binary error). 5 seeds,
leave-one-environment-out. Output: results/snt_encoder.{json,md}
"""
from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path
import json, sys, warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, _resolve_path("<REPO_ROOT>"))
from misfit.experiments.run_honest_estimator_table import (  # noqa: E402
    N_SEEDS, SEED_BASE, _predict_loadings, _rho, _train_fire, load_adni, load_oasis_ds,
)
from misfit.core.utils import seed_everything  # noqa: E402
warnings.filterwarnings("ignore")
OUT = Path(_resolve_path("<REPO_ROOT>/misfit/results"))
K = 10


def encoder_reps(model, x):
    """Encoder outputs for x as float64 numpy, no gradients."""
    model.eval()
    with torch.no_grad():
        z = model.encoder(x)
    return z.detach().cpu().numpy().astype(np.float64)


def run_seed(features, targets, site_ids, seed):
    """One leave-one-environment-out SNT evaluation at a fixed seed.

    Per fold: trains the IOP model on the source environments, computes each
    training subject's final per-example CE (the training misfit), then scores
    every held-out subject as the mean misfit of its K nearest training
    subjects in encoder space. Held-out error uses ridge-predicted loadings.

    Returns:
        Spearman rho between the pooled SNT score and binary held-out error.
    """
    n_sites = int(site_ids.max().item()) + 1
    dim = features.shape[1]
    seed_everything(seed)
    snt, err = [], []
    for held in range(n_sites):
        tr = site_ids != held; te = site_ids == held
        if te.sum() < 5 or tr.sum() < 20:
            continue
        trf, trt, trs = features[tr], targets[tr], site_ids[tr]
        tef, tet = features[te], targets[te]
        uniq = trs.unique(); smap = {int(s): i for i, s in enumerate(uniq)}
        rem = torch.tensor([smap[int(s)] for s in trs]); nts = len(uniq)
        model, loadings = _train_fire(trf, trt, rem, nts, dim, seed)
        with torch.no_grad():
            tr_loss = F.cross_entropy(model(trf, loadings, rem), trt.long(),
                                      reduction="none").numpy()
        ztr = encoder_reps(model, trf); zte = encoder_reps(model, tef)
        nn = NearestNeighbors(n_neighbors=min(K, len(ztr))).fit(ztr)
        _, idx = nn.kneighbors(zte)
        snt_te = tr_loss[idx].mean(1)
        tl = _predict_loadings(trf, loadings, tef)
        with torch.no_grad():
            pl = model(tef, tl, torch.zeros(len(tef), dtype=torch.long)).argmax(-1)
        err_te = 1.0 - (pl == tet.long()).float().numpy()
        snt.extend(snt_te.tolist()); err.extend(err_te.tolist())
    return _rho(np.array(snt), np.array(err))


def run_dataset(name, loader):
    f, t, s = loader()
    rhos = [run_seed(f, t, s, SEED_BASE + i) for i in range(N_SEEDS)]
    print(f"[{name}] encoder-space SNT rho = {np.mean(rhos):.3f} +/- {np.std(rhos):.3f} "
          f"per-seed {[round(x, 3) for x in rhos]}", flush=True)
    return {"rho_mean": float(np.mean(rhos)), "rho_std": float(np.std(rhos)),
            "per_seed": [round(float(x), 3) for x in rhos]}


def main():
    res = {"adni": run_dataset("adni", load_adni),
           "oasis": run_dataset("oasis", load_oasis_ds)}
    (OUT / "snt_encoder.json").write_text(json.dumps(res, indent=2))
    L = ["# Encoder-space SNT proxy (paper eq. 6), current in-repo recompute", "",
         f"SNT = mean training misfit of k={K} nearest train subjects in encoder space; "
         "novel held-out subject. Spearman rho(SNT, binary error), 5 seeds.", ""]
    for n in ["adni", "oasis"]:
        r = res[n]
        L.append(f"- **{n.upper()}**: rho = {r['rho_mean']:.3f} +/- {r['rho_std']:.3f} "
                 f"(per-seed {r['per_seed']})")
    (OUT / "snt_encoder.md").write_text("\n".join(L))
    print("saved snt_encoder.{json,md}", flush=True)


if __name__ == "__main__":
    main()
