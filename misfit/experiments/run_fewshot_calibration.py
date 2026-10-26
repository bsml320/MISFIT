"""Few-shot prospective calibration (reviewer): given k LABELED pilot subjects from a new
(held-out) site, predict per-subject error PROSPECTIVELY for the rest via a kNN calibrator on
the encoder representation. Sweeps k; the k=0 label-free baseline (1-maxprob) should rise toward
the retrospective misfit ceiling. Turns the paper's central limitation into a quantified result.
ADNI+OASIS, 5 seeds. Output: results/fewshot_calibration.json.
"""
from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path
import json, os, sys
from pathlib import Path
# CUDA visibility is selected by the caller.
import numpy as np, torch, torch.nn.functional as F  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402
sys.path.insert(0, _resolve_path("<REPO_ROOT>"))
from misfit.experiments.run_honest_estimator_table import (  # noqa: E402
    DEVICE, N_EPOCHS, N_SEEDS, SEED_BASE, PlainMLP, load_adni, load_oasis_ds)
RES = Path(_resolve_path("<REPO_ROOT>/misfit/results"))
KS, N_DRAWS = [0, 5, 10, 25, 50], 25


def train(Xtr, ytr, seed):
    """Fit a PlainMLP classifier on the training sites with the given seed."""
    torch.manual_seed(seed)
    net = PlainMLP(Xtr.shape[1], 2).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(N_EPOCHS):
        opt.zero_grad(); F.cross_entropy(net(Xtr), ytr).backward(); opt.step()
    return net


def fewshot(f, t, s, seed):
    """One seed of the few-shot sweep over held-out sites.

    For each held-out site: trains on the remaining sites, then for each pilot
    size k in KS draws k labeled pilot subjects (N_DRAWS times), fits a kNN
    error regressor on the penultimate-layer representation, and scores
    Spearman rho between predicted and observed error on the non-pilot
    subjects. k=0 is the label-free 1-maxprob baseline.

    Returns:
        Dict k -> mean rho across held-out sites (nan when no site was scorable).
    """
    rng = np.random.RandomState(seed)
    n_sites = int(s.max()) + 1
    Xall = f.to(DEVICE); yall = t.long().to(DEVICE)
    per = {k: [] for k in KS}
    for held in range(n_sites):
        tr = (s != held).numpy(); te = (s == held).numpy()
        if te.sum() < 20 or tr.sum() < 20:
            continue
        net = train(Xall[tr], yall[tr], seed)
        with torch.no_grad():
            Xte = Xall[te]; logit = net(Xte)
            rep = net.net[:4](Xte).cpu().numpy()
            pred = logit.argmax(1).cpu().numpy()
            msp = logit.softmax(1).max(1).values.cpu().numpy()
        err = (pred != t.numpy()[te]).astype(float)
        if not (0 < err.sum() < len(err)):
            continue
        n = len(err)
        for k in KS:
            if k == 0:
                r = spearmanr(1 - msp, err)[0]                # label-free prospective baseline
                if not np.isnan(r):
                    per[0].append(r)
            else:
                rr = []
                for _ in range(N_DRAWS):
                    idx = rng.choice(n, min(k, n), replace=False)
                    rest = np.setdiff1d(np.arange(n), idx)
                    if err[idx].sum() == 0 or len(rest) < 5:
                        continue
                    knn = KNeighborsRegressor(n_neighbors=min(k, 10)).fit(rep[idx], err[idx])
                    r = spearmanr(knn.predict(rep[rest]), err[rest])[0]
                    if not np.isnan(r):
                        rr.append(r)
                if rr:
                    per[k].append(float(np.mean(rr)))
    return {k: float(np.mean(v)) if v else float("nan") for k, v in per.items()}


def main():
    """Sweep pilot sizes on both cohorts and write results/fewshot_calibration.json."""
    RES.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, loader in [("adni", load_adni), ("oasis", load_oasis_ds)]:
        f, t, s = loader()
        runs = [fewshot(f, t, s, SEED_BASE + k) for k in range(N_SEEDS)]
        out[name] = {str(k): {"rho_mean": float(np.nanmean([r[k] for r in runs])),
                              "rho_std": float(np.nanstd([r[k] for r in runs]))} for k in KS}
        print(f"[{name}] prospective rho(pred error, true error) by pilot size k:")
        for k in KS:
            print(f"   k={k:2d}: {out[name][str(k)]['rho_mean']:.3f} +/- {out[name][str(k)]['rho_std']:.3f}", flush=True)
    (RES / "fewshot_calibration.json").write_text(json.dumps(out, indent=2))
    print("wrote results/fewshot_calibration.json")


if __name__ == "__main__":
    main()
