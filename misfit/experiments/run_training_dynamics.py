"""H2 (reviewer): a LABEL-FAIR comparison. Structural misfit uses the subject's label;
so do established training-dynamics example-difficulty scores. All of them are computed
from the SAME classifier training to test whether misfit (the final per-subject training
loss) is competitive with them:
  - AUM  (area under the margin, Pleiss et al. 2020) -- averaged true-vs-runner-up margin
  - forgetting events (Toneva et al. 2019) -- correct->incorrect transitions during training
  - EL2N (Paul et al. 2021) -- ||softmax - onehot|| averaged over epochs
  - misfit-proxy -- the final-epoch per-subject cross-entropy (== the paper's plain ERM loss)
All aggregated over the leave-one-environment-out folds in which a subject is in TRAINING,
scored against the subject's held-out error. ADNI + OASIS-1, 5 seeds. GPU 2.
Output: results/training_dynamics.json.
"""
from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path
import json, os, sys
from pathlib import Path
# CUDA visibility is selected by the caller.
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
sys.path.insert(0, _resolve_path("<REPO_ROOT>"))
from misfit.experiments.run_honest_estimator_table import (  # noqa: E402
    DEVICE, N_EPOCHS, N_SEEDS, SEED_BASE, PlainMLP, load_adni, load_oasis_ds)

RES = Path(_resolve_path("<REPO_ROOT>/misfit/results"))
SCORES = ["misfit", "aum_neg", "forgetting", "el2n"]   # all "high = unreliable"


def train_log_dynamics(Xtr, ytr, seed):
    """Train PlainMLP on (Xtr, ytr), logging per-example dynamics each epoch.

    Returns:
        (model, scores) where scores holds "misfit" (final-epoch CE),
        "aum_neg" (negated epoch-averaged true-vs-runner-up margin),
        "forgetting" (count of correct->incorrect transitions) and "el2n"
        (epoch-averaged ||softmax - onehot||); all oriented high = unreliable.
    """
    torch.manual_seed(seed)
    model = PlainMLP(Xtr.shape[1], 2).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ar = torch.arange(len(ytr), device=DEVICE)
    margins, corrects, el2ns = [], [], []
    onehot = F.one_hot(ytr, 2).float()
    model.train()
    for _ in range(N_EPOCHS):
        opt.zero_grad()
        loss = F.cross_entropy(model(Xtr), ytr)
        loss.backward(); opt.step()
        with torch.no_grad():
            lg = model(Xtr)
            true_lg = lg.gather(1, ytr.view(-1, 1)).squeeze(1)
            masked = lg.clone(); masked[ar, ytr] = float("-inf")
            margins.append((true_lg - masked.max(1).values).cpu())
            corrects.append((lg.argmax(1) == ytr).cpu())
            el2ns.append((lg.softmax(1) - onehot).norm(dim=1).cpu())
    M = torch.stack(margins); C = torch.stack(corrects); E = torch.stack(el2ns)
    with torch.no_grad():
        floss = F.cross_entropy(model(Xtr), ytr, reduction="none").cpu().numpy()
    aum = M.mean(0).numpy()
    forget = (C[:-1] & (~C[1:])).sum(0).float().numpy()
    el2n = E.mean(0).numpy()
    return model, {"misfit": floss, "aum_neg": -aum, "forgetting": forget, "el2n": el2n}


def collect_dynamics(features, targets, site_ids, seed):
    """Leave-one-environment-out collection of all four scores at one seed.

    Scores are averaged over the folds in which a subject is in training;
    the held-out fold's predictions define the binary error target.

    Returns:
        (scores dict keyed by SCORES, per-subject error, tested mask).
    """
    n_sites = int(site_ids.max()) + 1
    N = len(features)
    ssum = {k: np.zeros(N) for k in SCORES}
    scount = np.zeros(N)
    pred = np.full(N, -1); tested = np.zeros(N, bool)
    Xall = features.to(DEVICE); yall = targets.long().to(DEVICE)
    for held in range(n_sites):
        tr = (site_ids != held).numpy(); te = (site_ids == held).numpy()
        if te.sum() < 5 or tr.sum() < 20:
            continue
        idx_tr = np.where(tr)[0]
        model, sc = train_log_dynamics(Xall[tr], yall[tr], seed)
        for k in SCORES:
            ssum[k][idx_tr] += sc[k]
        scount[idx_tr] += 1
        with torch.no_grad():
            pred[te] = model(Xall[te]).argmax(1).cpu().numpy()
        tested[te] = True
    scores = {k: ssum[k] / np.maximum(scount, 1) for k in SCORES}
    err = (pred != targets.numpy().astype(int)).astype(float)
    return scores, err, tested


def main():
    out = {}
    for name, loader in [("adni", load_adni), ("oasis", load_oasis_ds)]:
        f, t, s = loader()
        per = {k: {"rho": [], "auroc": []} for k in SCORES}
        for seed in range(N_SEEDS):
            scores, err, m = collect_dynamics(f, t, s, SEED_BASE + seed)
            e = err[m]
            for k in SCORES:
                sc = scores[k][m]
                per[k]["rho"].append(float(spearmanr(sc, e)[0]))
                if 0 < e.sum() < len(e):
                    per[k]["auroc"].append(float(roc_auc_score(e, sc)))
        out[name] = {k: {"rho_mean": float(np.mean(v["rho"])), "rho_std": float(np.std(v["rho"])),
                         "auroc_mean": float(np.mean(v["auroc"])), "auroc_std": float(np.std(v["auroc"]))}
                     for k, v in per.items()}
        print(f"[{name}]")
        for k in SCORES:
            r = out[name][k]
            print(f"   {k:11s} rho {r['rho_mean']:.3f}±{r['rho_std']:.3f}  "
                  f"AUROC {r['auroc_mean']:.3f}±{r['auroc_std']:.3f}", flush=True)
    (RES / "training_dynamics.json").write_text(json.dumps(out, indent=2))
    print("wrote results/training_dynamics.json")


if __name__ == "__main__":
    main()
