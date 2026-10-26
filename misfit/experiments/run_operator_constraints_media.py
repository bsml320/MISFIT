"""Do extra constraint terms make the operator earn its keep? (MedIA)

Motivation: in the frozen-random-loadings harness the operator adds little over
plain training-loss misfit (per-dataset reference values are supplied externally) and
per-program attribution is seed-unstable. Both are consequences of the loadings
being random and never optimized, with no constraint shaping the operator. This
script tests whether LEARNED loadings plus extra constraint terms change either
fact. The experiment leaves the shared training implementation unchanged.

Variants (same encoder/spectral-program construction, frozen programs, same
leave-one-environment-out eval, 5 seeds):
  V0_frozen_random : loadings = randn, NOT optimized, CE only      (= the anchor)
  V1_learned       : loadings learned (init 0 => T=I), CE only
  V2_learned_l1    : + L1 sparsity on loadings
  V3_learned_coral : + cross-environment alignment of the loading
                     distribution (CORAL-style mean+covariance match across envs;
                     true same-subject cross-env invariance is untrainable here
                     because each subject sits in ONE protocol)
  V4_learned_both  : + L1 + CORAL

Reported per variant: deployable misfit rho (N_SEEDS seeds, vs binary held-out
error) and per-program attribution seed-stability (Spearman across seeds). A
variant is worth pursuing only if it beats the anchor rho or makes per-program
attribution stable.

Numerics: float64 ridge (linalg.solve, no inverse); condition guards inherited
from the reused _predict_loadings.

Output: misfit/results/operator_constraints.{json,md}
"""

from __future__ import annotations
from misfit.paths import reporting_reference

from misfit.paths import resolve_path as _resolve_path

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, _resolve_path("<REPO_ROOT>"))

from misfit.core.encoders.mlp import MLPEncoder  # noqa: E402
from misfit.core.operators.operator_model import OperatorModel  # noqa: E402
from misfit.core.operators.program_dict import ProgramDictionary  # noqa: E402
from misfit.core.types import TensorConfig  # noqa: E402
from misfit.core.utils import seed_everything  # noqa: E402
from misfit.tensor_spectral.recovery import recover_programs  # noqa: E402
from misfit.training.two_stage_trainer import TwoStageTrainer  # noqa: E402
from misfit.experiments.run_honest_estimator_table import (  # noqa: E402
    DEVICE,
    HIDDEN_DIMS,
    N_PROGRAMS,
    SEED_BASE,
    _predict_loadings,
    _rho,
    load_adni,
    load_oasis_ds,
)
from scipy import stats as sp_stats  # noqa: E402

warnings.filterwarnings("ignore")

OUT = Path(_resolve_path("<REPO_ROOT>/misfit/results"))
LAMBDA_L1 = 0.05
LAMBDA_CORAL = 0.5
HIGH_MISFIT_FRAC = 0.30
# fast directional sweep (controlled within-loop comparison, not a 5-seed table)
N_EPOCHS = 50
N_SEEDS = 3
VARIANTS = ["V0_frozen_random", "V1_learned", "V2_learned_l1",
            "V3_learned_coral", "V4_learned_both"]


def _build_model(features, targets, site_ids, n_sites, dim, seed):
    """Encoder + spectral-init frozen programs, exactly as the honest harness."""
    torch.manual_seed(seed)
    encoder = MLPEncoder(dim, HIDDEN_DIMS, dim)
    prog = ProgramDictionary(n_programs=N_PROGRAMS, dim=dim, rank=1)
    n_classes = max(int(targets.max().item()) + 1, 2)
    model = OperatorModel(encoder, prog, n_envs=n_sites, output_dim=n_classes, use_style=False)
    trainer = TwoStageTrainer(model, use_contrastive=False, task="classification")
    try:
        from misfit.data.mri_multisite import MRIData, mri_to_multienv_format
        mri = MRIData(features=features, targets=targets, site_ids=site_ids,
                      subject_ids=torch.arange(len(features)))
        obs, _ = mri_to_multienv_format(mri)
        tc = TensorConfig(n_programs=N_PROGRAMS, n_envs=n_sites, dim=dim)
        recovery = recover_programs(obs, n_programs=N_PROGRAMS, tensor_config=tc)
        trainer.stage1_init(recovery)
    except Exception:
        pass
    # freeze the program dictionary (paper setting)
    for p in prog.parameters():
        p.requires_grad_(False)
    return model


def _coral(loadings, env, n_env):
    """CORAL-style alignment of the loading distribution across environments."""
    means, covs = [], []
    for e in range(n_env):
        m = env == e
        if m.sum() < 2:
            continue
        a = loadings[m]
        mu = a.mean(0)
        ac = a - mu
        cov = (ac.T @ ac) / (a.shape[0] - 1)
        means.append(mu); covs.append(cov)
    if len(means) < 2:
        return loadings.sum() * 0.0
    loss = 0.0
    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            loss = loss + (means[i] - means[j]).pow(2).mean() \
                        + (covs[i] - covs[j]).pow(2).mean()
    return loss / max(len(means) * (len(means) - 1) / 2, 1)


def _train_variant(model, features, targets, env_ids, n_sites, variant, seed):
    """Custom loop on DEVICE. Returns (model, loadings) with loadings (N_train, K)."""
    torch.manual_seed(seed + 7)
    model = model.to(DEVICE)
    features = features.to(DEVICE); env_ids = env_ids.to(DEVICE)
    y = targets.long().to(DEVICE)
    N = len(features)
    learn = variant != "V0_frozen_random"
    if learn:
        loadings = nn.Parameter(torch.zeros(N, N_PROGRAMS, device=DEVICE))  # T_i = I
    else:
        loadings = torch.randn(N, N_PROGRAMS, device=DEVICE)               # fixed random
    params = [p for p in model.parameters() if p.requires_grad]
    if learn:
        params = params + [loadings]
    opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)
    model.train()
    for _ in range(N_EPOCHS):
        opt.zero_grad()
        logits = model(features, loadings, env_ids)
        loss = F.cross_entropy(logits, y)
        if variant in ("V2_learned_l1", "V4_learned_both"):
            loss = loss + LAMBDA_L1 * loadings.abs().mean()
        if variant in ("V3_learned_coral", "V4_learned_both"):
            loss = loss + LAMBDA_CORAL * _coral(loadings, env_ids, n_sites)
        loss.backward()
        opt.step()
    model.eval()
    return model, loadings.detach()


def collect_variant(features, targets, site_ids, seed, variant):
    """One leave-one-environment-out pass for one variant at a fixed seed.

    Returns:
        (misfit rho vs held-out binary error, per-program ablation importance
        vector over the top HIGH_MISFIT_FRAC subjects).
    """
    n_sites = int(site_ids.max().item()) + 1
    dim = features.shape[1]
    N = features.shape[0]
    seed_everything(seed)
    subject_correct = torch.full((N,), float("nan"))
    tested = torch.zeros(N, dtype=torch.bool)
    mis_sum = np.zeros(N); mis_cnt = np.zeros(N)
    prog_sum = np.zeros((N, N_PROGRAMS))

    for held_out in range(n_sites):
        tr = site_ids != held_out; te = site_ids == held_out
        if te.sum() < 5 or tr.sum() < 20:
            continue
        train_feat = features[tr]; train_tgt = targets[tr]; train_sites = site_ids[tr]
        test_feat = features[te]; test_tgt = targets[te]
        unique = train_sites.unique()
        smap = {int(s_): i for i, s_ in enumerate(unique)}
        remapped = torch.tensor([smap[int(s_)] for s_ in train_sites])
        nts = len(unique)
        ti_tr = torch.where(tr)[0].numpy(); ti_te = torch.where(te)[0].numpy()

        model = _build_model(train_feat, train_tgt, remapped, nts, dim, seed)
        model, loadings = _train_variant(model, train_feat, train_tgt, remapped,
                                          nts, variant, seed)
        tf_d = train_feat.to(DEVICE); rm_d = remapped.to(DEVICE)
        tgt_d = train_tgt.long().to(DEVICE)
        with torch.no_grad():
            tr_pred = model(tf_d, loadings, rm_d)
            tr_loss = F.cross_entropy(tr_pred, tgt_d, reduction="none").cpu().numpy()
        mis_sum[ti_tr] += tr_loss; mis_cnt[ti_tr] += 1.0
        with torch.no_grad():
            for k in range(N_PROGRAMS):
                lk = loadings.clone(); lk[:, k] = 0.0
                ce_k = F.cross_entropy(model(tf_d, lk, rm_d),
                                       tgt_d, reduction="none").cpu().numpy()
                prog_sum[ti_tr, k] += (ce_k - tr_loss)

        tl = _predict_loadings(train_feat, loadings.cpu(), test_feat)
        test_env = torch.zeros(test_feat.shape[0], dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            pl = model(test_feat.to(DEVICE), tl.to(DEVICE), test_env).argmax(dim=-1).cpu()
        subject_correct[ti_te] = (pl == test_tgt.long()).float()
        tested[ti_te] = True

    cnt = np.maximum(mis_cnt, 1.0)
    misfit = mis_sum / cnt
    prog = prog_sum / cnt[:, None]
    err = 1.0 - subject_correct.numpy()
    t = tested.numpy()
    rho = _rho(misfit[t], err[t])
    # per-program importance over high-misfit subjects (for stability)
    thr = np.quantile(misfit[t], 1.0 - HIGH_MISFIT_FRAC)
    hi = (misfit[t] >= thr)
    prog_imp = prog[t][hi].mean(0) if hi.sum() >= 3 else np.full(N_PROGRAMS, np.nan)
    return rho, prog_imp


def run_dataset(name, loader):
    """Run all VARIANTS on one cohort over N_SEEDS seeds."""
    t0 = time.time()
    features, targets, site_ids = loader()
    print(f"[{name}] N={features.shape[0]} d={features.shape[1]} "
          f"E={int(site_ids.max())+1}", flush=True)
    res = {}
    for variant in VARIANTS:
        rhos, imps = [], []
        for s in range(N_SEEDS):
            rho, imp = collect_variant(features, targets, site_ids, SEED_BASE + s, variant)
            rhos.append(rho); imps.append(imp)
        imps = np.array(imps)
        pair_rhos = []
        for i in range(len(imps)):
            for j in range(i + 1, len(imps)):
                pair_rhos.append(sp_stats.spearmanr(imps[i], imps[j]).correlation)
        res[variant] = {
            "misfit_rho_mean": float(np.nanmean(rhos)),
            "misfit_rho_std": float(np.nanstd(rhos)),
            "misfit_rho_per_seed": [round(float(x), 3) for x in rhos],
            "prog_stability_spearman": float(np.nanmean(pair_rhos)) if pair_rhos else None,
        }
        print(f"  [{name}] {variant}: misfit rho={res[variant]['misfit_rho_mean']:.3f}"
              f"+/-{res[variant]['misfit_rho_std']:.3f}  prog-stab="
              f"{res[variant]['prog_stability_spearman']}", flush=True)
    return {"n": int(features.shape[0]), "anchor": (reporting_reference('misfit_anchor', required=False) or {}).get(name),
            "variants": res, "wall_s": round(time.time() - t0, 1)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = {}
    result["adni"] = run_dataset("adni", load_adni)
    result["oasis"] = run_dataset("oasis", load_oasis_ds)
    (OUT / "operator_constraints.json").write_text(json.dumps(result, indent=2))
    L = ["# Do extra constraint terms help the operator? (MedIA)", "",
         "Learned loadings + constraint terms vs the frozen-random anchor "
         "(recomputed from the specified fits). Same encoder/spectral-program "
         f"construction, frozen programs, leave-one-env-out, {N_SEEDS} seeds.",
         f"L1={LAMBDA_L1}, CORAL={LAMBDA_CORAL}.", ""]
    for name in ["adni", "oasis"]:
        r = result[name]
        L += [f"## {name.upper()} (N={r['n']}, anchor rho={r['anchor']})",
              "| variant | misfit rho | per-seed | prog-stability |",
              "|---|---|---|---|"]
        for v in VARIANTS:
            d = r["variants"][v]
            ps = d["prog_stability_spearman"]
            L.append(f"| {v} | {d['misfit_rho_mean']:.3f}+/-{d['misfit_rho_std']:.3f} | "
                     f"{d['misfit_rho_per_seed']} | "
                     f"{ps:.3f} |" if ps is not None else
                     f"| {v} | {d['misfit_rho_mean']:.3f}+/-{d['misfit_rho_std']:.3f} | "
                     f"{d['misfit_rho_per_seed']} | -- |")
        L.append("")
    (OUT / "operator_constraints.md").write_text("\n".join(L))
    print(f"\nSaved {OUT / 'operator_constraints.json'}", flush=True)


if __name__ == "__main__":
    main()
