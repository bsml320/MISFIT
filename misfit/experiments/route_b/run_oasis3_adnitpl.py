"""Experiment 1(b): OASIS-3 under an INDEPENDENT template (the ADNI DARTEL template),
with all downstream learned stages source-fold-only at analysis time.

The ADNI Template_1..6 (built from ADNI subjects only; Template_6_2mni.mat present)
contains no OASIS-3 information, satisfying the register's/PI's "independently fixed
template" option. For the 1,360 leakage-free earliest sessions: copy rc1/rc2/c1 into a
fresh work tree (originals untouched), estimate per-subject flow fields against the ADNI
template (Dartel warp-to-existing), normalize to MNI (modulated, 3 mm smooth), then
evaluate with SOURCE-FOLD mask + standardization + PCA-50 and a PlainMLP (3 seeds).

Stages: prep | warp | norm | eval | all        (resumable; CPU/SPM, 12 parallel)
Output: results/oasis3/oasis3_adnitpl_sourceonly.json
"""
from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

import glob
import json
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, _resolve_path("<REPO_ROOT>"))
sys.path.insert(0, _resolve_path("<REPO_ROOT>/misfit/experiments/route_b"))
from vbm_pipeline import dartel_mni_norm, dartel_warp_to_existing  # noqa: E402

SRC = Path(_resolve_path("<REPO_ROOT>/data/oasis3/route_b_work"))
WORK = Path(_resolve_path("<REPO_ROOT>/data/oasis3/adnitpl_work"))
ADNI_TDIR = str(Path(glob.glob(
    _resolve_path("<REPO_ROOT>/misfit/experiments/route_b/_adni_work/*/Template_6.nii"))[0]).parent)
RES = Path(_resolve_path("<REPO_ROOT>/misfit/results/oasis3"))
JOBS = 12


def leakage_free_sessions():
    """Earliest session per participant with labels and frequency-coded scanner ids.

    Returns:
        Tuple (session id list, label array, scanner-site array), aligned.
    """
    d = np.load(_resolve_path("<REPO_ROOT>/data/oasis3/oasis3_features.npz"), allow_pickle=True)
    subj = d["subjects"].astype(str)
    sess = d["sessions"].astype(str)
    day = np.array([int(re.search(r"_d(\d+)$", x).group(1)) for x in sess])
    keep = np.array(sorted(np.where(subj == u)[0][np.argmin(day[subj == u])]
                           for u in np.unique(subj)))
    lab = d["label"].astype(int)[keep]
    scan = d["scanner"].astype(str)[keep]
    uniq, cnt = np.unique(scan, return_counts=True)
    order = uniq[np.argsort(-cnt)]
    code = {s: i for i, s in enumerate(order)}
    site = np.array([code[x] for x in scan])
    return list(sess[keep]), lab, site


def stage_prep(sessions):
    """Copy rc1/rc2/c1 inputs into the fresh work tree; originals stay untouched."""
    WORK.mkdir(exist_ok=True)
    n = 0
    for s in sessions:
        d = WORK / s
        d.mkdir(exist_ok=True)
        for f in ("rc1pT1.nii", "rc2pT1.nii", "c1pT1.nii"):
            if not (d / f).exists():
                shutil.copy2(SRC / s / f, d / f)
                n += 1
    print(f"[prep] copied {n} files for {len(sessions)} sessions -> {WORK}", flush=True)


def stage_warp(sessions):
    """Estimate per-subject DARTEL flow fields against the fixed ADNI template.

    Resumable: sessions with an existing u_rc1pT1.nii are skipped. Work is
    chunked (8 sessions per SPM call) and run with JOBS parallel workers;
    chunk failures are counted and retried on a re-run.
    """
    todo = [s for s in sessions if not (WORK / s / "u_rc1pT1.nii").exists()]
    CH = 8
    chunks = [todo[i:i + CH] for i in range(0, len(todo), CH)]
    print(f"[warp] {len(todo)} sessions vs ADNI template ({ADNI_TDIR}) "
          f"in {len(chunks)} chunks x{CH}, {JOBS} parallel", flush=True)

    def do(grp):
        rc1s = [str(WORK / s / "rc1pT1.nii") for s in grp]
        rc2s = [str(WORK / s / "rc2pT1.nii") for s in grp]
        return "WARP1_DONE" in dartel_warp_to_existing(rc1s, rc2s, ADNI_TDIR)

    done = fail = 0
    with ThreadPoolExecutor(max_workers=JOBS) as ex:
        futs = {ex.submit(do, g): g for g in chunks}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                done += fut.result()
            except Exception as e:  # noqa: BLE001
                fail += 1
                print(f"  chunk error: {e}", flush=True)
            if i % 15 == 0:
                nf = sum((WORK / s / "u_rc1pT1.nii").exists() for s in sessions)
                print(f"  ...{i}/{len(chunks)} chunks; flows {nf}/{len(sessions)}", flush=True)
    nf = sum((WORK / s / "u_rc1pT1.nii").exists() for s in sessions)
    print(f"[warp] complete: {nf}/{len(sessions)} flow fields ({fail} chunk failures; "
          f"re-run warp to retry)", flush=True)


def stage_norm(sessions):
    """Normalize warped sessions to MNI (modulated, smoothed); resumable like warp."""
    t6 = f"{ADNI_TDIR}/Template_6.nii"
    subj = [(str(WORK / s / "u_rc1pT1.nii"), str(WORK / s / "c1pT1.nii"))
            for s in sessions
            if (WORK / s / "u_rc1pT1.nii").exists() and not (WORK / s / "smwc1pT1.nii").exists()]
    CH = 20
    chunks = [subj[i:i + CH] for i in range(0, len(subj), CH)]
    print(f"[norm] {len(subj)} sessions in {len(chunks)} chunks x{CH}, {JOBS} parallel", flush=True)

    def do(grp):
        return "NORM_DONE" in dartel_mni_norm(t6, grp)

    with ThreadPoolExecutor(max_workers=JOBS) as ex:
        futs = {ex.submit(do, g): g for g in chunks}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"  chunk error: {e}", flush=True)
            if i % 10 == 0:
                nd = sum((WORK / s / "smwc1pT1.nii").exists() for s in sessions)
                print(f"  ...{i}/{len(chunks)} chunks; smwc1 {nd}/{len(sessions)}", flush=True)
    nd = sum((WORK / s / "smwc1pT1.nii").exists() for s in sessions)
    print(f"[norm] complete: {nd}/{len(sessions)} smwc1 written", flush=True)


def stage_eval(sessions, lab, site):
    """Evaluate reliability on the ADNI-template features with source-fold-only stages.

    Per leave-one-scanner-out fold: brain mask, standardization and PCA-50 are
    fit on the training scanners only (seeded by the held-out id), then a
    PlainMLP is trained per seed; each training subject's final-epoch CE,
    averaged over the folds it appears in, is the reliability score. Writes the
    JSON summary (pooled/within-scanner rho, AUROC over 3 seeds) and the
    per-seed arrays npz.
    """
    import torch
    import torch.nn.functional as F
    import nibabel as nib
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    from misfit.experiments.run_honest_estimator_table import PlainMLP
    torch.set_num_threads(16)

    have = np.array([(WORK / s / "smwc1pT1.nii").exists() for s in sessions])
    print(f"[eval] {have.sum()}/{len(sessions)} volumes available", flush=True)
    sessions = [s for s, h in zip(sessions, have) if h]
    lab, site = lab[have], site[have]
    with ThreadPoolExecutor(max_workers=16) as ex:
        vols = list(ex.map(lambda s: nib.load(str(WORK / s / "smwc1pT1.nii"))
                           .get_fdata(dtype=np.float32).ravel(), sessions))
    X = np.stack(vols)
    y = lab
    import collections
    res = {"n": int(len(y)), "note": "ADNI template (no OASIS-3 info) + source-fold "
                                     "mask/standardization/PCA = full-pipeline isolation",
           "per_scanner_available_n": dict(collections.Counter(site.tolist())),
           "sessions_missing_smwc1": int((~have).sum())}
    yt = torch.from_numpy(y).long()
    projs = {}
    for held in np.unique(site):
        tr = np.where(site != held)[0]
        te = np.where(site == held)[0]
        if len(te) < 5 or len(tr) < 20:
            continue
        m = X[tr].mean(0) > 0.1
        Xm = torch.from_numpy(X[:, m])
        mu = Xm[tr].mean(0)
        sd = Xm[tr].std(0) + 1e-6
        torch.manual_seed(int(held))
        _, _, V = torch.svd_lowrank((Xm[tr] - mu) / sd, q=50, niter=4)
        projs[held] = ((Xm - mu) / sd) @ V
        res.setdefault("included_folds", []).append(int(held))
        print(f"  fold {held}: features ready", flush=True)
    pooled, within, auroc = [], [], []
    score_seeds, err_seeds = [], []
    for s in range(3):
        msum, mcnt = np.zeros(len(y)), np.zeros(len(y))
        pred = np.full(len(y), -1)
        tested = np.zeros(len(y), bool)
        for held, Z in projs.items():
            tr = np.where(site != held)[0]
            te = np.where(site == held)[0]
            torch.manual_seed(s)
            mdl = PlainMLP(50, 2)
            opt = torch.optim.Adam(mdl.parameters(), lr=1e-3, weight_decay=1e-4)
            mdl.train()
            for _ in range(100):
                opt.zero_grad()
                F.cross_entropy(mdl(Z[tr]), yt[tr]).backward()
                opt.step()
            mdl.eval()
            with torch.no_grad():
                msum[tr] += F.cross_entropy(mdl(Z[tr]), yt[tr], reduction="none").numpy()
                pred[te] = mdl(Z[te]).argmax(1).numpy()
            mcnt[tr] += 1
            tested[te] = True
        err = (pred != y).astype(float)
        sc = msum / np.maximum(mcnt, 1)
        score_seeds.append(sc.copy()); err_seeds.append(err.copy())
        m_ = tested
        pooled.append(float(spearmanr(sc[m_], err[m_])[0]))
        auroc.append(float(roc_auc_score(err[m_], sc[m_]))
                     if 0 < err[m_].sum() < m_.sum() else float("nan"))
        wr = [spearmanr(sc[m_ & (site == e)], err[m_ & (site == e)])[0]
              for e in np.unique(site[m_])
              if (m_ & (site == e)).sum() >= 8
              and 0 < err[m_ & (site == e)].sum() < (m_ & (site == e)).sum()]
        within.append(float(np.mean(wr)))
        print(f"  seed {s}: within={within[-1]:.3f} AUROC={auroc[-1]:.3f}", flush=True)
    for k, v in (("pooled_rho", pooled), ("within_scanner_rho", within), ("auroc", auroc)):
        res[k] = {"mean": round(float(np.mean(v)), 3), "std": round(float(np.std(v)), 3)}
    np.savez(str(RES / "oasis3_adnitpl_arrays.npz"),
             sessions=np.array(sessions), site=site, label=y,
             misfit_per_seed=np.stack(score_seeds), err_per_seed=np.stack(err_seeds))
    res["arrays"] = "oasis3_adnitpl_arrays.npz (per-seed score/error, sessions, scanner, label)"
    json.dump(res, open(RES / "oasis3_adnitpl_sourceonly.json", "w"), indent=2)
    print("[eval] wrote oasis3_adnitpl_sourceonly.json", flush=True)


def main():
    """Dispatch the stage named on the command line (prep|warp|norm|eval|all)."""
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    sessions, lab, site = leakage_free_sessions()
    if stage in ("prep", "all"):
        stage_prep(sessions)
    if stage in ("warp", "all"):
        stage_warp(sessions)
    if stage in ("norm", "all"):
        stage_norm(sessions)
    if stage in ("eval", "all"):
        stage_eval(sessions, lab, site)


if __name__ == "__main__":
    main()
