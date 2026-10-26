"""Clinical-insight panel for the MIA submission (PI request 2026-10-03).

A) FALSE-NEGATIVE CAPTURE YIELD (retrospective audit, clinical error framing):
   fraction of all missed-impairment errors (FN = impaired subject misclassified)
   captured when the top-k% highest-misfit subjects are reviewed, k in {10,20,30,50},
   per seed; also the same for false positives. ADNI + OASIS-1 (per-seed dump) and
   leakage-free OASIS-3 (collect(), 5 seeds).
B) OASIS-3 CLINICAL ASSOCIATIONS on the disease-matched external cohort:
   misfit vs CDR stage (0/0.5/>=1), age, sex (labels CSV, earliest sessions),
   per-seed Spearman signs.
C) COVARIATE-ADJUSTED ANATOMY: age/sex-partialed rank association between raw GM and
   misfit, voxelwise on ADNI and participant-level OASIS-3; split-half stability of the
   adjusted map and its correlation with the unadjusted map; per-seed sign agreement
   (fraction of voxels with the same sign in all 5 per-seed adjusted maps, among the
   top-decile |association| voxels of the mean map).
D) FIELD-STRENGTH READING: mean within-scanner rho for 1.5T vs 3T scanners from
   oasis3_per_scanner.json (no new compute).
GPU 2 for the OASIS-3 collect; the rest CPU. Output: results/clinical_insight_panel.json
"""
from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

import json
import os
import re
import sys

# CUDA visibility is selected by the caller.
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402

sys.path.insert(0, _resolve_path("<REPO_ROOT>"))
from misfit.experiments.run_clinical_stratified_media import collect  # noqa: E402

RES = _resolve_path("<REPO_ROOT>/misfit/results")
OUT = f"{RES}/clinical_insight_panel.json"
out = {}


def capture_yield(mis_seeds, err_seeds, lab, ks=(10, 20, 30, 50)):
    """Per seed: among subjects with an error of each type, fraction whose misfit is in
    the top-k%. FN = error & impaired(1); FP = error & control(0)."""
    y = {}
    for kind, cls in (("fn", 1), ("fp", 0)):
        rows = {k: [] for k in ks}
        for s in range(mis_seeds.shape[0]):
            m, e = mis_seeds[s], err_seeds[s]
            bad = (e > 0.5) & (lab == cls)
            if bad.sum() == 0:
                continue
            pct = 100.0 * rankdata(m) / len(m)
            for k in ks:
                rows[k].append(float((pct[bad] >= 100 - k).mean()))
        y[kind] = {f"top{k}": {"mean": round(float(np.mean(v)), 3),
                               "sd": round(float(np.std(v)), 3),
                               "n_seeds": len(v)} for k, v in rows.items()}
    return y


# ---------------- A: ADNI + OASIS-1 from the per-seed dump
d = np.load(f"{RES}/perseed_misfit_two_phases.npz")
out["fn_capture"] = {
    "adni": capture_yield(d["adni_mis_A"], d["adni_err_A"], d["adni_label"]),
    "oasis1": capture_yield(d["oasis_mis_A"], d["oasis_err_A"], d["oasis_label"]),
}

# ---------------- OASIS-3: collect 5 seeds (leakage-free)
dd = np.load(_resolve_path("<REPO_ROOT>/data/oasis3/oasis3_features.npz"), allow_pickle=True)
subj = dd["subjects"].astype(str)
sess = dd["sessions"].astype(str)
day = np.array([int(re.search(r"_d(\d+)$", x).group(1)) for x in sess])
keep = np.array(sorted(np.where(subj == u)[0][np.argmin(day[subj == u])]
                       for u in np.unique(subj)))
sess_k = sess[keep]
lab3 = dd["label"].astype(int)[keep]
age3 = dd["age"].astype(float)[keep]
sex3 = np.array([1 if str(x).strip().upper().startswith("M") else 0
                 for x in dd["sex"].astype(str)])[keep]
scan = dd["scanner"].astype(str)[keep]
uniq, cnt = np.unique(scan, return_counts=True)
order = uniq[np.argsort(-cnt)]
code = {s: i for i, s in enumerate(order)}
site3 = torch.tensor([code[x] for x in scan])
f3 = torch.tensor(dd["features"][keep], dtype=torch.float32)
t3 = torch.tensor(lab3)
mis3, err3 = [], []
for sd in (42, 43, 44, 45, 46):
    c = collect(f3, t3, site3, sd)
    mis3.append(c["misfit"])
    err3.append(c["err"])
    print(f"[oasis3 collect] seed {sd} done", flush=True)
mis3 = np.stack(mis3)
err3 = np.stack(err3)
out["fn_capture"]["oasis3"] = capture_yield(mis3, err3, lab3)

# ---------------- B: OASIS-3 clinical associations
labdf = pd.read_csv(_resolve_path("<REPO_ROOT>/data/oasis3/oasis3_mr_labels.csv"))
cdr_by_sess = dict(zip(labdf["session"].astype(str), labdf["CDRTOT"].astype(float)))
cdr3 = np.array([cdr_by_sess.get(s, np.nan) for s in sess_k])
stage3 = np.where(cdr3 == 0, 0, np.where(cdr3 == 0.5, 1, 2))
mis3m = mis3.mean(0)
val = np.isfinite(cdr3)


def sp(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    r = spearmanr(a[m], b[m])
    return {"rho": round(float(r.statistic), 3), "p": float(f"{r.pvalue:.3g}"), "n": int(m.sum())}


out["oasis3_clinical"] = {
    "misfit_vs_cdr_stage": sp(stage3[val].astype(float), mis3m[val]),
    "misfit_vs_age": sp(age3, mis3m),
    "misfit_vs_sex_male1": sp(sex3.astype(float), mis3m),
    "per_seed_cdr_rho": [round(float(spearmanr(stage3[val], mis3[s][val])[0]), 3)
                         for s in range(5)],
    "per_seed_age_rho": [round(float(spearmanr(age3, mis3[s])[0]), 3) for s in range(5)],
    "stage_n": [int((stage3[val] == k).sum()) for k in (0, 1, 2)],
    "stage_median_misfit_percentile": [
        round(float(np.median(100 * rankdata(mis3m)[val & (stage3 == k)] / len(mis3m))), 1)
        for k in (0, 1, 2)],
}
print("[B] oasis3 clinical:", json.dumps(out["oasis3_clinical"])[:200], flush=True)

# ---------------- C: covariate-adjusted anatomy
def rank_residual(x, covs):
    """Ranks of x residualized against covariate ranks (plus intercept), unit variance."""
    r = rankdata(x).astype(np.float64)
    C = np.column_stack([rankdata(c) for c in covs] + [np.ones(len(x))])
    beta = np.linalg.lstsq(C, r, rcond=None)[0]
    res = r - C @ beta
    sdv = res.std()
    return res / (sdv + 1e-12)


def adjusted_map(Xm, mis_vec, covs, idx):
    """Partial rank correlation of each voxel with misfit, given covariates; rows idx."""
    sub = Xm[idx]
    g = np.argsort(np.argsort(sub, axis=0), axis=0).astype(np.float32)
    C = np.column_stack([rankdata(c[idx]) for c in covs] + [np.ones(len(idx))]).astype(np.float32)
    # residualize voxel ranks against covariate ranks (vectorized lstsq via pinv)
    P = C @ np.linalg.pinv(C)
    gr = g - P @ g
    gr /= (gr.std(0) + 1e-9)
    mr = rank_residual(mis_vec[idx], [c[idx] for c in covs]).astype(np.float32)
    return (gr * mr[:, None]).mean(0)


def anatomy(tag, paths_or_X, mis_seeds, covs):
    """Covariate-adjusted anatomical association summary for one cohort.

    Args:
        tag: Cohort name used in progress prints.
        paths_or_X: Volume paths to load, or a preloaded (N, voxels) array.
        mis_seeds: Per-seed misfit matrix (seeds, N).
        covs: Covariate vectors (age, sex) to partial out.

    Returns:
        Dict with the fraction of negative adjusted associations, split-half
        stability of the adjusted map, its correlation with the unadjusted
        map, and the per-seed sign agreement among top-decile voxels.
    """
    if isinstance(paths_or_X, np.ndarray):
        X = paths_or_X
    else:
        with ThreadPoolExecutor(max_workers=16) as ex:
            X = np.stack(list(ex.map(
                lambda p: __import__("nibabel").load(p).get_fdata(dtype=np.float32).ravel(),
                paths_or_X)))
    mask = X.mean(0) > 0.1
    Xm = X[:, mask]
    del X
    n = Xm.shape[0]
    mis_mean = mis_seeds.mean(0)
    allidx = np.arange(n)
    adj = adjusted_map(Xm, mis_mean, covs, allidx)
    rng = np.random.RandomState(0)
    perm = rng.permutation(n)
    h1 = adjusted_map(Xm, mis_mean, covs, perm[:n // 2])
    h2 = adjusted_map(Xm, mis_mean, covs, perm[n // 2:])
    split_half = float(np.corrcoef(h1, h2)[0, 1])
    # unadjusted for comparison
    g = np.argsort(np.argsort(Xm, axis=0), axis=0).astype(np.float32)
    g -= g.mean(0)
    g /= (g.std(0) + 1e-9)
    mr = rankdata(mis_mean).astype(np.float32)
    mr = (mr - mr.mean()) / mr.std()
    unadj = (g * mr[:, None]).mean(0)
    corr_unadj = float(np.corrcoef(adj, unadj)[0, 1])
    # per-seed adjusted maps: sign agreement among top-decile |adj| voxels
    top = np.abs(adj) >= np.quantile(np.abs(adj), 0.9)
    signs = []
    for s in range(mis_seeds.shape[0]):
        m_s = adjusted_map(Xm, mis_seeds[s], covs, allidx)
        signs.append(np.sign(m_s[top]))
        print(f"  [{tag}] per-seed adjusted map {s} done", flush=True)
    signs = np.stack(signs)
    agree = float((np.abs(signs.sum(0)) == mis_seeds.shape[0]).mean())
    return {"n": int(n), "n_voxels": int(mask.sum()),
            "adjusted_frac_negative": round(float((adj < 0).mean()), 3),
            "adjusted_split_half": round(split_half, 3),
            "corr_adjusted_vs_unadjusted": round(corr_unadj, 3),
            "per_seed_sign_agreement_topdecile": round(agree, 3)}


# ADNI: voxel data + age/sex from labels CSV (loader order)
from misfit.experiments.run_reliability_biology_reexport import adni_age  # noqa: E402
age_a, _ = adni_age()
import pandas as pd  # noqa: F811
from misfit.data.adni_loader import (  # noqa: E402
    IMPAIRED_GROUPS, LABELS_CSV, MIN_SUBJECTS_PER_SITE, NII_DIR, PROTOCOLS)
df = pd.read_csv(LABELS_CSV)
lut = {}
for _, r in df.iterrows():
    lut.setdefault((str(r["dataset_id"]).strip(), str(r["file_name"]).strip()), r)
paths_a, sex_a = [], []
for protocol in PROTOCOLS:
    pdir = NII_DIR / protocol
    if not pdir.exists():
        continue
    pp, ss = [], []
    for sdir in sorted(pdir.iterdir()):
        fpath = sdir / "smwc1pT1.nii"
        k = (protocol, sdir.name)
        if not fpath.exists() or k not in lut or str(lut[k]["Group"]).strip() not in {"CN"} | IMPAIRED_GROUPS:
            continue
        pp.append(str(fpath))
        ss.append(1.0 if str(lut[k]["Sex"]).strip().upper() == "M" else 0.0)
    if len(pp) >= MIN_SUBJECTS_PER_SITE:
        paths_a += pp
        sex_a += ss
sex_a = np.array(sex_a)
out["adjusted_anatomy"] = {}
out["adjusted_anatomy"]["adni"] = anatomy("adni", paths_a, d["adni_mis_A"], [age_a, sex_a])
print("[C] adni done", flush=True)

W = _resolve_path("<REPO_ROOT>/data/oasis3/route_b_work")
paths3 = [f"{W}/{s}/smwc1pT1.nii" for s in sess_k]
out["adjusted_anatomy"]["oasis3"] = anatomy("oasis3", paths3, mis3, [age3, sex3.astype(float)])
print("[C] oasis3 done", flush=True)

# ---------------- D: field strength
ps = json.load(open(f"{RES}/oasis3/oasis3_per_scanner.json"))
f15 = [p["rho_mean"] for p in ps if p["scanner"] in ("Vision", "Sonata") and np.isfinite(p.get("rho_mean", np.nan))]
f30 = [p["rho_mean"] for p in ps if p["scanner"] in ("TrioTim", "Biograph_mMR", "MAGNETOM_Vida")]
out["field_strength"] = {"mean_rho_15T_scored": round(float(np.mean(f15)), 3),
                         "scanners_15T": ["Vision", "Sonata"],
                         "mean_rho_3T": round(float(np.mean(f30)), 3),
                         "note": "descriptive; Avanto (1.5T, N=2) below scoring threshold"}

json.dump(out, open(OUT, "w"), indent=2)
print("wrote", OUT, flush=True)
