"""Verify two flagged demographic fields from the data itself (no metadata guessed).

UKBB sex coding: the label.csv 'sex' column is 0/1 with no local codebook. Decide the
mapping with total gray-matter volume: modulated smwc1 GM sums are ~10% larger in males
(large effect, d~1). Sample 250 subjects per code from the loader-eligible set.
ABIDE demographics: replicate load_abide()'s exact chain (cc200, cpac, quality-checked,
mean_timeseries validity, site>=10) and read AGE_AT_SCAN / SEX from the PCP phenotype
(ABIDE coding SEX 1=male 2=female; sanity anchor: ASD cohorts are ~85% male).
Output: results/provenance/ukbb_abide_demo_verification.json
"""

from misfit.paths import resolve_path as _resolve_path
import csv, json, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import nibabel as nib
sys.path.insert(0, _resolve_path("<REPO_ROOT>"))

out = {}

# ---------------- UKBB ----------------
from misfit.data.ukbb_loader import LABELS, _smwc1
rows = list(csv.DictReader(open(LABELS)))
recs = [(str(_smwc1(r["file_name"])), int(r["sex"])) for r in rows
        if r["sex"] in ("0", "1") and r["age"].isdigit() and _smwc1(r["file_name"]).exists()]
sex = np.array([r[1] for r in recs])
# 250 subjects per sex code, fixed RNG; GM totals decide which code is male
rng = np.random.RandomState(0)
pick = {c: rng.choice(np.where(sex == c)[0], 250, replace=False) for c in (0, 1)}
def gm_total(p):
    return float(nib.load(p).get_fdata(dtype=np.float32).sum())
res = {}
for c in (0, 1):
    with ThreadPoolExecutor(max_workers=16) as ex:
        v = np.array(list(ex.map(gm_total, [recs[i][0] for i in pick[c]])))
    res[c] = (float(v.mean()), float(v.std()))
    print(f"UKBB sex-code {c}: total modulated GM sum {v.mean():.0f} ± {v.std():.0f} (n=250)", flush=True)
d = (res[1][0] - res[0][0]) / np.sqrt((res[0][1]**2 + res[1][1]**2) / 2)
male_code = 1 if res[1][0] > res[0][0] else 0
print(f"Cohen's d (code1 - code0) = {d:.2f}  ->  male code = {male_code}", flush=True)
frac1 = float((sex == 1).mean())
out["ukbb"] = {"eligible_n": int(len(sex)), "gm_sum_code0": res[0], "gm_sum_code1": res[1],
               "cohens_d_code1_minus_code0": float(d), "inferred_male_code": int(male_code),
               "pct_female_full_extract": round(100 * (1 - frac1 if male_code == 1 else frac1), 1),
               "method": "total modulated GM volume, 250/class; male ~10% larger (large effect)"}

# ---------------- ABIDE ----------------
import misfit.data.abide_loader as al
from nilearn.datasets import fetch_abide_pcp
abide = fetch_abide_pcp(data_dir=str(al.DEFAULT_DATA_DIR), derivatives=["rois_cc200"],
                           pipeline="cpac", quality_checked=True, verbose=0)
pheno = abide.phenotypic
rois = abide.rois_cc200
# replicate load_abide()'s subject filtering exactly (feature validity, then site >= 10)
valid = []
for i in range(len(rois)):
    ts = rois[i]
    if ts is None or len(ts) == 0:
        continue
    feat = al._extract_features(ts, "mean_timeseries")
    if feat is not None and np.isfinite(feat).all():
        valid.append(i)
pv = pheno.iloc[valid].reset_index(drop=True)
site = pv["SITE_ID"].values
keep_sites = [s for s in np.unique(site) if (site == s).sum() >= 10]
pv = pv[np.isin(site, keep_sites)].reset_index(drop=True)
age = pv["AGE_AT_SCAN"].values.astype(float)
sx = pv["SEX"].values.astype(int)   # ABIDE PCP coding: 1=male, 2=female
pf = 100 * float((sx == 2).mean())
print(f"ABIDE: N={len(pv)} sites={len(keep_sites)} age {age.mean():.1f}±{age.std():.1f} "
      f"female(SEX==2) {pf:.1f}%  ASD {100*float((pv['DX_GROUP']==1).mean()):.1f}%", flush=True)
out["abide"] = {"n": int(len(pv)), "n_sites": int(len(keep_sites)),
                "age_mean": round(float(age.mean()), 1), "age_sd": round(float(age.std()), 1),
                "pct_female": round(pf, 1),
                "coding": "PCP SEX 1=male 2=female; sanity: ASD cohorts ~85% male",
                "matches_loader_n": bool(len(pv) == 871)}
json.dump(out, open(_resolve_path("<REPO_ROOT>/misfit/results/provenance/ukbb_abide_demo_verification.json"), "w"), indent=2)
print("wrote ukbb_abide_demo_verification.json", flush=True)
