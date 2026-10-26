"""CRT study: cohort construction for OASIS-3, ADNI and OASIS-1.

Returns a common CohortSpec (pandas frame + conventions) per cohort:
  columns: uid, vol_path, label (0/1), env (string), probe (bool: primary
  probe-stratum member), age, sex_male1
  env_order: environments in frozen descending-size order
  eligible_excl: environments whose exclusion fit is in the source bank
  primary: target environments (probe cells with >= MIN_CELL probes)

Clinical rules preserved from the established cohorts, with two documented
deviations required by the experiment directive (see protocol):
  - ADNI is deduplicated to one earliest-phase scan per participant
    (established MedIA cohort is scan-level, N=731 over 547 participants).
  - OASIS-1 uses the loader's cohort (CDR present, N=226, age-quartile
    environments) but volumes directly, not parcel features.

Identifiers stay on this server (OASIS-3 and ADNI DUAs; OASIS-1 DUA).
"""
from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

import glob
import hashlib
import os
import re

import numpy as np
import pandas as pd

ROOT = _resolve_path("<REPO_ROOT>")
TCS_MANIFEST = f"{ROOT}/misfit/results/provenance/tcs_feasibility_manifest_local.csv"
OASIS3_WORK = _resolve_path("<REPO_ROOT>/data/oasis3/adnitpl_work")
ADNI_NII = _resolve_path("<REPO_ROOT>/data/adni/smri/nii")
ADNI_CSV = _resolve_path("<REPO_ROOT>/data/adni/smri/labels/combined.csv")
OASIS1_DIR = _resolve_path("<NILEARN_DATA>/oasis1")
OASIS1_CSV = f"{OASIS1_DIR}/oasis_cross-sectional.csv"

MIN_CELL = 8           # minimum probes per primary target cell (prespecified)
ADNI_MIN_ENV = 30      # established environment threshold, applied post-dedup
SPLIT_SEED = 20261018

ADNI_PHASE_ORDER = {"ADNIGO": 0, "ADNI2": 1, "ADNI3": 2}
ADNI_IMPAIRED = {"AD", "MCI", "EMCI", "LMCI"}
ADNI_PROBE = {"EMCI", "MCI", "LMCI"}       # MCI spectrum ~ CDR 0.5 at entry
ADNI_EXCLUDED_GROUPS = {"SMC", "Patient"}  # established exclusion


def stream_seed(purpose, *parts):
    key = "|".join([purpose] + [str(p) for p in parts])
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % (2**31 - 1)


def sha16(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


class CohortSpec:
    def __init__(self, name, df, env_order, eligible_excl, primary):
        if df.uid.duplicated().any():
            raise ValueError(f"{name}: duplicate participants")
        for p in df.vol_path:
            if not os.path.exists(p):
                raise FileNotFoundError(p)
        self.name, self.df = name, df.reset_index(drop=True)
        self.env_order, self.eligible_excl, self.primary = env_order, eligible_excl, primary
        sanitized = [e.replace(" ", "_") for e in env_order]
        if len(set(sanitized)) != len(env_order):
            raise ValueError(f"{name}: sanitized environment names collide")
        for g in primary:
            n = int((df.probe & (df.env == g)).sum())
            if n < MIN_CELL:
                raise ValueError(f"{name}/{g}: only {n} probes (< {MIN_CELL})")

    def containing(self, g):
        """Eligible containing-refit exclusions for probes of environment g."""
        return [e for e in self.eligible_excl if e != g]


def oasis3():
    rec = pd.read_csv(TCS_MANIFEST)
    need = {"subject", "session", "scanner", "cdr_stage_0_05_1plus",
            "label_impaired1", "age", "sex_male1"}
    if need - set(rec.columns):
        raise ValueError("TCS manifest missing columns")
    df = pd.DataFrame({
        "uid": rec.subject, "env": rec.scanner,
        "vol_path": [f"{OASIS3_WORK}/{s}/smwc1pT1.nii" for s in rec.session],
        "label": rec.label_impaired1.astype(int),
        "probe": rec.cdr_stage_0_05_1plus.to_numpy() == 1,
        "age": rec.age.astype(float), "sex_male1": rec.sex_male1.astype(int)})
    env_order = ["TrioTim", "Biograph_mMR", "Vision", "MAGNETOM_Vida", "Sonata", "Avanto"]
    return CohortSpec("oasis3", df, env_order,
                      eligible_excl=env_order[:5], primary=env_order[:4])


def adni():
    csv = pd.read_csv(ADNI_CSV)
    first = csv.groupby(["dataset_id", "file_name"], sort=True).first().reset_index()
    conflict = csv.groupby(["dataset_id", "file_name"]).Group.nunique()
    if (conflict > 1).any():
        raise ValueError("ADNI label conflict within (protocol, PTID)")
    rows = []
    for _, r in first.iterrows():
        if r.Group in ADNI_EXCLUDED_GROUPS or r.Group not in ADNI_IMPAIRED | {"CN"}:
            continue
        p = f"{ADNI_NII}/{r.dataset_id}/{r.file_name}/smwc1pT1.nii"
        if not os.path.exists(p):
            continue
        phase = r.dataset_id.split("-")[0]
        rows.append({"uid": r.file_name, "env": r.dataset_id, "vol_path": p,
                     "label": int(r.Group in ADNI_IMPAIRED),
                     "probe": r.Group in ADNI_PROBE,
                     "age": float(r.Age), "sex_male1": int(str(r.Sex).startswith("M")),
                     "_phase": ADNI_PHASE_ORDER[phase]})
    df = pd.DataFrame(rows)
    # directive-required participant-level dedup: earliest phase, then protocol name
    df = (df.sort_values(["uid", "_phase", "env"])
            .drop_duplicates("uid", keep="first").drop(columns="_phase"))
    sizes = df.groupby("env").size()
    keep_env = sizes[sizes >= ADNI_MIN_ENV].index
    df = df[df.env.isin(keep_env)].reset_index(drop=True)
    env_order = list(df.env.value_counts().index)        # descending size
    primary = [g for g in env_order
               if (df.probe & (df.env == g)).sum() >= MIN_CELL]
    return CohortSpec("adni", df, env_order,
                      eligible_excl=env_order, primary=primary)


def oasis1():
    meta = pd.read_csv(OASIS1_CSV)
    vols = sorted(glob.glob(f"{OASIS1_DIR}/OAS1_*_MR1/mwrc1*.nii.gz"))
    have = {re.search(r"(OAS1_\d+_MR1)", v).group(1): v for v in vols}
    # replicate the established loader: nilearn's released subject list, CDR
    # present only, then age quartiles over the included subjects
    from nilearn.datasets import fetch_oasis_vbm
    rel = fetch_oasis_vbm(data_dir=os.path.dirname(OASIS1_DIR))
    rel_ids = [re.search(r"(OAS1_\d+_MR1)", p).group(1) for p in rel.gray_matter_maps]
    meta = meta[meta.ID.isin(rel_ids) & meta.CDR.notna() & meta.ID.isin(have)]
    meta = meta.sort_values("ID").reset_index(drop=True)
    qs = np.quantile(meta.Age.to_numpy(), [0.25, 0.5, 0.75])
    env = np.digitize(meta.Age.to_numpy(), qs)           # 0..3 as loader does
    df = pd.DataFrame({
        "uid": meta.ID, "env": [f"ageQ{int(q) + 1}" for q in env],
        "vol_path": [have[i] for i in meta.ID],
        "label": (meta.CDR.to_numpy() > 0).astype(int),
        "probe": meta.CDR.to_numpy() == 0.5,
        "age": meta.Age.astype(float).to_numpy(),
        "sex_male1": (meta["M/F"] == "M").astype(int).to_numpy()})
    env_order = list(df.env.value_counts().index)
    primary = [g for g in env_order
               if (df.probe & (df.env == g)).sum() >= MIN_CELL]
    return CohortSpec("oasis1", df, env_order,
                      eligible_excl=env_order, primary=primary)


LOADERS = {"oasis3": oasis3, "adni": adni, "oasis1": oasis1}


def split_probes(spec, seed=SPLIT_SEED):
    """Deterministic env-stratified dev/conf split of the probe stratum.
    n_dev = round(n/3) per primary cell. Non-primary-cell probes get no role."""
    role = pd.Series("none", index=spec.df.index)
    for g in spec.primary:
        idx = spec.df.index[spec.df.probe & (spec.df.env == g)].to_numpy()
        idx = idx[np.argsort(spec.df.uid.to_numpy()[idx])]      # stable base order
        rng = np.random.RandomState(stream_seed("crt_split", spec.name, g, seed))
        rng.shuffle(idx)
        n_dev = int(round(len(idx) / 3))
        role.iloc[idx[:n_dev]] = "dev"
        role.iloc[idx[n_dev:]] = "conf"
    out = spec.df.copy()
    out["role"] = role.to_numpy()
    return out


def manifest_digest(df):
    """Order-independent digest over the analysis-relevant columns."""
    key = df.sort_values("uid").apply(
        lambda r: f"{r.uid}|{r.env}|{r.label}|{int(r.probe)}|{r.age:.2f}|"
                  f"{r.sex_male1}|{r.get('role', '')}", axis=1)
    return hashlib.sha256("\n".join(key).encode()).hexdigest()
