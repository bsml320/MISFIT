"""CRT study: assemble per-(probe, repetition) feature/outcome tables.

All feature columns are source-bank quantities (final-checkpoint CE, AUM,
EL2N, refit variability, demographics, transport features a and b); the outcome
q is the probe's final CE under that repetition's fresh target fit excluding
the probe's own environment. Features never see target fits; the assembler
reads them from different files and the firewall manifest (confirmation) is
hashed before any target fit exists.
"""
from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

import os

import numpy as np
import pandas as pd

STUDY = _resolve_path("<REPO_ROOT>/misfit/crt_study")


def fit_path(cohort, rep, kind, env):
    return f"{STUDY}/runs/{cohort}/rep{rep}/{kind}_{env.replace(' ', '_')}.npz"


def feat_path(cohort, rep, env):
    return f"{STUDY}/feats/{cohort}/rep{rep}/src_{env.replace(' ', '_')}.npz"


def load_src(cohort, rep, env, chash):
    d = np.load(fit_path(cohort, rep, "src", env), allow_pickle=False)
    if str(d["cfg_hash"]) != chash:
        raise RuntimeError("source fit config hash mismatch")
    return d


def final_ce(d):
    """Final-checkpoint per-sample CE from a source fit (largest ce_ep* key)."""
    eps = [int(k[5:]) for k in d.files if k.startswith("ce_ep")]
    return d[f"ce_ep{max(eps)}"]


def ce_from_logits(logits, y):
    z = logits - logits.max(1, keepdims=True)
    lse = np.log(np.exp(z).sum(1))
    return lse - z[np.arange(len(y)), y]


def source_table(spec, rep, chash, damp_cs):
    """Feature rows for every primary-cell probe of one repetition."""
    df = spec.df
    src = {e: load_src(spec.name, rep, e, chash) for e in spec.eligible_excl}
    rows = []
    for g in spec.primary:
        C_g = spec.containing(g)
        pidx = np.where(df.probe.to_numpy() & (df.env.to_numpy() == g))[0]
        ce = np.stack([final_ce(src[e])[pidx] for e in C_g])
        aum, el2n = [], []
        for e in C_g:
            tm = src[e]["train_mask"].astype(bool)
            if not tm[pidx].all():
                raise RuntimeError(f"{g}: probe not in training of exclusion {e}")
            aum.append(src[e]["margins"][:, pidx].mean(0))
            el2n.append(src[e]["el2n"][pidx])
        ab = np.load(f"{STUDY}/feats/{spec.name}/rep{rep}/ab_{g.replace(' ', '_')}.npz",
                     allow_pickle=False)
        if str(ab["cfg_hash"]) != chash:
            raise RuntimeError("transport feature config hash mismatch")
        if not np.array_equal(ab["rows"], pidx):
            raise RuntimeError("transport feature rows misaligned")
        base = pd.DataFrame({
            "uid": df.uid.to_numpy()[pidx], "env": g, "rep": rep,
            "row": pidx,
            "L": ce.mean(0), "ce_first": ce[0], "ce_std": ce.std(0),
            "aum_neg": -np.stack(aum).mean(0), "el2n": np.stack(el2n).mean(0),
            "age": df.age.to_numpy()[pidx],
            "sex_male1": df.sex_male1.to_numpy()[pidx],
            "b": ab["b"]})
        for c in damp_cs:
            base[f"a_{c}"] = ab[f"a_{c}"]
        ab.close()
        rows.append(base)
    for v in src.values():
        v.close()
    return pd.concat(rows, ignore_index=True)


def attach_outcomes(tbl, spec, rep, chash):
    df = spec.df
    y = df.label.to_numpy()
    out = tbl.copy()
    out["q"] = np.nan
    for g in spec.primary:
        p = fit_path(spec.name, rep, "tgt", g)
        with np.load(p, allow_pickle=False) as d:
            if str(d["cfg_hash"]) != chash:
                raise RuntimeError("target fit config hash mismatch")
            tm = d["train_mask"].astype(bool)
            sel = out.env == g
            ridx = out.loc[sel, "row"].to_numpy()
            if tm[ridx].any():
                raise RuntimeError(f"{g}: probe present in its own target training")
            ce = ce_from_logits(d["logits"][ridx], y[ridx])
            out.loc[sel, "q"] = ce
    if out.q.isna().any():
        raise RuntimeError("missing target outcomes")
    return out
