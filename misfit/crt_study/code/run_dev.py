"""CRT development stage: dev banks (reps 0-2), damping and calibrator tuning,
freeze. Usage:
  python run_dev.py banks <cohort> [<cohort> ...]   # train + features + targets
  python run_dev.py freeze <cohort> [...]           # tune on dev, freeze models
Scientific fits are gated on protocol/PROTOCOL_FROZEN.txt.
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from misfit.crt_study.code import calib                                 # noqa: E402
from misfit.crt_study.code import cohorts                               # noqa: E402
from misfit.crt_study.code import driver                                # noqa: E402
from misfit.crt_study.code import features as feats                     # noqa: E402
from misfit.crt_study.code import transport                             # noqa: E402

STUDY = driver.STUDY
gate = driver.gate_frozen


def dev_table(spec, chash, damp_cs):
    tabs = []
    for rep in driver.DEV_REPS:
        t = feats.source_table(spec, rep, chash, damp_cs)
        t = feats.attach_outcomes(t, spec, rep, chash)
        tabs.append(t)
    return pd.concat(tabs, ignore_index=True)


def banks(names):
    gate()
    for cn in names:
        spec, X, chash = driver.prepare(cn)
        split = cohorts.split_probes(spec)
        md = cohorts.manifest_digest(split)
        print(f"{cn}: N={len(spec.df)} probes={int(spec.df.probe.sum())} "
              f"primary={spec.primary} digest={md[:16]}", flush=True)
        for rep in driver.DEV_REPS:
            driver.run_source_bank(spec, X, chash, rep)
            driver.run_transport(spec, X, chash, rep, transport.DAMP_GRID)
            driver.run_target_bank(spec, X, chash, rep, conf=False)
        del X


def freeze(names):
    gate()
    out_all = {}
    for cn in names:
        spec, chash = driver.spec_and_hash(cn)
        split = cohorts.split_probes(spec)
        dev_uids = set(split.loc[split.role == "dev", "uid"])
        tbl = dev_table(spec, chash, transport.DAMP_GRID)
        tbl = tbl[tbl.uid.isin(dev_uids)].reset_index(drop=True)
        # damping selection: best M3 CV over the prespecified grid
        cv_by_c = {}
        for c in transport.DAMP_GRID:
            t = tbl.copy()
            t["a"] = t[f"a_{c}"]
            bases = {col: calib.fit_basis(t[col].to_numpy(float))
                     for col in calib.MODEL_COLS["M3"]}
            lam_best = min(calib.RIDGE_GRID,
                           key=lambda l: calib.cv_score(t, calib.MODEL_COLS["M3"], bases, l))
            cv_by_c[c] = calib.cv_score(t, calib.MODEL_COLS["M3"], bases, lam_best)
        c_star = min(cv_by_c, key=cv_by_c.get)
        tbl["a"] = tbl[f"a_{c_star}"]
        models = {"M0": None}
        for m in ("M1", "M2", "M3"):
            models[m] = calib.tune_and_fit(tbl, calib.MODEL_COLS[m])
        frozen = {"cohort": cn, "cfg_hash": chash, "damp_c": c_star,
                  "split_digest": cohorts.manifest_digest(split),
                  "n_dev_rows": int(len(tbl)),
                  "cv_by_damping": {str(k): v for k, v in cv_by_c.items()},
                  "dev_cv": {m: (models[m]["cv"] if models[m] else None)
                             for m in models},
                  "models": {m: models[m] for m in ("M1", "M2", "M3")}}
        fp = f"{STUDY}/dev/frozen_calibrators_{cn}.json"
        driver.json_atomic(fp, frozen)
        out_all[cn] = {"damp_c": c_star, "cv_by_damping": cv_by_c,
                       "dev_cv": frozen["dev_cv"]}
        print(f"{cn}: damp_c={c_star} dev CV: " +
              " ".join(f"{m}={v:.5f}" for m, v in frozen["dev_cv"].items()
                       if v is not None), flush=True)
    driver.json_atomic(f"{STUDY}/dev/dev_summary.json", out_all)


if __name__ == "__main__":
    cmd, names = sys.argv[1], sys.argv[2:]
    {"banks": banks, "freeze": freeze}[cmd](names)
