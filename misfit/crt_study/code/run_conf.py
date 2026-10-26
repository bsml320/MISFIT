"""CRT confirmation stage. Subcommands:
  sources <cohort> [...]  : conf source banks + transport at the frozen damping,
                            frozen-model predictions, firewall manifest
  targets <cohort> [...]  : conf target banks (require the firewall manifest)
  evaluate <cohort> [...] : the locked analysis, run once; writes results/
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from misfit.crt_study.code import analysis                              # noqa: E402
from misfit.crt_study.code import calib                                 # noqa: E402
from misfit.crt_study.code import cohorts                               # noqa: E402
from misfit.crt_study.code import driver                                # noqa: E402
from misfit.crt_study.code import features as feats                     # noqa: E402

STUDY = driver.STUDY


def frozen_for(cn):
    fp = f"{STUDY}/dev/frozen_calibrators_{cn}.json"
    if not os.path.exists(fp):
        raise RuntimeError(f"{cn}: no frozen calibrators; confirmation locked")
    return json.load(open(fp))


def pred_path(cn, rep):
    return f"{STUDY}/conf/predictions_{cn}_rep{rep}.npz"


def conf_table(spec, rep, chash, c_star, with_q):
    t = feats.source_table(spec, rep, chash, (c_star,))
    t["a"] = t[f"a_{c_star}"]
    if with_q:
        t = feats.attach_outcomes(t, spec, rep, chash)
    return t


def check_split(spec, fz):
    split = cohorts.split_probes(spec)
    if cohorts.manifest_digest(split) != fz["split_digest"]:
        raise RuntimeError(f"{spec.name}: split digest drifted since freeze")
    return split


def sources(names):
    from misfit.crt_study.code import transport
    driver.gate_frozen()
    for cn in names:
        fz = frozen_for(cn)
        spec, X, chash = driver.prepare(cn)
        if chash != fz["cfg_hash"]:
            raise RuntimeError(f"{cn}: config hash drifted since freeze")
        check_split(spec, fz)
        c_star = fz["damp_c"]
        for rep in driver.CONF_REPS:
            driver.run_source_bank(spec, X, chash, rep)
            driver.run_transport(spec, X, chash, rep, (c_star,))
            pp = pred_path(cn, rep)
            if not os.path.exists(pp):
                t = conf_table(spec, rep, chash, c_star, with_q=False)
                qhat = {m: calib.predict(fz["models"].get(m), t)
                        for m in ("M0", "M1", "M2", "M3")}
                tmp = pp + ".tmp.npz"
                np.savez(tmp, uid=t.uid.to_numpy().astype(str),
                         env=t.env.to_numpy().astype(str),
                         row=t.row.to_numpy(), rep=np.array(rep),
                         cfg_hash=np.array(chash),
                         **{f"qhat_{m}": v for m, v in qhat.items()},
                         **{c: t[c].to_numpy(float) for c in
                            ("L", "ce_first", "aum_neg", "el2n", "ce_std",
                             "age", "b", "a")})
                os.replace(tmp, pp)
        for rep in driver.CONF_REPS:
            fw = f"{STUDY}/conf/FIREWALL_{cn}_rep{rep}.json"
            if not os.path.exists(fw):
                driver.write_firewall(cn, rep, extra_files=[pred_path(cn, rep)])
                print(f"firewall written: {cn} rep {rep}", flush=True)
        del X


def targets(names):
    driver.gate_frozen()
    for cn in names:
        fz = frozen_for(cn)
        spec, X, chash = driver.prepare(cn)
        if chash != fz["cfg_hash"]:
            raise RuntimeError(f"{cn}: config hash drifted since freeze")
        check_split(spec, fz)
        for rep in driver.CONF_REPS:
            driver.run_target_bank(spec, X, chash, rep, conf=True)
        del X


def evaluate(names):
    driver.gate_frozen()
    for cn in names:
        out = f"{STUDY}/results/{cn}_confirmation.json"
        if os.path.exists(out):
            raise RuntimeError(f"{out} exists: confirmation already run")
        fz = frozen_for(cn)
        spec, chash = driver.spec_and_hash(cn)
        c_star = fz["damp_c"]
        split = check_split(spec, fz)
        conf_uids = set(split.loc[split.role == "conf", "uid"])
        tabs = []
        for rep in driver.CONF_REPS:
            driver.verify_firewall(cn, rep)
            t = conf_table(spec, rep, chash, c_star, with_q=True)
            with np.load(pred_path(cn, rep), allow_pickle=False) as pr:
                if not (np.array_equal(pr["uid"].astype(str),
                                       t.uid.to_numpy().astype(str))
                        and np.array_equal(pr["row"], t.row.to_numpy())):
                    raise RuntimeError("prediction rows misaligned with table")
                for m in ("M0", "M1", "M2", "M3"):
                    if not np.allclose(pr[f"qhat_{m}"],
                                       calib.predict(fz["models"].get(m), t),
                                       atol=1e-10):
                        raise RuntimeError(f"{cn} rep{rep}: frozen prediction "
                                           f"mismatch for {m} (post-firewall change?)")
                    t[f"qhat_{m}"] = pr[f"qhat_{m}"]
            tabs.append(t)
        tbl = pd.concat(tabs, ignore_index=True)
        tbl = tbl[tbl.uid.isin(conf_uids)].reset_index(drop=True)
        env = tbl.env.to_numpy()
        rep = tbl.rep.to_numpy()
        q = tbl.q.to_numpy(float)
        resid = {m: q - tbl[f"qhat_{m}"].to_numpy(float)
                 for m in ("M0", "M1", "M2", "M3")}
        resid_raw = q - (tbl.L.to_numpy(float) + tbl.a.to_numpy(float))
        envs, reps = spec.primary, driver.CONF_REPS
        tabs_E = analysis.endpoint(resid, env, rep, envs, reps)
        E = analysis.summarize(tabs_E)
        E["raw_transport"] = float(np.nanmean(
            analysis.endpoint({"r": resid_raw}, env, rep, envs, reps)["r"]))
        boot = analysis.boot_participants(tbl, resid, envs, reps)
        boot_x = analysis.boot_participants(tbl, resid, envs, reps,
                                            also_reps=True, seed=777)
        for m in boot:
            boot[m]["est"] = E[m] - E["M3"]
            boot_x[m]["est"] = E[m] - E["M3"]
        rel = {m: (E[m] - E["M3"]) / E[m] if E[m] > 0 else float("nan")
               for m in ("M0", "M1", "M2")}
        d_src = tbl.L.to_numpy(float)
        rank = {m: analysis.rank_metrics(q, tbl[f"qhat_{m}"].to_numpy(float),
                                         d_src, env, rep, envs, reps)
                for m in ("M0", "M1", "M2", "M3")}
        support = {
            "primary_M3_beats_M2": bool(boot["M2"]["ci95"][0] > 0),
            "conjunction_all_controls": bool(all(boot[m]["ci95"][0] > 0
                                                 for m in ("M0", "M1", "M2"))),
            "relative_improvement_vs_M2": rel["M2"],
            "practical_benchmark_5pct": bool(rel["M2"] >= 0.05)}
        res = {
            "cohort": cn, "stage": "confirmation (locked, run once)",
            "n_conf_probes": int(tbl.uid.nunique()), "n_rows": int(len(tbl)),
            "primary_envs": envs, "conf_reps": reps, "damp_c": c_star,
            "E": E, "delta_vs_M3": {m: E[m] - E["M3"] for m in ("M0", "M1", "M2")},
            "relative": rel,
            "participant_block_ci": boot,
            "crossed_ci_fewblock_caveat": boot_x,
            "per_rep_delta": analysis.per_rep_delta(resid, env, rep, envs, reps),
            "per_env_E": {m: {g: [None if np.isnan(x) else float(x)
                                  for x in tabs_E[m][gi]]
                              for gi, g in enumerate(envs)} for m in tabs_E},
            "influence_loo_env": analysis.influence_loo_env(resid, env, rep,
                                                            envs, reps),
            "rank_metrics": rank,
            "support": support}
        driver.json_atomic(out, res)
        tbl.to_csv(f"{STUDY}/results/{cn}_conf_rows_local.csv", index=False)
        print(json.dumps({"cohort": cn, "E": E, "support": support}, indent=2,
                         default=float), flush=True)


if __name__ == "__main__":
    cmd, names = sys.argv[1], sys.argv[2:]
    {"sources": sources, "targets": targets, "evaluate": evaluate}[cmd](names)
