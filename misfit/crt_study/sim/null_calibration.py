"""Null-calibration gate (protocol section 7; statistical review F9/F14).

Runs AFTER the development stage and BEFORE any confirmation outcome exists.
From the OASIS-3 development tables: participant-blocked CV residuals of the
frozen-family M2, and the M3-minus-M2 CV prediction gap delta (pure fitted
noise under the null). Per environment, decompose M2 residuals into participant
effects m_i and repetition noise e_ik. Simulate confirmation-shaped data
(cells 107/40/22/16, 5 reps) by resampling participants and their own rep
noises (tails preserved):
  u2_ik = m_i + e_ik                     (M2 residuals)
  u3_ik = u2_ik - delta_i - g_i          (M3 residuals; delta_i resampled,
                                          centered per env; g_i = lam_g m_i
                                          injects relative reduction f)
For f in {0, .05, .10, .15}: >= 500 simulations, each evaluated with the
PRODUCTION bootstrap (analysis.boot_participants, B = 2000, percentile).
Acceptance: f = 0 rejection rate <= 3.5%. Engineering seeds >= 990000.
"""
from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, _resolve_path("<REPO_ROOT>/misfit/crt_study/code"))
from misfit.crt_study.code import analysis                              # noqa: E402
from misfit.crt_study.code import calib                                 # noqa: E402
from misfit.crt_study.code import cohorts                               # noqa: E402
from misfit.crt_study.code import driver                                # noqa: E402

STUDY = driver.STUDY
CONF_N = {"TrioTim": 107, "Biograph_mMR": 40, "Vision": 22, "MAGNETOM_Vida": 16}
N_SIM, F_GRID = 500, (0.0, 0.05, 0.10, 0.15)
N_REPS = 5


def dev_components():
    from misfit.crt_study.code.run_dev import dev_table
    fz = json.load(open(f"{STUDY}/dev/frozen_calibrators_oasis3.json"))
    spec, chash = driver.spec_and_hash("oasis3")
    split = cohorts.split_probes(spec)
    dev_uids = set(split.loc[split.role == "dev", "uid"])
    from misfit.crt_study.code import transport
    tbl = dev_table(spec, chash, transport.DAMP_GRID)
    tbl = tbl[tbl.uid.isin(dev_uids)].reset_index(drop=True)
    tbl["a"] = tbl[f"a_{fz['damp_c']}"]
    p2 = calib.cv_predictions(tbl, calib.MODEL_COLS["M2"],
                              lam=fz["models"]["M2"]["lam"])
    p3 = calib.cv_predictions(tbl, calib.MODEL_COLS["M3"],
                              lam=fz["models"]["M3"]["lam"])
    q = tbl.q.to_numpy(float)
    out = {}
    for g in spec.primary:
        sub = tbl.env.to_numpy() == g
        d = pd.DataFrame({"uid": tbl.uid.to_numpy()[sub],
                          "u2": q[sub] - p2[sub],
                          "delta": p3[sub] - p2[sub]})
        gm = d.groupby("uid").agg(m=("u2", "mean"), dmean=("delta", "mean"))
        e = d.u2.to_numpy() - gm.m.reindex(d.uid).to_numpy()
        out[g] = {"m": gm.m.to_numpy() - gm.m.to_numpy().mean(),
                  "delta": gm.dmean.to_numpy() - gm.dmean.to_numpy().mean(),
                  "e_by_uid": {u: d.u2.to_numpy()[(d.uid == u).to_numpy()]
                               - gm.m[u] for u in gm.index}}
        out[g]["uids"] = list(gm.index)
    return out


def one_sim(args):
    comp, f, seed = args
    rng = np.random.RandomState(seed)
    rows = []
    for g, n_g in CONF_N.items():
        c = comp[g]
        var_m = np.var(c["m"]) + 1e-12
        var_e = np.mean([np.var(v) for v in c["e_by_uid"].values()]) + 1e-12
        lam_g = 1.0 - np.sqrt(max(0.0, 1.0 - f * (var_m + var_e) / var_m))
        pick = rng.randint(0, len(c["uids"]), n_g)
        # NULL DECOUPLING: the M3-minus-M2 prediction gap delta is drawn
        # INDEPENDENTLY of the participant effect (a dev-fitted gap carries the
        # dev-estimated signal; under the null it must be noise with the same
        # marginal). The injected alternative signal enters only through g_i.
        dpick = rng.randint(0, len(c["uids"]), n_g)
        for j, pi in enumerate(pick):
            u = c["uids"][pi]
            m_i, d_i = c["m"][pi], c["delta"][dpick[j]]
            e_pool = c["e_by_uid"][u]
            for k in range(N_REPS):
                e_ik = e_pool[rng.randint(0, len(e_pool))]
                u2 = m_i + e_ik
                u3 = u2 - d_i - lam_g * m_i
                rows.append((f"{g}_{j}", g, k, u2, u3))
    df = pd.DataFrame(rows, columns=["uid", "env", "rep", "u2", "u3"])
    resid = {"M2": df.u2.to_numpy(), "M3": df.u3.to_numpy()}
    envs, reps = list(CONF_N), list(range(N_REPS))
    boot = analysis.boot_participants(df, resid, envs, reps, B=2000, seed=seed + 1)
    e2 = np.nanmean(analysis.endpoint(resid, df.env.to_numpy(),
                                      df.rep.to_numpy(), envs, reps)["M2"])
    e3 = np.nanmean(analysis.endpoint(resid, df.env.to_numpy(),
                                      df.rep.to_numpy(), envs, reps)["M3"])
    return (boot["M2"]["ci95"][0] > 0, (e2 - e3) / e2)


def main():
    comp = dev_components()
    res = {}
    with Pool(32) as pool:
        for f in F_GRID:
            args = [(comp, f, 990100 + i + int(f * 10000) * 1000)
                    for i in range(N_SIM)]
            out = pool.map(one_sim, args)
            rej = float(np.mean([o[0] for o in out]))
            rel = float(np.mean([o[1] for o in out]))
            res[str(f)] = {"reject_rate": rej, "mean_realized_rel": rel}
            print(f"f={f}: reject={rej:.3f} realized_rel={rel:.3f}", flush=True)
    res["acceptance_f0_leq_0.035"] = bool(res["0.0"]["reject_rate"] <= 0.035)
    json.dump(res, open(f"{STUDY}/sim/null_calibration_results.json", "w"),
              indent=2)
    print("wrote sim/null_calibration_results.json")


if __name__ == "__main__":
    main()
