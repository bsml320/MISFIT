"""Engineering pilot: cohort counts, one timed OASIS-3 fit, one timed
per-sample-gradient pass, one timed Woodbury solve set. Pilot artifacts live
under pilot/ with engineering seeds and are never reused scientifically."""
from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from misfit.crt_study.code import cohorts
from misfit.crt_study.code import train
from misfit.crt_study.code import transport

STUDY = _resolve_path("<REPO_ROOT>/misfit/crt_study")
rep = {}
for directory in ("manifests", "pilot"):
    os.makedirs(f"{STUDY}/{directory}", exist_ok=True)

for cn in ("oasis3", "adni", "oasis1"):
    spec = cohorts.LOADERS[cn]()
    split = cohorts.split_probes(spec)
    probe = spec.df[spec.df.probe]
    rep[cn] = {
        "N": len(spec.df), "n_label1": int(spec.df.label.sum()),
        "envs": {g: int((spec.df.env == g).sum()) for g in spec.env_order},
        "probes_per_env": {g: int((probe.env == g).sum()) for g in spec.env_order},
        "eligible_excl": spec.eligible_excl, "primary": spec.primary,
        "dev": split[split.role == "dev"].groupby("env").size().to_dict(),
        "conf": split[split.role == "conf"].groupby("env").size().to_dict(),
        "split_digest": cohorts.manifest_digest(split)}
    split.to_csv(f"{STUDY}/manifests/{cn}_split_manifest_local.csv", index=False)
    print(cn, json.dumps(rep[cn], indent=1, default=str)[:600], flush=True)

spec = cohorts.LOADERS["oasis3"]()
t0 = time.time()
X = train.load_volumes(spec.df.vol_path)
rep["load_volumes_s"] = round(time.time() - t0, 1)
mu, sd = train.norm_constants(X)
X = train.apply_norm(X, mu, sd)
rep["oasis3_norm"] = [mu, sd]
rep["X_gb"] = round(X.element_size() * X.nelement() / 1e9, 2)

y = spec.df.label.to_numpy()
env = spec.df.env.to_numpy()
tr_idx = np.where(env != "Sonata")[0]        # largest training set
t0 = time.time()
state, traces, logits = train.fit(X, y, tr_idx, seed=990001)
fit_s = time.time() - t0
rep["fit_wall_s_traces"] = round(fit_s, 1)
rep["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)

t0 = time.time()
res = transport.endpoint_features(spec, state, X, y, env, "Sonata")
rep["endpoint_wall_s"] = round(time.time() - t0, 1)
rep["cond"] = res["cond"]
rep["resid_max"] = max(max(t["resid"].values()) for t in res["targets"].values())
rep["trG_over_P"] = res["trG"] / res["P"]
rep["P"] = res["P"]
rep["gpu_peak_gb_transport"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)

n_fits = {"oasis3": 8 * (5 + 4), "adni": 8 * (len(cohorts.LOADERS['adni']().eligible_excl)
                                              + len(cohorts.LOADERS['adni']().primary))}
rep["estimate_note"] = ("oasis3 fit+traces %.0fs, endpoint %.0fs; ADNI/OASIS-1 "
                        "scale by N" % (fit_s, rep["endpoint_wall_s"]))
json.dump(rep, open(f"{STUDY}/pilot/pilot_report.json", "w"), indent=2, default=str)
print(json.dumps({k: v for k, v in rep.items() if not isinstance(v, dict)},
                 indent=2, default=str), flush=True)
print("wrote pilot/pilot_report.json")
