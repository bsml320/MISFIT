"""End-to-end smoke on a synthetic cohort in a sandbox directory: banks ->
transport -> features -> calibration freeze -> firewall -> targets -> locked
evaluation. Catches integration bugs (paths, keys, alignment) without touching
real cohorts or the real study tree. Engineering only."""
from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

import json
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, _resolve_path("<REPO_ROOT>/misfit/crt_study/code"))
from misfit.crt_study.code import calib
from misfit.crt_study.code import cohorts
from misfit.crt_study.code import driver
from misfit.crt_study.code import features as feats
from misfit.crt_study.code import train
from misfit.crt_study.code import transport

SAND = tempfile.mkdtemp(prefix="crt_smoke_")
for sub in ("protocol", "runs", "feats", "conf", "dev", "results"):
    os.makedirs(f"{SAND}/{sub}", exist_ok=True)

# tiny synthetic cohort: 3 envs, 90 subjects, 16^3 volumes
rng = np.random.RandomState(0)
N = 90
envs = np.array(["E1", "E2", "E3"])[rng.randint(0, 3, N)]
y = rng.randint(0, 2, N)
df = pd.DataFrame({
    "uid": [f"s{i:03d}" for i in range(N)], "env": envs, "vol_path": "none",
    "label": y, "probe": (y == 1), "age": 70 + 5 * rng.randn(N),
    "sex_male1": rng.randint(0, 2, N)})


class FakeSpec:
    name = "fake"
    df = df
    env_order = ["E1", "E2", "E3"]
    eligible_excl = ["E1", "E2", "E3"]
    primary = [g for g in ["E1", "E2", "E3"]
               if ((df.probe) & (df.env == g)).sum() >= 5]

    def containing(self, g):
        return [e for e in self.eligible_excl if e != g]


spec = FakeSpec()
X = torch.tensor(rng.randn(N, 1, 32, 32, 32).astype(np.float32))
X[y == 1] += 0.15 * torch.tensor(rng.randn(1, 32, 32, 32).astype(np.float32))

# patch study paths and scale
driver.STUDY = SAND
feats.STUDY = SAND
driver.DEV_REPS, driver.CONF_REPS = [0], [1]
train.EPOCHS, train.CKPTS, train.BS = 3, (1, 2, 3), 16
cohorts.LOADERS["fake"] = lambda: spec

cfg = train.cfg_dict("fake", 0.0, 1.0, (32, 32, 32))
json.dump(cfg, open(f"{SAND}/protocol/cfg_fake.json", "w"))
chash = train.cfg_hash(cfg)

damp = (0.1, 1.0)
for rep in (0, 1):
    driver.run_source_bank(spec, X, chash, rep)
    driver.run_transport(spec, X, chash, rep, damp if rep == 0 else (1.0,))

# dev: table, calibrators
t0 = feats.source_table(spec, 0, chash, damp)
driver.run_target_bank(spec, X, chash, 0, conf=False)
t0 = feats.attach_outcomes(t0, spec, 0, chash)
t0["a"] = t0["a_1.0"]
models = {"M0": None}
for m in ("M1", "M2", "M3"):
    models[m] = calib.tune_and_fit(t0, calib.MODEL_COLS[m])
print("dev CV:", {m: (models[m]["cv"] if models[m] else None) for m in models})

# conf: predictions -> firewall -> targets -> evaluate-style checks
t1 = feats.source_table(spec, 1, chash, (1.0,))
t1["a"] = t1["a_1.0"]
qhat = {m: calib.predict(models.get(m), t1) for m in ("M0", "M1", "M2", "M3")}
pp = f"{SAND}/conf/predictions_fake_rep1.npz"
np.savez(pp, uid=t1.uid.to_numpy().astype(str), row=t1.row.to_numpy(),
         **{f"qhat_{m}": v for m, v in qhat.items()})
driver.write_firewall("fake", 1, extra_files=[pp])
try:
    driver.run_target_bank(spec, X, chash, 2, conf=True)
    raise SystemExit("FAIL: conf target ran without firewall for rep 2")
except RuntimeError:
    pass
driver.run_target_bank(spec, X, chash, 1, conf=True)
driver.verify_firewall("fake", 1)
t1 = feats.attach_outcomes(t1, spec, 1, chash)
for m in ("M0", "M1", "M2", "M3"):
    assert np.allclose(qhat[m], calib.predict(models.get(m), t1), atol=1e-12)

from misfit.crt_study.code import analysis
resid = {m: t1.q.to_numpy(float) - qhat[m] for m in qhat}
E = analysis.summarize(analysis.endpoint(resid, t1.env.to_numpy(),
                                         t1.rep.to_numpy(), spec.primary, [1]))
boot = analysis.boot_participants(t1, resid, spec.primary, [1], B=50)
print("E:", {k: round(v, 4) for k, v in E.items()})
print("boot M2 ci:", boot["M2"]["ci95"])
rank = analysis.rank_metrics(t1.q.to_numpy(float), qhat["M3"],
                             t1.L.to_numpy(float), t1.env.to_numpy(),
                             t1.rep.to_numpy(), spec.primary, [1])
print("rank:", {k: round(v, 3) for k, v in rank.items()})
print("SMOKE OK")
shutil.rmtree(SAND)
