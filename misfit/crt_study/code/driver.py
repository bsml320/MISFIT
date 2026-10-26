"""CRT study driver: banks, transport features, firewall, stage gating.

Order enforced per repetition and cohort:
  1. source bank (eligible exclusion fits, traces on)
  2. transport endpoint features from source checkpoints only
  3. (confirmation only) frozen-calibrator predictions + FIREWALL hash manifest
  4. target bank (primary-environment fresh fits, distinct seeds)
Confirmation target fits REFUSE to run unless the firewall manifest exists;
the locked evaluation refuses to run unless the manifest hashes still match.
Dev repetitions are 0..2, confirmation 3..7 (frozen). Seeds are
purpose-namespaced; source and target banks never share a seed.
"""
from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from misfit.crt_study.code import cohorts                              # noqa: E402
from misfit.crt_study.code import train                                # noqa: E402
from misfit.crt_study.code import transport                            # noqa: E402
from misfit.crt_study.code.features import feat_path, fit_path    # noqa: E402

STUDY = _resolve_path("<REPO_ROOT>/misfit/crt_study")
R_DEV, R_CONF = 3, 5
DEV_REPS = list(range(R_DEV))
CONF_REPS = list(range(R_DEV, R_DEV + R_CONF))


def seed_for(cohort, rep, kind, env):
    return cohorts.stream_seed("crt_fit", cohort, rep, kind, env)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def json_atomic(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    os.replace(tmp, path)


def gate_frozen():
    """Scientific stages require the freeze record AND an unchanged code/.
    Documented-deviation override: CRT_ALLOW_CODE_DRIFT=1 (logged loudly)."""
    fp = f"{STUDY}/protocol/PROTOCOL_FROZEN.txt"
    if not os.path.exists(fp):
        raise RuntimeError("protocol not frozen; scientific fits are locked")
    rec = json.load(open(fp))
    drift = []
    for rel, h in rec["files"].items():
        if not rel.startswith("code/"):
            continue
        p = f"{STUDY}/{rel}"
        cur = hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
        if cur != h:
            drift.append(rel)
    if drift:
        if os.environ.get("CRT_ALLOW_CODE_DRIFT") == "1":
            print(f"WARNING: code drift overridden for {drift} "
                  "(document in DEVIATIONS.md)", flush=True)
        else:
            raise RuntimeError(f"code drifted since freeze: {drift}")


def spec_and_hash(cohort_name):
    """Cohort spec + frozen config hash without loading volumes (analysis side)."""
    spec = cohorts.LOADERS[cohort_name]()
    cfg = json.load(open(f"{STUDY}/protocol/cfg_{cohort_name}.json"))
    return spec, train.cfg_hash(cfg)


def prepare(cohort_name):
    spec = cohorts.LOADERS[cohort_name]()
    X = train.load_volumes(spec.df.vol_path)
    mu, sd = train.norm_constants(X)
    cfgp = f"{STUDY}/protocol/cfg_{cohort_name}.json"
    cfg = train.cfg_dict(cohort_name, mu, sd, tuple(X.shape[2:]))
    if os.path.exists(cfgp):                      # frozen constants win
        cfg = json.load(open(cfgp))
        if list(X.shape[2:]) != cfg["input_shape"]:
            raise RuntimeError(f"{cohort_name}: volume shape "
                               f"{tuple(X.shape[2:])} != frozen "
                               f"{cfg['input_shape']}")
    else:
        json_atomic(cfgp, cfg)
    X = train.apply_norm(X, cfg["norm_mu"], cfg["norm_sd"])
    return spec, X, train.cfg_hash(cfg)


def run_source_bank(spec, X, chash, rep):
    y = spec.df.label.to_numpy()
    env = spec.df.env.to_numpy()
    for e in spec.eligible_excl:
        p = fit_path(spec.name, rep, "src", e)
        if os.path.exists(p):
            continue
        tr_idx = np.where(env != e)[0]
        t0 = time.time()
        state, traces, logits = train.fit(X, y, tr_idx, seed_for(spec.name, rep, "src", e))
        train.save_fit(p, state, traces, logits, (env != e), chash,
                       seed_for(spec.name, rep, "src", e), time.time() - t0)
        log(spec.name, rep, "src", e, time.time() - t0)


def run_transport(spec, X, chash, rep, damp_cs):
    if transport.LAM != train.WD:
        raise RuntimeError("transport.LAM != train.WD: ridge/weight-decay "
                           "identity broken")
    y = spec.df.label.to_numpy()
    env = spec.df.env.to_numpy()
    results = {}
    for e in spec.eligible_excl:
        fp = feat_path(spec.name, rep, e)
        src_p = fit_path(spec.name, rep, "src", e)
        src_sha = sha_file(src_p)
        if os.path.exists(fp):
            with np.load(fp, allow_pickle=False) as d:
                if str(d["cfg_hash"]) != chash:
                    raise RuntimeError(f"{fp}: config hash mismatch on resume")
                if str(d["src_sha"]) != src_sha:
                    raise RuntimeError(f"{fp}: stale features (source fit "
                                       "changed after feature computation)")
                results[e] = _unpack_endpoint(d, spec, damp_cs)
            continue
        _, state = train.load_state(src_p, chash)
        t0 = time.time()
        res = transport.endpoint_features(spec, state, X, y, env, e, damp_cs)
        _save_endpoint(fp, res, chash, src_sha)
        results[e] = res
        log(spec.name, rep, "feat", e, time.time() - t0)
    for g in spec.primary:
        gp = f"{STUDY}/feats/{spec.name}/rep{rep}/ab_{g.replace(' ', '_')}.npz"
        if os.path.exists(gp):
            with np.load(gp, allow_pickle=False) as d:
                if str(d["cfg_hash"]) != chash:
                    raise RuntimeError(f"{gp}: config hash mismatch on resume")
            continue
        arrs = {}
        for c in damp_cs:
            rows, a, b = transport.assemble_ab(spec, results, g, c)
            arrs[f"a_{c}"] = a
            arrs["b"] = b
            arrs["rows"] = rows
        tmp = gp + ".tmp.npz"
        np.savez(tmp, cfg_hash=np.array(chash), **arrs)
        os.replace(tmp, gp)


def _save_endpoint(fp, res, chash, src_sha):
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    arrs = {"trG": np.array(res["trG"]), "P": np.array(res["P"]),
            "cfg_hash": np.array(chash), "src_sha": np.array(src_sha),
            "meta": np.array(json.dumps({"alphas": res["alphas"],
                                         "cond": res["cond"],
                                         "cond_warn": res["cond_warn"],
                                         "stationarity_grad_norm":
                                             res["stationarity_grad_norm"]}))}
    for g, t in res["targets"].items():
        k = g.replace(" ", "_")
        arrs[f"{k}_rows"] = t["probe_rows"]
        arrs[f"{k}_c"] = t["c"]
        arrs[f"{k}_dot_delta"] = t["dot_delta"]
        arrs[f"{k}_delta"] = t["delta"]
        arrs[f"{k}_delta_norm"] = np.array(t["delta_norm"])
        for cc, dv in t["dot_v"].items():
            arrs[f"{k}_dot_v_{cc}"] = dv
            arrs[f"{k}_resid_{cc}"] = np.array(t["resid"][cc])
            arrs[f"{k}_v_{cc}"] = t[f"v_{float(cc)}"]
    tmp = fp + ".tmp.npz"
    np.savez_compressed(tmp, **arrs)
    os.replace(tmp, fp)


def _unpack_endpoint(d, spec, damp_cs):
    res = {"targets": {}}
    for g in spec.primary:
        k = g.replace(" ", "_")
        if f"{k}_rows" not in d.files:
            continue
        res["targets"][g] = {
            "probe_rows": d[f"{k}_rows"], "c": d[f"{k}_c"],
            "dot_delta": d[f"{k}_dot_delta"],
            "dot_v": {str(cc): d[f"{k}_dot_v_{cc}"] for cc in damp_cs}}
    return res


def run_target_bank(spec, X, chash, rep, conf=False):
    if conf:
        fw = f"{STUDY}/conf/FIREWALL_{spec.name}_rep{rep}.json"
        if not os.path.exists(fw):
            raise RuntimeError(f"rep {rep}: firewall manifest missing; conf "
                               "target bank is locked until source features "
                               "are hashed")
        man = json.load(open(fw))
        if man.get("cohort") != spec.name or man.get("rep") != rep \
                or not man.get("files"):
            raise RuntimeError(f"{fw}: malformed firewall manifest")
    y = spec.df.label.to_numpy()
    env = spec.df.env.to_numpy()
    for g in spec.primary:
        p = fit_path(spec.name, rep, "tgt", g)
        if os.path.exists(p):
            continue
        tr_idx = np.where(env != g)[0]
        t0 = time.time()
        state, traces, logits = train.fit(X, y, tr_idx,
                                          seed_for(spec.name, rep, "tgt", g),
                                          want_traces=False)
        train.save_fit(p, state, traces, logits, (env != g), chash,
                       seed_for(spec.name, rep, "tgt", g), time.time() - t0)
        log(spec.name, rep, "tgt", g, time.time() - t0)


def write_firewall(cn, rep, extra_files=()):
    man = {"cohort": cn, "rep": rep,
           "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "files": {}}
    spec = cohorts.LOADERS[cn]()
    for e in spec.eligible_excl:
        for p in (fit_path(cn, rep, "src", e), feat_path(cn, rep, e)):
            man["files"][p] = sha_file(p)
    for g in spec.primary:
        gp = f"{STUDY}/feats/{cn}/rep{rep}/ab_{g.replace(' ', '_')}.npz"
        man["files"][gp] = sha_file(gp)
    for p in extra_files:
        man["files"][p] = sha_file(p)
    out = f"{STUDY}/conf/FIREWALL_{cn}_rep{rep}.json"
    if os.path.exists(out):
        raise RuntimeError(f"{out} already exists; refusing to overwrite a lock")
    json_atomic(out, man)
    return out


def verify_firewall(cn, rep):
    man = json.load(open(f"{STUDY}/conf/FIREWALL_{cn}_rep{rep}.json"))
    for p, h in man["files"].items():
        if sha_file(p) != h:
            raise RuntimeError(f"firewall violation: {p} changed after lock")


def log(cohort, rep, kind, env, wall):
    os.makedirs(f"{STUDY}/runs", exist_ok=True)
    with open(f"{STUDY}/runs/ledger.jsonl", "a") as f:
        f.write(json.dumps({"cohort": cohort, "rep": rep, "kind": kind,
                            "env": env, "wall_s": round(wall, 1),
                            "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
    print(f"  [{cohort} rep{rep}] {kind} {env}: {wall:.1f}s", flush=True)
