"""CRT engineering tests (directive section 11). Run: pytest tests/test_crt.py -v
The synthetic transport checks default to CPU; real-cohort tests use the slow marker."""

from misfit.paths import resolve_path as _resolve_path
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, _resolve_path("<REPO_ROOT>/misfit/crt_study/code"))
from misfit.crt_study.code import analysis
from misfit.crt_study.code import calib
from misfit.crt_study.code import cohorts
from misfit.crt_study.code import transport
from misfit.crt_study.code.model import CNN3D

DEV = os.environ.get("MISFIT_TEST_DEVICE", "cpu")


@pytest.fixture(autouse=True)
def _test_device(monkeypatch):
    from misfit.crt_study.code import train
    monkeypatch.setattr(transport, "DEV", DEV)
    monkeypatch.setattr(train, "DEV", DEV)


# ---------- pair-MSE identity ----------
def test_pair_mse_identity():
    rng = np.random.RandomState(0)
    for n in (2, 3, 7, 40):
        u = rng.randn(n)
        assert np.isclose(analysis.pair_mse_naive(u), analysis.pair_mse(u))


# ---------- ridge-logistic known-derivative test ----------
class Logistic(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.head = nn.Linear(d, 2)

    def forward(self, x):
        return self.head(x.reshape(len(x), -1))


def _toy(n=60, d=5, seed=0):
    rng = np.random.RandomState(seed)
    X = torch.tensor(rng.randn(n, 1, d).astype(np.float32))
    y = (rng.rand(n) < 0.5).astype(int)
    env = np.array(["A", "B", "C"])[rng.randint(0, 3, n)]
    return X, y, env


def _u_logits(X):
    """per-sample u and logits via transport machinery on a logistic model."""
    from misfit.crt_study.code import transport as tr
    saved = tr.CNN3D
    tr.CNN3D = lambda: Logistic(X.shape[-1])
    try:
        torch.manual_seed(3)
        m = Logistic(X.shape[-1])
        state = {k: v.clone() for k, v in m.state_dict().items()}
        U, logits = tr.per_sample_u(state, X)
        return state, U, logits
    finally:
        tr.CNN3D = saved


def test_logistic_gradient_exact():
    X, y, env = _toy()
    state, U, logits = _u_logits(X)
    # analytic: d(z1 - z0)/dW = x stacked by class sign, d/db = (-1, 1)
    keys = sorted(state)
    assert keys == ["head.bias", "head.weight"]
    for i in (0, 7, 59):
        ub = U[i, :2].cpu().numpy()
        uw = U[i, 2:].cpu().numpy().reshape(2, -1)
        x = X[i, 0].numpy()
        assert np.allclose(ub, [-1.0, 1.0], atol=1e-5)
        assert np.allclose(uw[0], -x, atol=1e-5)
        assert np.allclose(uw[1], x, atol=1e-5)


def test_woodbury_matches_dense_hessian():
    """For logistic regression GGN == exact CE Hessian; compare the Woodbury
    solve against torch.linalg.solve on the dense autograd Hessian."""
    X, y, env = _toy()
    state, U, logits = _u_logits(X)
    c, w = transport.cw(logits, y)
    idx = torch.arange(len(X), device=DEV)
    alpha = 0.05
    ws = transport.WoodburyFamily(U, w, idx).solver(alpha)
    rng = np.random.RandomState(1)
    d = torch.tensor(rng.randn(U.shape[1]).astype(np.float32), device=DEV)
    v = ws.solve(d)
    assert ws.residual(v, d) < 1e-8
    # dense Hessian of mean CE via autograd
    m = Logistic(X.shape[-1]).to(DEV)
    m.load_state_dict(state)
    params = [p for _, p in sorted(m.named_parameters())]
    yt = torch.tensor(y).long().to(DEV)
    H = torch.autograd.functional.hessian(
        lambda *ps: nn.functional.cross_entropy(
            torch.func.functional_call(m, dict(zip([n for n, _ in sorted(m.named_parameters())], ps)),
                                       (X.to(DEV),)), yt),
        tuple(p.detach() for p in params))
    P = U.shape[1]
    Hd = torch.zeros(P, P, device=DEV, dtype=torch.float64)
    sizes = [p.numel() for p in params]
    off = np.concatenate([[0], np.cumsum(sizes)])
    for i in range(len(params)):
        for j in range(len(params)):
            Hd[off[i]:off[i + 1], off[j]:off[j + 1]] = \
                H[i][j].reshape(sizes[i], sizes[j]).double()
    v_dense = torch.linalg.solve(Hd + alpha * torch.eye(P, device=DEV, dtype=torch.float64),
                                 d.double())
    assert torch.allclose(v, v_dense, atol=1e-5, rtol=1e-4)


def test_contrast_matches_autograd():
    X, y, env = _toy()
    state, U, logits = _u_logits(X)
    c, w = transport.cw(logits, y)
    envs = np.asarray(env)
    gA = transport.mean_grad(U, c, torch.from_numpy(np.where(envs != "A")[0]).to(DEV))
    gB = transport.mean_grad(U, c, torch.from_numpy(np.where(envs != "B")[0]).to(DEV))
    delta = (gA - gB).cpu()          # grad[R_{-A} - R_{-B}]
    m = Logistic(X.shape[-1]).to(DEV)
    m.load_state_dict(state)
    yt = torch.tensor(y).long().to(DEV)
    iA = torch.tensor(np.where(envs != "A")[0])
    iB = torch.tensor(np.where(envs != "B")[0])
    obj = (nn.functional.cross_entropy(m(X[iA].to(DEV)), yt[iA])
           - nn.functional.cross_entropy(m(X[iB].to(DEV)), yt[iB]))
    obj.backward()
    ga = torch.cat([p.grad.reshape(-1).cpu()
                    for _, p in sorted(m.named_parameters())])
    assert torch.allclose(delta, ga, atol=1e-5)


def test_newton_prediction_on_logistic():
    """a_i approximates the actual loss change between the two refit optima of
    a ridge-logistic problem (quadratic-regime sanity, not CNN evidence)."""
    rng = np.random.RandomState(2)
    n, d = 300, 4
    Xn = rng.randn(n, d).astype(np.float32)
    beta_true = rng.randn(d)
    y = (1 / (1 + np.exp(-Xn @ beta_true)) > rng.rand(n)).astype(int)
    env = np.array(["A", "B", "C"])[rng.randint(0, 3, n)]
    lam = 0.1

    def fit(mask):
        mm = Logistic(d).to(DEV)
        opt = torch.optim.LBFGS(mm.parameters(), max_iter=200)
        Xt = torch.tensor(Xn[mask])[:, None, :].to(DEV)
        yt = torch.tensor(y[mask]).long().to(DEV)

        def cl():
            opt.zero_grad()
            l = nn.functional.cross_entropy(mm(Xt), yt)
            l = l + lam / 2 * sum((p ** 2).sum() for p in mm.parameters())
            l.backward()
            return l
        opt.step(cl)
        return {k: v.cpu().clone() for k, v in mm.state_dict().items()}

    th_e = fit(env != "B")        # source: exclude B
    th_g = fit(env != "A")        # target: exclude A
    from misfit.crt_study.code import transport as tr
    saved = tr.CNN3D
    tr.CNN3D = lambda: Logistic(d)
    try:
        Xt = torch.tensor(Xn)[:, None, :]
        U, logits = tr.per_sample_u(th_e, Xt)
        c, w = tr.cw(logits, y)
        iA = torch.from_numpy(np.where(env != "A")[0]).to(DEV)
        iB = torch.from_numpy(np.where(env != "B")[0]).to(DEV)
        delta = tr.mean_grad(U, c, iA) - tr.mean_grad(U, c, iB)
        ws = tr.WoodburyFamily(U, w, iB).solver(lam)   # curvature of R_{-B} + ridge
        v = ws.solve(delta)
        probes = np.where(env == "A")[0]
        a = -(c[probes][:, None].cpu().numpy()
              * (U[probes].double() @ v)[:, None].cpu().numpy())[:, 0]
        # actual per-sample loss change source -> target
        def ce_at(th):
            mm = Logistic(d)
            mm.load_state_dict(th)
            with torch.no_grad():
                z = mm(torch.tensor(Xn)[:, None, :])
            return nn.functional.cross_entropy(
                z, torch.tensor(y).long(), reduction="none").numpy()
        actual = (ce_at(th_g) - ce_at(th_e))[probes]
        r = np.corrcoef(a, actual)[0, 1]
        assert r > 0.9, f"Newton prediction corr too low: {r:.3f}"
    finally:
        tr.CNN3D = saved


# ---------- calibration ----------
def test_block_intercepts_cancel():
    rng = np.random.RandomState(4)
    n = 120
    tbl = pd.DataFrame({
        "uid": [f"s{i}" for i in range(n)],
        "env": np.repeat(["A", "B"], n // 2), "rep": np.tile([0, 1], n // 2),
        "L": rng.randn(n), "ce_first": rng.randn(n), "aum_neg": rng.randn(n),
        "el2n": rng.rand(n), "ce_std": rng.rand(n), "age": 60 + 10 * rng.rand(n),
        "sex_male1": rng.randint(0, 2, n), "b": rng.randn(n), "a": rng.randn(n)})
    tbl["q"] = tbl.L + 0.3 * tbl.a + 0.1 * rng.randn(n)
    m1 = calib.tune_and_fit(tbl, calib.MODEL_COLS["M3"])
    shifted = tbl.copy()
    blocks = shifted.env.astype(str) + "|" + shifted.rep.astype(str)
    shift = blocks.map({"A|0": 5.0, "A|1": -2.0, "B|0": 0.7, "B|1": 9.0})
    shifted["q"] = shifted.q + shift
    m2 = calib.tune_and_fit(shifted, calib.MODEL_COLS["M3"])
    assert np.allclose(m1["beta"], m2["beta"], atol=1e-8)
    assert np.isclose(m1["cv"], m2["cv"], atol=1e-10)


def test_group_folds_keep_participants_together():
    uids = np.array(["a", "b", "a", "c", "b", "a"])
    f = calib.group_folds(uids)
    for u in np.unique(uids):
        assert len(np.unique(f[uids == u])) == 1


def test_spline_recovers_nonlinearity():
    rng = np.random.RandomState(5)
    x = rng.randn(400)
    y = np.sin(1.5 * x)
    knots = calib.fit_basis(x)
    B = calib.ncs_basis(x, knots)
    beta = np.linalg.lstsq(B - B.mean(0), y - y.mean(), rcond=None)[0]
    pred = (B - B.mean(0)) @ beta
    assert np.corrcoef(pred, y - y.mean())[0, 1] > 0.95


# ---------- bootstrap block structure ----------
def test_bootstrap_keeps_participant_rows_together():
    """White-box: replicate the RNG stream and verify every draw equals the
    value obtained when whole participant blocks are resampled together."""
    rng = np.random.RandomState(6)
    n_sub, n_rep, B, seed = 7, 2, 11, 123
    uids = [f"s{i}" for i in range(n_sub)]
    df = pd.DataFrame({
        "uid": np.repeat(uids, n_rep),
        "env": "A", "rep": np.tile(range(n_rep), n_sub)})
    base = rng.randn(n_sub * n_rep)                   # varies across reps too
    d_sub = np.repeat(rng.randn(n_sub), n_rep)        # participant offset
    resid = {"M3": base, "M2": base + d_sub}
    out = analysis.boot_participants(df, resid, ["A"], list(range(n_rep)),
                                     B=B, seed=seed)
    # manual replication with guaranteed whole-uid blocks
    rng2 = np.random.RandomState(seed)
    man = np.empty(B)
    uu = np.unique(df.uid.to_numpy())
    for b in range(B):
        pick = rng2.choice(len(uu), len(uu), replace=True)
        rows = np.concatenate([np.where(df.uid.to_numpy() == uu[p])[0]
                               for p in pick])
        vals = []
        for m in ("M2", "M3"):
            cell = []
            for k in range(n_rep):
                x = resid[m][rows][df.rep.to_numpy()[rows] == k]
                cell.append(analysis.pair_mse(x))
            vals.append(np.mean(cell))
        man[b] = vals[0] - vals[1]
    lo, hi = np.percentile(man, 2.5), np.percentile(man, 97.5)
    assert np.isclose(out["M2"]["ci95"][0], lo) and np.isclose(out["M2"]["ci95"][1], hi)


# ---------- cohort integrity ----------
@pytest.mark.slow
def test_oasis3_counts_match_directive():
    spec = cohorts.oasis3()
    df = spec.df
    assert len(df) == 1360
    probe = df[df.probe]
    got = probe.groupby("env").size().to_dict()
    assert got == {"TrioTim": 160, "Biograph_mMR": 60, "Vision": 33,
                   "MAGNETOM_Vida": 24}
    assert int(df.label.sum()) == 277 + 107
    split = cohorts.split_probes(spec)
    dev = split[split.role == "dev"].groupby("env").size().to_dict()
    assert dev == {"TrioTim": 53, "Biograph_mMR": 20, "Vision": 11,
                   "MAGNETOM_Vida": 8}
    conf = split[split.role == "conf"].groupby("env").size().to_dict()
    assert conf == {"TrioTim": 107, "Biograph_mMR": 40, "Vision": 22,
                    "MAGNETOM_Vida": 16}


@pytest.mark.slow
def test_adni_dedup_and_oasis1():
    a = cohorts.adni()
    assert not a.df.uid.duplicated().any()
    o = cohorts.oasis1()
    assert not o.df.uid.duplicated().any()
    assert set(o.df.env) <= {"ageQ1", "ageQ2", "ageQ3", "ageQ4"}


# ---------- seeds, resume, firewall ----------
def test_seed_separation():
    from misfit.crt_study.code import driver
    seen = set()
    for cn in ("oasis3", "adni"):
        for rep in range(8):
            for kind in ("src", "tgt"):
                for e in ("E1", "E2"):
                    s = driver.seed_for(cn, rep, kind, e)
                    assert s not in seen
                    seen.add(s)


def test_resume_refuses_wrong_hash(tmp_path):
    from misfit.crt_study.code import train
    p = str(tmp_path / "x.npz")
    np.savez(p, cfg_hash=np.array("aaaa"), w_head=np.zeros(2))
    with pytest.raises(RuntimeError):
        train.load_state(p, "bbbb")


def test_conf_targets_locked_without_firewall(monkeypatch):
    from misfit.crt_study.code import driver

    class FakeSpec:
        name = "fake"
        primary = ["A"]
    with pytest.raises(RuntimeError, match="firewall"):
        driver.run_target_bank(FakeSpec(), None, "h", rep=991, conf=True)


def test_backbone_architecture_identity():
    sys.path.insert(0, _resolve_path("<REPO_ROOT>/misfit/experiments"))
    from misfit.experiments import run_oasis3_cnn_backbone as bb
    m_ours, m_theirs = CNN3D(), bb.CNN3D()
    ours, theirs = m_ours.state_dict(), m_theirs.state_dict()
    assert sorted(ours) == sorted(theirs)
    for k in ours:
        assert ours[k].shape == theirs[k].shape
    # semantics, not just shapes: identical weights -> identical outputs
    m_theirs.load_state_dict(ours)
    m_ours.eval(), m_theirs.eval()
    torch.manual_seed(11)
    x = torch.randn(2, 1, 32, 36, 32)
    with torch.no_grad():
        assert torch.allclose(m_ours(x), m_theirs(x), atol=1e-6)
