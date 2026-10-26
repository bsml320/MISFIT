"""CRT study: source-endpoint gradients, objective contrasts, damped-GGN solve.

Binary CE structure makes everything exact. With logits z and s = z1 - z0:
  per-sample loss gradient     grad l_i = c_i u_i,  c_i = p_i1 - y_i
  GGN of the mean CE over T    G = (1/|T|) sum_{i in T} w_i u_i u_i^T,
                               w_i = p_i1 (1 - p_i1)
where u_i = grad_theta s_i. So the damped operator C = G + (lam + rho) I is a
rank-|T| update of a scaled identity and (C)^{-1} d is computed EXACTLY by the
Woodbury identity through the |T| x |T| kernel in float64 (torch.linalg.solve,
never inverse; condition number logged). This is matrix-free in the parameter
dimension: no P x P matrix is ever formed.

Objective contrast at a source endpoint theta_{-e}:
  Delta_{e,g} = grad[R_{-g} - R_{-e}](theta_{-e})
             = mean_{j not in g} c_j u_j - mean_{j not in e} c_j u_j
(the shared ridge term lam * theta cancels). R_{-S} = mean CE over the cohort
minus environment S plus (lam/2)||theta||^2 with lam = weight decay = 1e-4.

Features for probe i on environment g (containing refits C_g):
  b_i = -(1/|C_g|) sum_e c_i u_i . Delta_{e,g}          (gradient-only)
  a_i = -(1/|C_g|) sum_e c_i u_i . v_{e,g},  v = C_e^{-1} Delta_{e,g}
Damping rho = c * tr(G_e)/P with c from a prespecified grid; c chosen on
development data only, one value frozen for confirmation.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.func import functional_call, grad, vmap

from misfit.crt_study.code.model import CNN3D

DEV = "cuda"
LAM = 1e-4                       # ridge = Adam coupled weight decay
DAMP_GRID = (0.03, 0.1, 0.3, 1.0, 3.0)   # c values; rho = c * tr(G)/P
COND_WARN = 1e6


def per_sample_u(state, X, bs=48):
    """U[i] = grad_theta (z1 - z0)(x_i), flattened in sorted-key order; and
    the deterministic logits. fp32, on GPU."""
    skel = CNN3D().to(DEV)
    params = {k: v.to(DEV) for k, v in state.items()}
    keys = sorted(params)

    def sgap(pr, x):
        z = functional_call(skel, pr, (x[None],))[0]
        return z[1] - z[0]

    gfn = vmap(grad(sgap), in_dims=(None, 0))
    N = len(X)
    P = int(sum(params[k].numel() for k in keys))
    U = torch.empty(N, P, device=DEV)
    logits = torch.empty(N, 2)
    with torch.no_grad():
        skel.eval()
        for i in range(0, N, 128):
            zb = functional_call(skel, params, (X[i:i + 128].to(DEV),))
            logits[i:i + 128] = zb.cpu()
    for i in range(0, N, bs):
        gd = gfn(params, X[i:i + bs].to(DEV))
        U[i:i + len(gd[keys[0]])] = torch.cat(
            [gd[k].reshape(gd[k].shape[0], -1) for k in keys], dim=1)
    return U, logits


def cw(logits, y):
    """c_i = p1 - y, w_i = p1 p0 from deterministic logits."""
    p1 = torch.softmax(logits, 1)[:, 1]
    c = (p1 - torch.as_tensor(y, dtype=torch.float32)).to(DEV)
    w = (p1 * (1 - p1)).to(DEV)
    return c, w


def mean_grad(U, c, idx):
    """grad of mean CE over subset idx (data part only)."""
    return (c[idx, None] * U[idx]).mean(dim=0)


def ggn_trace(U, w, idx):
    return float((w[idx] * (U[idx] ** 2).sum(dim=1)).mean())


class WoodburyFamily:
    """Shared fp64 factor A (|T| x P, built once) for G = A^T A; per-damping
    solvers reuse it, only the |T| x |T| kernel differs."""

    def __init__(self, U, w, idx):
        sw = torch.sqrt(w[idx].double() / len(idx))
        self.A = U[idx].double() * sw[:, None]
        self.K0 = self.A @ self.A.T

    def solver(self, alpha):
        return WoodburySolver(self, float(alpha))


class WoodburySolver:
    """Exact (G + alpha I)^{-1} d via the Woodbury identity (fp64, solve)."""

    def __init__(self, fam, alpha):
        self.fam, self.alpha = fam, alpha
        K = fam.K0.clone()
        K.diagonal().add_(alpha)
        ev = torch.linalg.eigvalsh(K)
        self.cond = float(ev.max() / ev.min())
        self.K = K

    def solve(self, d):
        d64 = d.double()
        rhs = self.fam.A @ d64
        v = (d64 - self.fam.A.T @ torch.linalg.solve(self.K, rhs)) / self.alpha
        return v

    def apply(self, v):
        """(G + alpha I) v, for residual checks."""
        return self.fam.A.T @ (self.fam.A @ v) + self.alpha * v

    def residual(self, v, d):
        r = self.apply(v) - d.double()
        return float(r.norm() / d.double().norm())


def endpoint_features(spec, state, X, y, env_codes, e_env, damp_cs=DAMP_GRID):
    """All transport quantities at one source endpoint theta_{-e_env}.

    Returns per-target-environment dict with Delta, v per damping, and per-probe
    dot products (c_i, u_i.Delta, u_i.v_c) for every probe of that environment,
    plus solver diagnostics. Only source-side quantities are used: no target
    parameters or target outcomes enter.
    """
    U, logits = per_sample_u(state, X)
    c, w = cw(logits, y)
    envs = np.asarray(env_codes)
    tr_idx = torch.from_numpy(np.where(envs != e_env)[0]).to(DEV)
    g_all = (1.0 / len(tr_idx)) * ((c[tr_idx, None] * U[tr_idx]).sum(dim=0))
    grad_norm = float((g_all + LAM * torch.cat(
        [state[k].reshape(-1).to(DEV) for k in sorted(state)])).norm())
    trG = ggn_trace(U, w, tr_idx)
    P = U.shape[1]
    alphas = {cc: LAM + cc * trG / P for cc in damp_cs}
    fam = WoodburyFamily(U, w, tr_idx)
    solvers = {cc: fam.solver(alphas[cc]) for cc in damp_cs}
    conds = {str(cc): solvers[cc].cond for cc in damp_cs}
    warn = any(v > COND_WARN for v in conds.values())
    if warn:
        print(f"WARNING: Woodbury kernel condition > {COND_WARN:.0e} "
              f"at endpoint {e_env}: {conds}", flush=True)
    out = {"trG": trG, "P": P, "alphas": {str(k): v for k, v in alphas.items()},
           "cond": conds, "cond_warn": warn,
           "stationarity_grad_norm": grad_norm, "targets": {}}
    probe_mask = spec.df.probe.to_numpy()
    for g_env in spec.primary:
        if g_env == e_env or e_env not in spec.containing(g_env):
            continue
        ng_idx = torch.from_numpy(np.where(envs != g_env)[0]).to(DEV)
        g_ng = (1.0 / len(ng_idx)) * ((c[ng_idx, None] * U[ng_idx]).sum(dim=0))
        delta = g_ng - g_all
        rows = np.where(probe_mask & (envs == g_env))[0]
        Up = U[torch.from_numpy(rows).to(DEV)]
        tgt = {"delta_norm": float(delta.norm()),
               "probe_rows": rows,
               "c": c[torch.from_numpy(rows).to(DEV)].cpu().numpy(),
               "dot_delta": (Up @ delta).cpu().numpy(),
               "dot_v": {}, "resid": {}, "delta": delta.cpu().numpy()}
        for cc in damp_cs:
            v = solvers[cc].solve(delta)
            tgt["resid"][str(cc)] = solvers[cc].residual(v, delta)
            tgt["dot_v"][str(cc)] = (Up.double() @ v).cpu().numpy()
            tgt[f"v_{cc}"] = v.float().cpu().numpy()
        out["targets"][g_env] = tgt
    del U, solvers, fam
    torch.cuda.empty_cache()
    return out


def assemble_ab(spec, endpoint_results, g_env, damp_c):
    """a_i, b_i for probes of g_env, averaging over containing endpoints."""
    C_g = spec.containing(g_env)
    rows = None
    a = b = None
    for e_env in C_g:
        t = endpoint_results[e_env]["targets"][g_env]
        if rows is None:
            rows = t["probe_rows"]
            a = np.zeros(len(rows))
            b = np.zeros(len(rows))
        elif not np.array_equal(rows, t["probe_rows"]):
            raise RuntimeError("probe row mismatch across endpoints")
        b += -t["c"] * t["dot_delta"]
        a += -t["c"] * t["dot_v"][str(damp_c)]
    return rows, a / len(C_g), b / len(C_g)
