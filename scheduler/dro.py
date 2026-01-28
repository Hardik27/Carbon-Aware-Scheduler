from __future__ import annotations

import time
from typing import Tuple, Optional

import numpy as np
import cvxpy as cp
from scipy.linalg import cholesky

from .data import Instance, ExperimentConfig
from .baseline import Solution


def solve_dro_cvar_drcc(
    inst: Instance,
    cfg: ExperimentConfig,
    carbon_budget: float,
    solver: str = "SCS",
    timeout_s: float = 120.0,
) -> Tuple[Optional[Solution], str]:
    """
    Convex DRO-style surrogate:
    - x relaxed to [0,1] (large-scale friendly). If you want MILP, swap x to boolean and use ECOS_BB (small cases).
    - load = sum p_j x_ijt
    - energy: g + r >= alpha + beta*load, 0<=r<=ren_cap_avg, g>=0
    - Objective:
        w_cost * E[cost] + w_carbon * E[carbon] + w_place * placement + w_late * lateness + w_cvar * CVaR_beta(carbon)
    - DRCC-style SOC carbon budget using sample mean/cov of carbon intensities:
        mu^T g + k * ||Sigma^{1/2} g||_2 <= B
      where k = sqrt((1-eps)/eps)
    """

    I, T, J, S = inst.I, inst.T, inst.J, inst.S
    t_start = time.time()

    # Decision variables
    x = cp.Variable((I, J, T))
    load = cp.Variable((I, T))
    g = cp.Variable((I, T))
    r = cp.Variable((I, T))

    # lateness slack per job (soft deadline)
    s = cp.Variable(J)  # lateness in slots >=0
    # scheduled time per job (expected time index)
    t_idx = np.arange(T, dtype=float)

    constraints = []

    # Bounds
    constraints += [x >= 0, x <= 1]
    constraints += [g >= 0, r >= 0, s >= 0]

    # Assignment within window
    for j in range(J):
        mask = np.zeros((I, T))
        mask[:, inst.a[j]:inst.d[j] + 1] = 1.0
        constraints += [cp.sum(cp.multiply(mask, x[:, j, :])) == 1.0]

    # Capacity and load definition
    constraints += [load == cp.sum(cp.multiply(x, inst.p.reshape(1, J, 1)), axis=1)]
    constraints += [load <= inst.C]

    # Energy mapping
    phi = inst.alpha + cp.multiply(inst.beta, load)
    ren_cap_avg = inst.ren_cap.mean(axis=0)
    constraints += [g + r >= phi]
    constraints += [r <= ren_cap_avg]

    # Soft deadlines: sum_{i,t} t*x <= d_j + s_j
    for j in range(J):
        constraints += [cp.sum(cp.multiply(x[:, j, :], t_idx.reshape(1, T))) <= inst.d[j] + s[j]]

    # Expected cost/carbon
    avg_price = inst.price.mean(axis=0)
    avg_carbon = inst.carbon.mean(axis=0)

    exp_cost = cp.sum(cp.multiply(avg_price, g))
    exp_carbon = cp.sum(cp.multiply(avg_carbon, g))

    # Placement penalty uses x aggregated over time
    place_pen = cp.sum(cp.multiply(inst.Pij, cp.sum(x, axis=2)))

    # Lateness penalty (total lateness)
    late_pen = cp.sum(s)

    # Scenario carbon losses for CVaR
    scen_carbon = []
    for k in range(S):
        scen_carbon.append(cp.sum(cp.multiply(inst.carbon[k], g)))
    scen_carbon = cp.hstack(scen_carbon)

    eta = cp.Variable()
    xi = cp.Variable(S)
    constraints += [xi >= 0, xi >= scen_carbon - eta]
    cvar_carbon = eta + (1.0 / (1.0 - cfg.beta)) * cp.sum(cp.multiply(inst.prob, xi))

    # DRCC-style SOC budget (from sample mean/cov of carbon intensities)
    # Flatten g
    gvec = cp.reshape(g, (I*T, 1))

    # sample mean/cov over scenarios for carbon intensities (flatten per scenario)
    Cmat = inst.carbon.reshape(S, I*T)
    mu = Cmat.mean(axis=0)  # (I*T,)
    Sigma = np.cov(Cmat, rowvar=False)  # (I*T, I*T)

    # Cholesky with jitter for numerical stability
    jitter = 1e-6 * np.eye(I*T)
    try:
        L = cholesky(Sigma + jitter, lower=True)
    except Exception:
        # fallback: diagonal approximation if covariance is ill-conditioned
        L = np.diag(np.sqrt(np.maximum(np.diag(Sigma), 1e-8)))

    kappa = np.sqrt((1.0 - cfg.eps) / cfg.eps)
    constraints += [mu @ cp.vec(g) + kappa * cp.norm(L @ gvec, 2) <= carbon_budget]

    # Objective
    obj = (
        cfg.cost_w * exp_cost
        + cfg.carbon_w * exp_carbon
        + cfg.place_w * place_pen
        + cfg.late_w * late_pen
        + cfg.cvar_w * cvar_carbon
    )

    problem = cp.Problem(cp.Minimize(obj), constraints)

    # Solve with robust solver choice
    try:
        if solver == "SCS":
            problem.solve(solver=cp.SCS, verbose=False, max_iters=20000, time_limit_secs=timeout_s)
        else:
            problem.solve(solver=cp.ECOS, verbose=False, max_iters=5000)
    except Exception as e:
        return None, f"error:{type(e).__name__}"

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        return None, f"status:{problem.status}"

    # Extract solution
    xval = np.clip(x.value, 0, 1)
    gval = np.maximum(g.value, 0)
    rval = np.maximum(r.value, 0)
    sval = np.maximum(s.value, 0)

    # Optional: simple rounding for x (keeps feasibility mostly; good for reporting)
    xbin = np.zeros_like(xval)
    for j in range(J):
        # choose max (i,t)
        idx = np.argmax(xval[:, j, :])
        i = idx // T
        t = idx % T
        xbin[i, j, t] = 1

    # recompute load with rounded xbin
    load_bin = np.sum(xbin * inst.p.reshape(1, J, 1), axis=1)

    # recompute energy with rounded schedule (keep renewables cap)
    phi_bin = inst.alpha + inst.beta * load_bin
    ren_cap_avg = inst.ren_cap.mean(axis=0)
    r_bin = np.minimum(phi_bin, ren_cap_avg)
    g_bin = np.maximum(phi_bin - r_bin, 0)

    # lateness from slack (rounded schedule lateness computed later in metrics)
    sol = Solution(x=xbin.astype(int), load=load_bin, g=g_bin, r=r_bin, lateness=sval)

    if (time.time() - t_start) > timeout_s:
        return sol, "ok_timeout_soft"

    return sol, "ok"
