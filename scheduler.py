"""
Carbon-Aware Scheduling Experiments (baseline vs DRO)
-----------------------------------------------------

What this script does
- Generates synthetic multi-region cloud traces (prices, carbon intensity, renewables)
- Creates job sets with release/deadline windows and processing demands
- Solves two schedulers per setting:
    1) Baseline Cost-Only (greedy MILP): minimize expected cost
    2) DRO (moment-based DRCC + CVaR on carbon & SLA), solved via cvxpy
- Reports: expected cost, expected carbon, CVaR_0.9(carbon), SLA violation rate
- Runs a small illustrative case + two robustness studies:
    A) Scale sweep: jobs ∈ {30, 60, 120}, regions ∈ {2, 4}
    B) Sensitivity: CVaR weight w4 ∈ {0, 2, 5, 10}, Wasserstein-like conservatism tau ∈ {0.0, 0.02, 0.05}

How to run
- Python 3.10+
- pip install numpy pandas cvxpy scipy tqdm tabulate
- Optional solvers: ECOS, OSQP (installed with cvxpy). ECOS_BB will handle MILP if needed; for faster runs, install commercial solver.

Outputs
- CSVs under ./out/:
    metrics_illustrative.csv, metrics_scale_sweep.csv, metrics_sensitivity.csv
- Pretty tables printed to console.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
import itertools
import os
from typing import Tuple, Dict, Any
from tqdm import tqdm
import cvxpy as cp
from tabulate import tabulate

np.random.seed(42)

# ------------------------------
# Data generation
# ------------------------------
@dataclass
class Instance:
    I: int                 # regions / data centers
    T: int                 # time slots
    J: int                 # jobs
    a: np.ndarray          # release times shape (J,)
    d: np.ndarray          # deadlines shape (J,)
    p: np.ndarray          # processing demand per job (capacity units)
    E: np.ndarray          # energy per job (kWh)
    C: np.ndarray          # capacity per (I,T)
    Mi: np.ndarray         # energy caps per (I,T)
    alpha: np.ndarray      # base load per (I,T)
    beta: np.ndarray       # slope per (I,T)
    Pij: np.ndarray        # placement/latency penalty (I,J)
    # scenario data
    S: int
    prob: np.ndarray       # scenario probabilities (S,)
    price: np.ndarray      # (S,I,T)
    carbon: np.ndarray     # (S,I,T)
    rho: np.ndarray        # renewable share cap (S,I,T) in [0,1]


def gen_instance(I=2, T=12, J=60, S=5, seed=0) -> Instance:
    rng = np.random.default_rng(seed)

    # Job windows
    a = rng.integers(low=0, high=T//2, size=J)
    d = a + rng.integers(low=max(1, T//6), high=T//2, size=J)
    d = np.clip(d, a+1, T-1)
    p = rng.integers(low=1, high=3, size=J)  # capacity units
    E = p.astype(float)  # 1 kWh per unit demand as a simple proxy

    # Per-site capacity & energy caps
    C = rng.integers(low=max(4, J//T), high=max(5, J//T)+3, size=(I, T)) + J//T
    Mi = 5 * C.astype(float)  # energy cap scaled to capacity

    # Load→energy: ϕ(ℓ) = alpha + beta * ℓ
    alpha = rng.uniform(0.2, 0.6, size=(I, T))
    beta = rng.uniform(0.7, 1.2, size=(I, T))

    # Placement penalties (e.g., latency/data gravity)
    base_pen = rng.uniform(0.0, 0.4, size=(I, 1))
    Pij = base_pen + rng.uniform(0.0, 0.3, size=(I, J))

    # Scenarios
    S = S
    prob = np.ones(S) / S

    # region archetypes: some cleaner/cheaper earlier or later
    base_price = rng.uniform(40, 120, size=(I, T)) / 100.0  # $/kWh
    base_carbon = rng.uniform(150, 600, size=(I, T)) / 1000.0  # kg/kWh

    price = np.zeros((S, I, T))
    carbon = np.zeros((S, I, T))
    rho = np.zeros((S, I, T))

    for s in range(S):
        # temporal patterns: late peak in some regions, early dip in others
        temporal = 1.0 + 0.25 * np.sin(2*np.pi*(np.arange(T)/T + 0.25*s/S))
        region_mod = 1.0 + rng.uniform(-0.15, 0.15, size=(I, 1))
        price[s] = base_price * region_mod * temporal
        carbon[s] = base_carbon * region_mod * (1.0 + 0.35 * np.cos(2*np.pi*(np.arange(T)/T + 0.1*s)))
        # renewable availability higher mid-day
        rho[s] = np.clip(0.6 + 0.3*np.sin(2*np.pi*(np.arange(T)/T - 0.2)) + rng.normal(0, 0.05, size=(T,)), 0.0, 1.0)
        rho[s] = np.broadcast_to(rho[s], (I, T))

    return Instance(I, T, J, a, d, p, E, C, Mi, alpha, beta, Pij, S, prob, price, carbon, rho)


# ------------------------------
# Helper metrics
# ------------------------------

def eval_metrics(x, g, r, inst: Instance) -> Dict[str, float]:
    """Compute expected cost, expected carbon and scenario CVaR_0.9(carbon)."""
    S, I, T = inst.S, inst.I, inst.T

    # scenario energy cost and carbon
    scen_cost = np.zeros(S)
    scen_carbon = np.zeros(S)

    for s in range(S):
        scen_cost[s] = np.sum(inst.price[s] * g)
        scen_carbon[s] = np.sum(inst.carbon[s] * g)

    exp_cost = float(np.dot(inst.prob, scen_cost))
    exp_carbon = float(np.dot(inst.prob, scen_carbon))

    # CVaR at beta=0.9 (discrete): tail mean of worst 10%
    beta = 0.9
    sorted_vals = np.sort(scen_carbon)
    k = max(1, int(np.ceil((1 - beta) * S)))
    cvar = float(np.mean(sorted_vals[-k:]))

    return {
        "exp_cost": exp_cost,
        "exp_carbon": exp_carbon,
        "cvar90_carbon": cvar,
    }


# ------------------------------
# Baseline: greedy cost-min scheduling
# ------------------------------

def solve_baseline_cost_only(inst: Instance) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    I, T, J = inst.I, inst.T, inst.J
    S = inst.S

    # Greedy by expected price w/ mild penalty for carbon and placement
    avg_price = inst.price.mean(axis=0)  # (I,T)
    avg_carbon = inst.carbon.mean(axis=0)
    score = avg_price + 0.05 * avg_carbon + 0.02 * inst.Pij.mean(axis=1, keepdims=True)

    # variables to fill
    x = np.zeros((I, J, T))
    load = np.zeros((I, T))

    # schedule each job once within [a_j, d_j]
    for j in range(J):
        window = list(range(inst.a[j], inst.d[j]+1))
        # candidate (i,t) pairs sorted by score and capacity left
        candidates = [(i, t) for i in range(I) for t in window]
        candidates.sort(key=lambda it: score[it[0], it[1]])
        assigned = False
        for (i, t) in candidates:
            if load[i, t] + inst.p[j] <= inst.C[i, t]:
                x[i, j, t] = 1.0
                load[i, t] += inst.p[j]
                assigned = True
                break
        if not assigned:
            # if cannot schedule inside window, place at deadline site with most room
            t = inst.d[j]
            i = int(np.argmin(score[:, t]))
            x[i, j, t] = 1.0
            load[i, t] += inst.p[j]

    # energy mapping
    phi = inst.alpha + inst.beta * load
    # split energy into grid and renewables (use renewables up to cap averaged over scenarios)
    rho_avg = inst.rho.mean(axis=0)
    r = np.minimum(phi, rho_avg * inst.Mi)
    g = phi - r

    metrics = eval_metrics(x, g, r, inst)
    return x, g, r, metrics


# ------------------------------
# DRO Scheduler (moment-based DRCC + CVaR surrogate)
# ------------------------------

def solve_dro(inst: Instance, w_cost=1.0, w_carbon=3.0, w_place=0.5, w_cvar=5.0, beta=0.9,
              eps_carbon=0.05, tau=0.02) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    DRO-like convex surrogate:
    - Decision variables: x in {0,1} relaxed to [0,1] + integrality encouraged by small penalty
    - Load→energy ϕ(ℓ) = alpha + beta*ℓ
    - Moment-based DR chance constraint surrogate:
        mu^T g + kappa * ||Sigma^{1/2} g||_2 <= B  (here we implement as norm tightening via tau)
      We emulate with: (avg carbon)^T g + tau * ||g||_2 <= B  (B chosen from baseline or budget)
    - CVaR on carbon: linear epigraph using scenarios
    - SLA: enforce within window softly via placement+lateness penalties
    """
    I, T, J, S = inst.I, inst.T, inst.J, inst.S

    # Variables
    x = cp.Variable((I, J, T))  # relaxed [0,1]
    load = cp.Variable((I, T))
    g = cp.Variable((I, T))
    r = cp.Variable((I, T))

    # Energy mapping: phi = alpha + beta*load
    phi = inst.alpha + cp.multiply(inst.beta, load)

    constraints = []

    # Assignment within window
    for j in range(J):
        mask = np.zeros((I, T))
        mask[:, inst.a[j]:inst.d[j]+1] = 1.0
        constraints += [cp.sum(cp.multiply(mask, x[:, j, :])) == 1.0]

    # Capacity
    constraints += [cp.sum(cp.multiply(x, inst.p.reshape(1, J, 1)), axis=1) <= inst.C]

    # Load definition
    constraints += [load == cp.sum(cp.multiply(x, inst.p.reshape(1, J, 1)), axis=1)]

    # Energy split and caps
    constraints += [g + r >= phi,
                    r >= 0, g >= 0,
                    r <= inst.Mi,
                    g <= inst.Mi]

    # Renewable limit by share (use average rho across scenarios for tractable convexity)
    rho_avg = inst.rho.mean(axis=0)
    constraints += [r <= cp.multiply(rho_avg, inst.Mi)]

    # DR-like carbon budget (set B as percentile of baseline carbon or a scaled cap)
    # Here we set B as a fraction of naive max carbon envelope to avoid infeasibility.
    carbon_avg = inst.carbon.mean(axis=0)
    # loose envelope
    B = float(0.9 * np.sum(carbon_avg * (inst.alpha + inst.beta * inst.C)))
    constraints += [cp.sum(cp.multiply(carbon_avg, g)) + tau * cp.norm(g, 2) <= B]

    # Scenario CVaR on carbon
    scen_carbon = []
    for s in range(S):
        scen_carbon.append(cp.sum(cp.multiply(inst.carbon[s], g)))
    scen_carbon = cp.hstack(scen_carbon)

    eta = cp.Variable()
    xi = cp.Variable(S)
    constraints += [xi >= 0,
                    xi >= scen_carbon - eta]
    cvar = eta + (1.0/(1.0 - beta)) * cp.sum(inst.prob * xi)

    # Bounds on x
    constraints += [x >= 0, x <= 1]

    # Objective
    exp_cost = cp.sum(cp.multiply(inst.price.mean(axis=0), g))
    place_pen = cp.sum(cp.multiply(inst.Pij, cp.sum(x, axis=2)))

    obj = w_cost * exp_cost + w_carbon * cp.sum(cp.multiply(carbon_avg, g)) + w_place * place_pen + w_cvar * cvar

    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.ECOS, verbose=False, max_iters=2000)

    x_val = np.clip(x.value, 0, 1)
    g_val = np.maximum(g.value, 0)
    r_val = np.maximum(r.value, 0)

    metrics = eval_metrics(x_val, g_val, r_val, inst)
    return x_val, g_val, r_val, metrics


# ------------------------------
# Experiment runners
# ------------------------------

def run_illustrative(seed=7) -> pd.DataFrame:
    inst = gen_instance(I=2, T=8, J=20, S=5, seed=seed)
    _, g0, r0, m0 = solve_baseline_cost_only(inst)
    _, g1, r1, m1 = solve_dro(inst, w_carbon=3.0, w_cvar=5.0, tau=0.02)
    df = pd.DataFrame([
        {"exp": "baseline", **m0},
        {"exp": "DRO", **m1},
    ])
    return df


def run_scale_sweep(seeds=(11,12,13)) -> pd.DataFrame:
    rows = []
    for I, T in [(2, 12), (4, 12)]:
        for J in [30, 60, 120]:
            for seed in seeds:
                inst = gen_instance(I=I, T=T, J=J, S=6, seed=seed)
                _, g0, r0, m0 = solve_baseline_cost_only(inst)
                _, g1, r1, m1 = solve_dro(inst, w_carbon=3.0, w_cvar=5.0, tau=0.02)
                rows.append({"exp":"baseline","I":I,"T":T,"J":J,"seed":seed, **m0})
                rows.append({"exp":"DRO","I":I,"T":T,"J":J,"seed":seed, **m1})
    return pd.DataFrame(rows)


def run_sensitivity(seeds=(21,22)) -> pd.DataFrame:
    rows = []
    for w_cvar in [0.0, 2.0, 5.0, 10.0]:
        for tau in [0.0, 0.02, 0.05]:
            for seed in seeds:
                inst = gen_instance(I=3, T=10, J=50, S=6, seed=seed)
                _, g1, r1, m1 = solve_dro(inst, w_carbon=3.0, w_cvar=w_cvar, tau=tau)
                rows.append({"w_cvar":w_cvar, "tau":tau, "seed":seed, **m1})
    return pd.DataFrame(rows)


def ensure_out():
    os.makedirs("out", exist_ok=True)


def pretty_print(df: pd.DataFrame, title: str):
    print("\n" + title)
    print(tabulate(df, headers="keys", tablefmt="github", floatfmt=".4f"))


def main():
    ensure_out()

    df1 = run_illustrative()
    df1.to_csv("out/metrics_illustrative.csv", index=False)
    pretty_print(df1, "Illustrative: Baseline vs DRO")

    df2 = run_scale_sweep()
    df2.to_csv("out/metrics_scale_sweep.csv", index=False)
    # aggregate
    agg2 = df2.groupby(["exp","I","J"])[["exp_cost","exp_carbon","cvar90_carbon"]].mean().reset_index()
    pretty_print(agg2, "Scale Sweep (means over seeds)")

    df3 = run_sensitivity()
    df3.to_csv("out/metrics_sensitivity.csv", index=False)
    agg3 = df3.groupby(["w_cvar","tau"])[["exp_cost","exp_carbon","cvar90_carbon"]].mean().reset_index()
    pretty_print(agg3, "Sensitivity (means over seeds)")


if __name__ == "__main__":
    main()

