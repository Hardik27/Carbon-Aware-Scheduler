from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict
from .data import Instance, ExperimentConfig


@dataclass
class Solution:
    x: np.ndarray       # (I,J,T) binary
    load: np.ndarray    # (I,T)
    g: np.ndarray       # (I,T) grid kWh
    r: np.ndarray       # (I,T) renewable kWh
    lateness: np.ndarray  # (J,) slots late (0 if on time)


def _compute_energy(inst: Instance, load: np.ndarray, ren_cap_avg: np.ndarray):
    phi = inst.alpha + inst.beta * load
    r = np.minimum(phi, ren_cap_avg)
    g = np.clip(phi - r, 0.0, None)
    return g, r


def _lateness_from_x(inst: Instance, x: np.ndarray):
    I, J, T = x.shape
    lateness = np.zeros(J)
    for j in range(J):
        t = int(np.argmax(x[:, j, :]) % T)
        lateness[j] = max(0, t - inst.d[j])
    return lateness


def solve_cost_only_greedy(inst: Instance, cfg: ExperimentConfig) -> Solution:
    I, T, J = inst.I, inst.T, inst.J
    avg_price = inst.price.mean(axis=0)
    x = np.zeros((I, J, T), dtype=int)
    load = np.zeros((I, T), dtype=float)

    for j in range(J):
        window = range(inst.a[j], inst.d[j] + 1)
        candidates = [(i, t) for i in range(I) for t in window]
        candidates.sort(key=lambda it: avg_price[it[0], it[1]] + 0.02*inst.Pij[it[0], j])
        placed = False
        for i, t in candidates:
            if load[i, t] + inst.p[j] <= inst.C[i, t]:
                x[i, j, t] = 1
                load[i, t] += inst.p[j]
                placed = True
                break
        if not placed:
            # force at earliest feasible anywhere to avoid infeasibility
            for t in range(inst.a[j], T):
                for i in range(I):
                    if load[i, t] + inst.p[j] <= inst.C[i, t]:
                        x[i, j, t] = 1
                        load[i, t] += inst.p[j]
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                # last resort: drop into min-load slot (will violate capacity slightly)
                i, t = np.unravel_index(np.argmin(load), load.shape)
                x[i, j, t] = 1
                load[i, t] += inst.p[j]

    ren_cap_avg = inst.ren_cap.mean(axis=0)
    g, r = _compute_energy(inst, load, ren_cap_avg)
    lateness = _lateness_from_x(inst, x)

    return Solution(x=x, load=load, g=g, r=r, lateness=lateness)


def solve_carbon_only_greedy(inst: Instance, cfg: ExperimentConfig) -> Solution:
    I, T, J = inst.I, inst.T, inst.J
    avg_carbon = inst.carbon.mean(axis=0)

    x = np.zeros((I, J, T), dtype=int)
    load = np.zeros((I, T), dtype=float)

    for j in range(J):
        window = range(inst.a[j], inst.d[j] + 1)
        candidates = [(i, t) for i in range(I) for t in window]
        candidates.sort(key=lambda it: avg_carbon[it[0], it[1]] + 0.02*inst.Pij[it[0], j])
        placed = False
        for i, t in candidates:
            if load[i, t] + inst.p[j] <= inst.C[i, t]:
                x[i, j, t] = 1
                load[i, t] += inst.p[j]
                placed = True
                break
        if not placed:
            # fallback: cost-only behavior if carbon-only can't place
            return solve_cost_only_greedy(inst, cfg)

    ren_cap_avg = inst.ren_cap.mean(axis=0)
    g, r = _compute_energy(inst, load, ren_cap_avg)
    lateness = _lateness_from_x(inst, x)
    return Solution(x=x, load=load, g=g, r=r, lateness=lateness)


def solve_weighted_expected_greedy(inst: Instance, cfg: ExperimentConfig) -> Solution:
    I, T, J = inst.I, inst.T, inst.J
    avg_price = inst.price.mean(axis=0)
    avg_carbon = inst.carbon.mean(axis=0)

    x = np.zeros((I, J, T), dtype=int)
    load = np.zeros((I, T), dtype=float)

    for j in range(J):
        window = range(inst.a[j], inst.d[j] + 1)
        candidates = [(i, t) for i in range(I) for t in window]
        candidates.sort(
            key=lambda it: cfg.cost_w*avg_price[it[0], it[1]] + cfg.carbon_w*avg_carbon[it[0], it[1]] + cfg.place_w*inst.Pij[it[0], j]
        )
        placed = False
        for i, t in candidates:
            if load[i, t] + inst.p[j] <= inst.C[i, t]:
                x[i, j, t] = 1
                load[i, t] += inst.p[j]
                placed = True
                break
        if not placed:
            return solve_cost_only_greedy(inst, cfg)

    ren_cap_avg = inst.ren_cap.mean(axis=0)
    g, r = _compute_energy(inst, load, ren_cap_avg)
    lateness = _lateness_from_x(inst, x)
    return Solution(x=x, load=load, g=g, r=r, lateness=lateness)


def solve_random_feasible(inst: Instance, cfg: ExperimentConfig) -> Solution:
    rng = np.random.default_rng(1234)
    I, T, J = inst.I, inst.T, inst.J

    x = np.zeros((I, J, T), dtype=int)
    load = np.zeros((I, T), dtype=float)

    for j in range(J):
        window = list(range(inst.a[j], inst.d[j] + 1))
        rng.shuffle(window)
        placed = False
        for t in window:
            order = list(range(I))
            rng.shuffle(order)
            for i in order:
                if load[i, t] + inst.p[j] <= inst.C[i, t]:
                    x[i, j, t] = 1
                    load[i, t] += inst.p[j]
                    placed = True
                    break
            if placed:
                break
        if not placed:
            return solve_cost_only_greedy(inst, cfg)

    ren_cap_avg = inst.ren_cap.mean(axis=0)
    g, r = _compute_energy(inst, load, ren_cap_avg)
    lateness = _lateness_from_x(inst, x)
    return Solution(x=x, load=load, g=g, r=r, lateness=lateness)
