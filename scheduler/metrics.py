from __future__ import annotations
import numpy as np
from .data import Instance
from .baseline import Solution


def _scenario_cost(inst: Instance, g: np.ndarray):
    return np.array([np.sum(inst.price[s] * g) for s in range(inst.S)], dtype=float)

def _scenario_carbon(inst: Instance, g: np.ndarray):
    return np.array([np.sum(inst.carbon[s] * g) for s in range(inst.S)], dtype=float)

def _cvar(values: np.ndarray, prob: np.ndarray, beta: float):
    # discrete CVaR: take worst tail mass (1-beta) and compute tail expectation (approx)
    # For equal weights this reduces to mean of worst k scenarios.
    order = np.argsort(values)
    vals = values[order]
    p = prob[order]
    tail_mass = 1.0 - beta
    acc = 0.0
    num = 0.0
    den = 0.0
    for v, w in zip(vals[::-1], p[::-1]):
        if acc >= tail_mass:
            break
        take = min(w, tail_mass - acc)
        num += v * take
        den += take
        acc += take
    return float(num / max(den, 1e-12))

def _schedule_time(inst: Instance, sol: Solution):
    # return assigned time per job
    I, J, T = sol.x.shape
    t_assigned = np.zeros(J, dtype=int)
    for j in range(J):
        idx = np.argmax(sol.x[:, j, :])
        t_assigned[j] = idx % T
    return t_assigned

def compute_metrics(inst: Instance, sol: Solution):
    scen_cost = _scenario_cost(inst, sol.g)
    scen_carbon = _scenario_carbon(inst, sol.g)

    exp_cost = float(np.dot(inst.prob, scen_cost))
    exp_carbon = float(np.dot(inst.prob, scen_carbon))
    worst_carbon = float(np.max(scen_carbon))
    cvar_carbon = _cvar(scen_carbon, inst.prob, beta=0.9)

    t_assigned = _schedule_time(inst, sol)
    lateness = np.maximum(0, t_assigned - inst.d)
    late_jobs_frac = float(np.mean(lateness > 0))
    late_slots_mean = float(np.mean(lateness))

    return {
        "exp_cost": exp_cost,
        "exp_carbon": exp_carbon,
        "cvar_carbon": cvar_carbon,
        "worst_carbon": worst_carbon,
        "late_slots_mean": late_slots_mean,
        "late_jobs_frac": late_jobs_frac,
    }
