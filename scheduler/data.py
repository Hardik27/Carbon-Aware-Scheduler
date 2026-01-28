from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ExperimentConfig:
    beta: float = 0.9
    eps: float = 0.05

    cost_w: float = 1.0
    carbon_w: float = 2.0
    cvar_w: float = 5.0
    place_w: float = 0.2
    late_w: float = 2.0

    budget_frac: float = 0.95
    budget_mode: str = "cvar"  # "cvar" or "worst"


@dataclass
class Instance:
    I: int
    T: int
    J: int

    a: np.ndarray
    d: np.ndarray
    p: np.ndarray

    # capacity and energy mapping
    C: np.ndarray          # (I,T) capacity units
    alpha: np.ndarray      # (I,T) kWh base
    beta: np.ndarray       # (I,T) kWh per capacity unit

    # placement penalty
    Pij: np.ndarray        # (I,J)

    # scenarios
    S: int
    prob: np.ndarray       # (S,)
    price: np.ndarray      # (S,I,T) $/kWh grid
    carbon: np.ndarray     # (S,I,T) kg/kWh grid
    ren_cap: np.ndarray    # (S,I,T) kWh max renewable usable


def gen_instance(I=2, T=12, J=60, S=10, seed=0) -> Instance:
    rng = np.random.default_rng(seed)

    # Jobs: tighter windows create SLA pressure
    a = rng.integers(low=0, high=max(1, T // 2), size=J)
    win = rng.integers(low=max(1, T // 6), high=max(2, T // 3), size=J)
    d = np.clip(a + win, a + 1, T - 1)

    p = rng.integers(low=1, high=3, size=J)  # 1..2 capacity units

    # Compute capacity: sized so it’s feasible but not trivial
    baseC = max(6, int(np.ceil(J / T)))
    C = rng.integers(low=baseC, high=baseC + 3, size=(I, T))

    # Energy mapping (kWh): phi = alpha + beta * load
    # alpha small, beta around 1
    alpha = rng.uniform(0.1, 0.4, size=(I, T))
    beta = rng.uniform(0.8, 1.3, size=(I, T))

    # Placement penalties
    region_bias = rng.uniform(0.0, 0.4, size=(I, 1))
    Pij = region_bias + rng.uniform(0.0, 0.3, size=(I, J))

    # Scenario probabilities
    prob = np.ones(S) / S

    # Base price/carbon profiles with regional heterogeneity
    base_price = rng.uniform(0.08, 0.18, size=(I, T))  # $/kWh
    base_carbon = rng.uniform(0.20, 0.65, size=(I, T)) # kg/kWh

    price = np.zeros((S, I, T))
    carbon = np.zeros((S, I, T))
    ren_cap = np.zeros((S, I, T))

    # Renewable caps (kWh) must be scarce to avoid g=0 degeneracy.
    # We tie renewable caps to capacity but keep them significantly lower than typical phi.
    # This guarantees grid usage in most runs.
    # Expected load per slot: ~J/T * avg(p) ~ (J/T*1.5)
    # Expected phi ~ alpha + beta * load; renewable cap should cover only ~20-50% of phi.
    expected_load = (J / T) * 1.5
    expected_phi = alpha.mean() + beta.mean() * expected_load
    # region-specific renewable scale
    ren_scale = rng.uniform(0.15, 0.45, size=(I, 1)) * expected_phi

    tgrid = np.arange(T)

    for s in range(S):
        # temporal modulation: prices and carbon vary over day
        price_temporal = 1.0 + 0.25 * np.sin(2*np.pi*(tgrid/T + 0.15*s/S))
        carbon_temporal = 1.0 + 0.30 * np.cos(2*np.pi*(tgrid/T + 0.10*s/S))

        # scenario shock: a subset of scenarios have late-slot carbon spikes in one region
        carbon_spike = np.ones((I, T))
        if s >= int(0.7 * S):
            spike_region = rng.integers(0, I)
            late = slice(max(0, T//2), T)
            carbon_spike[spike_region, late] *= rng.uniform(1.25, 1.70)

        # apply noise + heterogeneity
        price[s] = base_price * price_temporal + rng.normal(0, 0.005, size=(I, T))
        price[s] = np.clip(price[s], 0.02, None)

        carbon[s] = base_carbon * carbon_temporal * carbon_spike + rng.normal(0, 0.02, size=(I, T))
        carbon[s] = np.clip(carbon[s], 0.05, None)

        # renewables: midday-ish bump but still scarce
        ren_shape = 0.4 + 0.6*np.exp(-0.5*((tgrid - 0.55*T)/(0.18*T))**2)  # bell curve
        ren = ren_scale * ren_shape  # (I,1)*(T,) -> broadcast
        ren = ren + rng.normal(0, 0.05*expected_phi, size=(I, T))
        ren = np.clip(ren, 0.0, None)

        # occasional curtailment in some scenarios
        if s % 4 == 0:
            cut_region = rng.integers(0, I)
            ren[cut_region, :] *= rng.uniform(0.4, 0.7)

        ren_cap[s] = ren

    return Instance(I, T, J, a, d, p, C, alpha, beta, Pij, S, prob, price, carbon, ren_cap)
