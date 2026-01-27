## run_experiments.py

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from tabulate import tabulate

from scheduler.data import gen_instance, Instance, ExperimentConfig
from scheduler.baseline import (
    solve_cost_only_greedy,
    solve_carbon_only_greedy,
    solve_weighted_expected_greedy,
    solve_random_feasible,
)
from scheduler.dro import solve_dro_cvar_drcc
from scheduler.metrics import compute_metrics
from scheduler.plots import (
    plot_illustrative_bars,
    plot_scaling_lines,
    plot_sensitivity_heatmap,
)
from scheduler.utils import ensure_dir, now_utc_iso


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["quick", "paper", "large"], default="quick")
    p.add_argument("--outdir", type=str, default="out")

    # sweep controls
    p.add_argument("--jobs", nargs="*", type=int, default=None)
    p.add_argument("--regions", nargs="*", type=int, default=None)
    p.add_argument("--slots", type=int, default=12)
    p.add_argument("--scenarios", type=int, default=10)
    p.add_argument("--seeds", type=int, default=5)

    # model knobs
    p.add_argument("--beta", type=float, default=0.9)          # CVaR level
    p.add_argument("--eps", type=float, default=0.05)          # DRCC risk tolerance
    p.add_argument("--cvar_w", type=float, default=5.0)        # weight on carbon CVaR
    p.add_argument("--late_w", type=float, default=2.0)        # weight on lateness
    p.add_argument("--carbon_w", type=float, default=2.0)      # expected carbon weight
    p.add_argument("--cost_w", type=float, default=1.0)        # expected cost weight
    p.add_argument("--place_w", type=float, default=0.2)       # placement penalty weight

    # carbon budget tightening relative to baseline
    p.add_argument("--budget_frac", type=float, default=0.95)  # B = budget_frac * baseline_CVaR (or worst)
    p.add_argument("--budget_mode", choices=["cvar", "worst"], default="cvar")

    # large experiment runtime knobs
    p.add_argument("--dro_timeout_s", type=float, default=120.0)
    p.add_argument("--solver", choices=["ECOS", "SCS"], default="SCS")

    return p.parse_args()


def run_one(inst: Instance, cfg: ExperimentConfig, solver: str, timeout_s: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    # --- Baselines ---
    baselines = {
        "baseline_cost": solve_cost_only_greedy,
        "baseline_carbon": solve_carbon_only_greedy,
        "baseline_weighted": solve_weighted_expected_greedy,
        "baseline_random": solve_random_feasible,
    }

    baseline_solutions = {}
    for name, fn in baselines.items():
        t0 = time.time()
        sol = fn(inst, cfg)
        t1 = time.time()
        m = compute_metrics(inst, sol)
        rows.append({
            "method": name,
            "solve_status": "ok",
            "solve_time_s": t1 - t0,
            **m
        })
        baseline_solutions[name] = (sol, m)

    # choose budget reference baseline
    ref_name = "baseline_cost"
    ref = baseline_solutions[ref_name][1]
    if cfg.budget_mode == "cvar":
        B = cfg.budget_frac * ref["cvar_carbon"]
    else:
        B = cfg.budget_frac * ref["worst_carbon"]

    # --- DRO ---
    t0 = time.time()
    dro_sol, status = solve_dro_cvar_drcc(
        inst=inst,
        cfg=cfg,
        carbon_budget=B,
        solver=solver,
        timeout_s=timeout_s,
    )
    t1 = time.time()

    if dro_sol is None:
        rows.append({
            "method": "dro",
            "solve_status": status,
            "solve_time_s": t1 - t0,
            "exp_cost": np.nan,
            "exp_carbon": np.nan,
            "cvar_carbon": np.nan,
            "worst_carbon": np.nan,
            "late_slots_mean": np.nan,
            "late_jobs_frac": np.nan,
        })
    else:
        m = compute_metrics(inst, dro_sol)
        rows.append({
            "method": "dro",
            "solve_status": status,
            "solve_time_s": t1 - t0,
            **m
        })

    return rows


def make_config_from_args(args) -> ExperimentConfig:
    return ExperimentConfig(
        beta=args.beta,
        eps=args.eps,
        cost_w=args.cost_w,
        carbon_w=args.carbon_w,
        cvar_w=args.cvar_w,
        place_w=args.place_w,
        late_w=args.late_w,
        budget_frac=args.budget_frac,
        budget_mode=args.budget_mode,
    )


def main():
    args = parse_args()
    ensure_dir(args.outdir)

    cfg = make_config_from_args(args)

    # mode presets
    if args.mode == "quick":
        regions = [2]
        jobs = [30]
        slots = 8
        scenarios = 8
        seeds = 2
    elif args.mode == "paper":
        regions = [2, 4]
        jobs = [30, 60, 120]
        slots = args.slots
        scenarios = args.scenarios
        seeds = args.seeds
    else:
        regions = args.regions if args.regions else [2, 4, 6]
        jobs = args.jobs if args.jobs else [50, 100, 200, 400]
        slots = args.slots
        scenarios = args.scenarios
        seeds = args.seeds

    raw_rows: List[Dict[str, Any]] = []

    meta = {
        "timestamp_utc": now_utc_iso(),
        "mode": args.mode,
        "regions": regions,
        "jobs": jobs,
        "slots": slots,
        "scenarios": scenarios,
        "seeds": seeds,
        "solver": args.solver,
        "dro_timeout_s": args.dro_timeout_s,
        "config": asdict(cfg),
    }
    with open(os.path.join(args.outdir, "run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    for I in regions:
        for J in jobs:
            for seed in tqdm(range(seeds), desc=f"I={I}, J={J}"):
                inst = gen_instance(I=I, T=slots, J=J, S=scenarios, seed=seed)
                rows = run_one(inst, cfg, solver=args.solver, timeout_s=args.dro_timeout_s)
                for r in rows:
                    r.update({"I": I, "J": J, "T": slots, "S": scenarios, "seed": seed})
                raw_rows.extend(rows)

    df_raw = pd.DataFrame(raw_rows)
    raw_path = os.path.join(args.outdir, "metrics_raw.csv")
    df_raw.to_csv(raw_path, index=False)

    # aggregate: mean + 95% CI over seeds (skip failed/NaN)
    def agg_ci(x):
        x = x.dropna()
        if len(x) == 0:
            return pd.Series({"mean": np.nan, "ci95": np.nan})
        m = x.mean()
        s = x.std(ddof=1) if len(x) > 1 else 0.0
        ci = 1.96 * s / np.sqrt(len(x)) if len(x) > 1 else 0.0
        return pd.Series({"mean": m, "ci95": ci})

    grp = df_raw[df_raw["solve_status"].str.startswith("ok")].groupby(["method", "I", "J"])
    metrics = ["exp_cost", "exp_carbon", "cvar_carbon", "worst_carbon", "late_slots_mean", "late_jobs_frac"]
    agg = grp[metrics].apply(lambda g: pd.concat([agg_ci(g[c]).rename(lambda k: f"{c}_{k}") for c in metrics], axis=0))
    agg = agg.reset_index()

    agg_path = os.path.join(args.outdir, "metrics_agg.csv")
    agg.to_csv(agg_path, index=False)

    # quick console tables
    view = agg[["method", "I", "J",
                "exp_cost_mean", "exp_carbon_mean", "cvar_carbon_mean", "worst_carbon_mean",
                "late_jobs_frac_mean"]].copy()
    print("\nAggregated results (means):")
    print(tabulate(view, headers="keys", tablefmt="github", floatfmt=".4f"))

    # plots for paper/large
    try:
        plot_illustrative_bars(df_raw, outdir=args.outdir)
        plot_scaling_lines(agg, outdir=args.outdir)
        plot_sensitivity_heatmap(df_raw, outdir=args.outdir)
    except Exception as e:
        print(f"[warn] Plotting failed: {e}")

    print(f"\nSaved: {raw_path}")
    print(f"Saved: {agg_path}")
    print(f"Figures saved under: {args.outdir}/")


if __name__ == "__main__":
    main()
