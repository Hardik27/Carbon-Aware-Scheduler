# run_experiments.py
# Copy-paste this entire file.

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, replace
from typing import List, Dict, Any, Tuple

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
from scheduler.utils import ensure_dir, now_utc_iso


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--mode", choices=["quick", "paper", "large"], default="quick")
    p.add_argument("--outdir", type=str, default="out")

    # scaling sweep controls
    p.add_argument("--jobs", nargs="*", type=int, default=None)
    p.add_argument("--regions", nargs="*", type=int, default=None)
    p.add_argument("--slots", type=int, default=12)
    p.add_argument("--scenarios", type=int, default=10)
    p.add_argument("--seeds", type=int, default=5)

    # model knobs
    p.add_argument("--beta", type=float, default=0.9)          # CVaR level
    p.add_argument("--eps", type=float, default=0.05)          # DRCC risk tolerance
    p.add_argument("--cvar_w", type=float, default=5.0)        # w4
    p.add_argument("--late_w", type=float, default=2.0)
    p.add_argument("--carbon_w", type=float, default=2.0)
    p.add_argument("--cost_w", type=float, default=1.0)
    p.add_argument("--place_w", type=float, default=0.2)

    # carbon budget tightening relative to baseline
    p.add_argument("--budget_frac", type=float, default=0.95)  # tau (budget fraction)
    p.add_argument("--budget_mode", choices=["cvar", "worst"], default="cvar")

    # runtime knobs
    p.add_argument("--dro_timeout_s", type=float, default=120.0)
    p.add_argument("--solver", choices=["ECOS", "SCS"], default="SCS")

    # sensitivity sweep controls
    p.add_argument("--do_sensitivity", action="store_true")
    p.add_argument("--sens_regions", type=int, default=3)
    p.add_argument("--sens_jobs", type=int, default=50)
    p.add_argument("--sens_slots", type=int, default=10)
    p.add_argument("--sens_scenarios", type=int, default=20)
    p.add_argument("--sens_seeds", type=int, default=5)
    p.add_argument("--sens_w4_grid", nargs="*", type=float, default=None)
    p.add_argument("--sens_tau_grid", nargs="*", type=float, default=None)

    return p.parse_args()


def mode_presets(args) -> Dict[str, Any]:
    if args.mode == "quick":
        return dict(regions=[2], jobs=[30], slots=8, scenarios=8, seeds=2)
    if args.mode == "paper":
        return dict(regions=[2, 4], jobs=[30, 60, 120], slots=args.slots, scenarios=args.scenarios, seeds=args.seeds)
    # large
    return dict(
        regions=args.regions if args.regions else [2, 4, 6, 8],
        jobs=args.jobs if args.jobs else [200, 500, 1000],
        slots=args.slots,
        scenarios=args.scenarios,
        seeds=args.seeds,
    )


# ----------------------------
# Metrics aggregation (robust)
# ----------------------------
def ci95(x: pd.Series) -> float:
    x = x.dropna()
    n = len(x)
    if n <= 1:
        return 0.0 if n == 1 else np.nan
    s = float(x.std(ddof=1))
    return float(1.96 * s / np.sqrt(n))


def aggregate(df_raw: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """
    Returns a flat-column dataframe with:
      <metric>_mean, <metric>_ci95, <metric>_n
    plus the group_cols.

    IMPORTANT: This avoids the buggy merge-based approach that caused:
      - duplicate columns like level_3_x
      - losing group columns
    """
    metrics = [
        "exp_cost",
        "exp_carbon",
        "cvar_carbon",
        "worst_carbon",
        "late_slots_mean",
        "late_jobs_frac",
        "carbon_budget",
        "solve_time_s",
    ]

    ok = df_raw["solve_status"].astype(str).str.startswith("ok") | (df_raw["solve_status"].astype(str) == "optimal")
    df_ok = df_raw[ok].copy()

    # Base group frame (unique groups) so group cols always exist
    groups = df_raw[group_cols].drop_duplicates().reset_index(drop=True)

    out = groups.copy()
    if df_ok.empty:
        # still return columns
        for m in metrics:
            out[f"{m}_mean"] = np.nan
            out[f"{m}_ci95"] = np.nan
            out[f"{m}_n"] = 0
        return out

    grp = df_ok.groupby(group_cols, dropna=False)

    for m in metrics:
        mean_s = grp[m].mean().rename(f"{m}_mean")
        ci_s = grp[m].apply(ci95).rename(f"{m}_ci95")
        n_s = grp[m].count().rename(f"{m}_n")
        block = pd.concat([mean_s, ci_s, n_s], axis=1).reset_index()
        out = out.merge(block, on=group_cols, how="left")

    return out


def summarize_status(df_raw: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    def is_ok(v: str) -> int:
        s = str(v)
        return int(s.startswith("ok") or s == "optimal")

    g = df_raw.groupby(group_cols, dropna=False)
    out = g.agg(
        runs=("solve_status", "count"),
        ok_runs=("solve_status", lambda x: int(np.sum([is_ok(v) for v in x]))),
    ).reset_index()
    out["ok_rate"] = out["ok_runs"] / out["runs"]
    return out


# ----------------------------
# Config + single run
# ----------------------------
def make_config_from_args(args) -> ExperimentConfig:
    return ExperimentConfig(
        beta=float(args.beta),
        eps=float(args.eps),
        cost_w=float(args.cost_w),
        carbon_w=float(args.carbon_w),
        cvar_w=float(args.cvar_w),
        place_w=float(args.place_w),
        late_w=float(args.late_w),
        budget_frac=float(args.budget_frac),
        budget_mode=str(args.budget_mode),
    )


def run_one(inst: Instance, cfg: ExperimentConfig, solver: str, timeout_s: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    baselines = {
        "baseline_cost": solve_cost_only_greedy,
        "baseline_carbon": solve_carbon_only_greedy,
        "baseline_weighted": solve_weighted_expected_greedy,
        "baseline_random": solve_random_feasible,
    }

    baseline_solutions: Dict[str, Any] = {}

    for name, fn in baselines.items():
        t0 = time.time()
        sol = fn(inst, cfg)
        t1 = time.time()
        m = compute_metrics(inst, sol)
        rows.append({
            "method": name,
            "solve_status": "ok",
            "solve_time_s": t1 - t0,
            "carbon_budget": np.nan,
            **m,
        })
        baseline_solutions[name] = (sol, m)

    ref = baseline_solutions["baseline_cost"][1]
    if cfg.budget_mode == "cvar":
        B = float(cfg.budget_frac) * float(ref["cvar_carbon"])
    else:
        B = float(cfg.budget_frac) * float(ref["worst_carbon"])

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
            "solve_status": str(status),
            "solve_time_s": t1 - t0,
            "carbon_budget": B,
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
            "solve_status": str(status),
            "solve_time_s": t1 - t0,
            "carbon_budget": B,
            **m,
        })

    return rows


# ----------------------------
# Runs: scaling + sensitivity
# ----------------------------
def run_scaling(args, cfg: ExperimentConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    preset = mode_presets(args)
    regions = preset["regions"]
    jobs = preset["jobs"]
    slots = preset["slots"]
    scenarios = preset["scenarios"]
    seeds = preset["seeds"]

    raw_rows: List[Dict[str, Any]] = []

    for I in regions:
        for J in jobs:
            for seed in tqdm(range(seeds), desc=f"I={I}, J={J}"):
                inst = gen_instance(I=I, T=slots, J=J, S=scenarios, seed=seed)
                rows = run_one(inst, cfg, solver=args.solver, timeout_s=args.dro_timeout_s)
                for r in rows:
                    r.update({"I": I, "J": J, "T": slots, "S": scenarios, "seed": seed})
                    r.update({
                        "beta": cfg.beta,
                        "eps": cfg.eps,
                        "cvar_w": cfg.cvar_w,
                        "budget_frac": cfg.budget_frac,
                        "budget_mode": cfg.budget_mode,
                    })
                raw_rows.extend(rows)

    df_raw = pd.DataFrame(raw_rows)
    df_agg = aggregate(df_raw, group_cols=["method", "I", "J"])
    df_status = summarize_status(df_raw, group_cols=["method", "I", "J"])
    return df_raw, df_agg, df_status


def run_sensitivity(args, cfg_base: ExperimentConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    I = int(args.sens_regions)
    J = int(args.sens_jobs)
    T = int(args.sens_slots)
    S = int(args.sens_scenarios)
    seeds = int(args.sens_seeds)

    w4_grid = args.sens_w4_grid if args.sens_w4_grid else [0.0, 2.0, 5.0, 10.0]
    tau_grid = args.sens_tau_grid if args.sens_tau_grid else [0.90, 0.95, 0.98]

    raw_rows: List[Dict[str, Any]] = []

    for w4 in w4_grid:
        for tau in tau_grid:
            cfg = replace(cfg_base, cvar_w=float(w4), budget_frac=float(tau))
            for seed in tqdm(range(seeds), desc=f"sens w4={w4}, tau={tau}"):
                inst = gen_instance(I=I, T=T, J=J, S=S, seed=seed)
                rows = run_one(inst, cfg, solver=args.solver, timeout_s=args.dro_timeout_s)
                for r in rows:
                    r.update({"I": I, "J": J, "T": T, "S": S, "seed": seed})
                    r.update({
                        "beta": cfg.beta,
                        "eps": cfg.eps,
                        "cvar_w": cfg.cvar_w,
                        "tau": cfg.budget_frac,
                        "budget_mode": cfg.budget_mode,
                    })
                raw_rows.extend(rows)

    df_raw = pd.DataFrame(raw_rows)
    df_agg = aggregate(df_raw, group_cols=["method", "cvar_w", "tau"])
    df_status = summarize_status(df_raw, group_cols=["method", "cvar_w", "tau"])
    return df_raw, df_agg, df_status


# ----------------------------
# Plotting (works without seaborn)
# ----------------------------
def plot_more_scaling(df_agg: pd.DataFrame, outdir: str):
    import matplotlib.pyplot as plt

    ensure_dir(outdir)

    # Scaling: CVaR carbon vs J for baseline_cost and dro
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sub = df_agg[df_agg["method"].isin(["baseline_cost", "dro"])].copy()
    if not sub.empty:
        for I in sorted(sub["I"].unique()):
            for method in ["baseline_cost", "dro"]:
                s = sub[(sub["I"] == I) & (sub["method"] == method)].sort_values("J")
                if s.empty:
                    continue
                ax.plot(s["J"].values, s["cvar_carbon_mean"].values, marker="o", label=f"I={I}, {method}")
        ax.set_title("Scaling: CVaR0.9(carbon) vs job count")
        ax.set_xlabel("Number of jobs (J)")
        ax.set_ylabel("CVaR0.9 carbon (kg)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "fig_scaling_cvar_carbon.png"), dpi=200)
    plt.close(fig)

    # Tradeoff: expected cost vs CVaR carbon
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if not sub.empty:
        for I in sorted(sub["I"].unique()):
            for method in ["baseline_cost", "dro"]:
                s = sub[(sub["I"] == I) & (sub["method"] == method)].sort_values("J")
                if s.empty:
                    continue
                ax.plot(s["exp_cost_mean"].values, s["cvar_carbon_mean"].values, marker="o", label=f"I={I}, {method}")
        ax.set_title("Tradeoff: expected cost vs CVaR0.9(carbon)")
        ax.set_xlabel("Expected cost")
        ax.set_ylabel("CVaR0.9 carbon (kg)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "fig_tradeoff_cost_vs_cvar_carbon.png"), dpi=200)
    plt.close(fig)

    # Percent reduction: dro vs baseline_cost (CVaR carbon)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    base = df_agg[df_agg["method"] == "baseline_cost"][["I", "J", "cvar_carbon_mean"]].copy()
    dro = df_agg[df_agg["method"] == "dro"][["I", "J", "cvar_carbon_mean"]].copy()
    if not base.empty and not dro.empty:
        m = base.merge(dro, on=["I", "J"], suffixes=("_base", "_dro"))
        m["pct_reduction"] = 100.0 * (m["cvar_carbon_mean_base"] - m["cvar_carbon_mean_dro"]) / m["cvar_carbon_mean_base"]
        for I in sorted(m["I"].unique()):
            s = m[m["I"] == I].sort_values("J")
            ax.plot(s["J"].values, s["pct_reduction"].values, marker="o", label=f"I={I}")
        ax.set_title("Tail carbon reduction: DRO vs baseline cost")
        ax.set_xlabel("Number of jobs (J)")
        ax.set_ylabel("Percent reduction in CVaR0.9 carbon")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "fig_cvar_reduction_pct.png"), dpi=200)
    plt.close(fig)


def plot_sensitivity_heatmap(df_sens_raw: pd.DataFrame, outdir: str):
    import matplotlib.pyplot as plt

    ensure_dir(outdir)

    dro = df_sens_raw[df_sens_raw["method"] == "dro"].copy()
    if dro.empty:
        return

    ok = dro["solve_status"].astype(str).str.startswith("ok") | (dro["solve_status"].astype(str) == "optimal")
    dro = dro[ok].copy()
    if dro.empty:
        return

    if "tau" not in dro.columns:
        dro["tau"] = dro.get("budget_frac", np.nan)

    pivot = dro.pivot_table(
        index="tau",
        columns="cvar_w",
        values="cvar_carbon",
        aggfunc="mean",
    )
    if pivot.empty:
        return

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(pivot.values, aspect="auto", origin="lower")

    ax.set_title("Sensitivity (DRO): mean CVaR0.9(carbon)")
    ax.set_xlabel("CVaR weight w4")
    ax.set_ylabel("Robustness tau (budget fraction)")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(v) for v in pivot.columns])

    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(v) for v in pivot.index])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean CVaR0.9 carbon")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_sensitivity_heatmap.png"), dpi=200)
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()
    ensure_dir(args.outdir)

    cfg = make_config_from_args(args)

    meta = {
        "timestamp_utc": now_utc_iso(),
        "mode": args.mode,
        "solver": args.solver,
        "dro_timeout_s": args.dro_timeout_s,
        "config": asdict(cfg),
        "scaling": mode_presets(args),
        "sensitivity": {
            "enabled": bool(args.do_sensitivity or args.mode in ["paper", "large"]),
            "sens_regions": args.sens_regions,
            "sens_jobs": args.sens_jobs,
            "sens_slots": args.sens_slots,
            "sens_scenarios": args.sens_scenarios,
            "sens_seeds": args.sens_seeds,
            "sens_w4_grid": args.sens_w4_grid,
            "sens_tau_grid": args.sens_tau_grid,
        },
    }
    with open(os.path.join(args.outdir, "run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # scaling
    df_raw, df_agg, df_status = run_scaling(args, cfg)

    raw_path = os.path.join(args.outdir, "metrics_raw.csv")
    agg_path = os.path.join(args.outdir, "metrics_agg.csv")
    status_path = os.path.join(args.outdir, "solve_status_summary.csv")

    df_raw.to_csv(raw_path, index=False)
    df_agg.to_csv(agg_path, index=False)
    df_status.to_csv(status_path, index=False)

    # console view (only if columns exist)
    base_cols = ["method", "I", "J"]
    metric_cols = [
        "exp_cost_mean",
        "exp_carbon_mean",
        "cvar_carbon_mean",
        "worst_carbon_mean",
        "late_jobs_frac_mean",
        "solve_time_s_mean",
    ]
    cols = [c for c in base_cols + metric_cols if c in df_agg.columns]
    view = df_agg[cols].copy()

    print("\nAggregated results (means):")
    print(tabulate(view, headers="keys", tablefmt="github", floatfmt=".4f"))

    # plots
    plot_more_scaling(df_agg, outdir=args.outdir)

    # sensitivity (paper/large or explicit)
    do_sens = bool(args.do_sensitivity or args.mode in ["paper", "large"])
    if do_sens:
        df_sens_raw, df_sens_agg, df_sens_status = run_sensitivity(args, cfg)

        sens_raw_path = os.path.join(args.outdir, "metrics_sensitivity_raw.csv")
        sens_agg_path = os.path.join(args.outdir, "metrics_sensitivity_agg.csv")
        sens_status_path = os.path.join(args.outdir, "solve_status_sensitivity.csv")

        df_sens_raw.to_csv(sens_raw_path, index=False)
        df_sens_agg.to_csv(sens_agg_path, index=False)
        df_sens_status.to_csv(sens_status_path, index=False)

        plot_sensitivity_heatmap(df_sens_raw, outdir=args.outdir)

    print(f"\nSaved: {raw_path}")
    print(f"Saved: {agg_path}")
    print(f"Saved: {status_path}")
    if do_sens:
        print(f"Saved: {os.path.join(args.outdir, 'metrics_sensitivity_raw.csv')}")
        print(f"Saved: {os.path.join(args.outdir, 'metrics_sensitivity_agg.csv')}")
        print(f"Saved: {os.path.join(args.outdir, 'solve_status_sensitivity.csv')}")
        print(f"Saved: {os.path.join(args.outdir, 'fig_sensitivity_heatmap.png')}")
    print(f"Figures saved under: {args.outdir}/")


if __name__ == "__main__":
    main()
