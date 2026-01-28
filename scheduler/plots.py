from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _save(fig, outdir: str, name: str):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_illustrative_bars_with_ci(df_raw: pd.DataFrame, df_agg: pd.DataFrame, outdir: str):
    # choose smallest I,J for a clean illustrative plot
    ij = df_raw[["I", "J"]].drop_duplicates().sort_values(["I", "J"]).iloc[0]
    I, J = int(ij.I), int(ij.J)

    # use aggregated means and ci for baseline_cost vs dro
    sub = df_agg[(df_agg["I"] == I) & (df_agg["J"] == J) & (df_agg["method"].isin(["baseline_cost", "dro"]))].copy()
    if len(sub) == 0:
        return

    metrics = [
        ("exp_cost", "Expected Cost"),
        ("exp_carbon", "Expected Carbon"),
        ("cvar_carbon", "CVaR 0.9 Carbon"),
    ]

    labels = ["baseline_cost", "dro"]
    x = np.arange(len(metrics))
    width = 0.35

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)

    for idx, method in enumerate(labels):
        mrow = sub[sub["method"] == method]
        if len(mrow) == 0:
            continue

        vals = []
        errs = []
        for m, _ in metrics:
            vals.append(float(mrow[f"{m}_mean"].iloc[0]))
            errs.append(float(mrow[f"{m}_ci95"].iloc[0]))

        ax.bar(x + (idx - 0.5) * width, vals, width, yerr=errs, capsize=4, label=method)

    ax.set_xticks(x)
    ax.set_xticklabels([t for _, t in metrics])
    ax.set_title(f"Illustrative comparison (I={I}, J={J}, mean and 95% CI over seeds)")
    ax.set_ylabel("Metric value")
    ax.legend()

    _save(fig, outdir, "fig_illustrative_ci.png")


def plot_scaling_cvar_with_ci(df_agg: pd.DataFrame, outdir: str):
    # lines of CVaR carbon vs job count, with CI whiskers if available
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    methods = ["baseline_cost", "dro"]
    for I in sorted(df_agg["I"].dropna().unique()):
        for method in methods:
            sub = df_agg[(df_agg["I"] == I) & (df_agg["method"] == method)].sort_values("J")
            if len(sub) == 0:
                continue
            ax.errorbar(
                sub["J"].values,
                sub["cvar_carbon_mean"].values,
                yerr=sub["cvar_carbon_ci95"].values,
                marker="o",
                linestyle="-",
                capsize=3,
                label=f"I={int(I)}, {method}",
            )

    ax.set_title("Scaling behavior: CVaR 0.9 carbon vs job count (mean and 95% CI)")
    ax.set_xlabel("Number of jobs (J)")
    ax.set_ylabel("CVaR 0.9 carbon")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)

    _save(fig, outdir, "fig_scaling_cvar_ci.png")


def plot_cost_overhead_pct(df_agg: pd.DataFrame, outdir: str):
    # percent overhead of DRO cost relative to baseline_cost
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    for I in sorted(df_agg["I"].dropna().unique()):
        b = df_agg[(df_agg["I"] == I) & (df_agg["method"] == "baseline_cost")].sort_values("J")
        d = df_agg[(df_agg["I"] == I) & (df_agg["method"] == "dro")].sort_values("J")
        if len(b) == 0 or len(d) == 0:
            continue
        m = b.merge(d, on=["I", "J"], suffixes=("_b", "_d"))
        pct = 100.0 * (m["exp_cost_mean_d"] - m["exp_cost_mean_b"]) / m["exp_cost_mean_b"]
        ax.plot(m["J"], pct, marker="o", label=f"I={int(I)}")

    ax.set_title("Cost overhead of DRO relative to baseline cost only")
    ax.set_xlabel("Number of jobs (J)")
    ax.set_ylabel("Cost overhead percent")
    ax.grid(True, alpha=0.3)
    ax.legend()

    _save(fig, outdir, "fig_cost_overhead_pct.png")


def plot_cvar_reduction_pct(df_agg: pd.DataFrame, outdir: str):
    # percent reduction in CVaR carbon relative to baseline_cost
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    for I in sorted(df_agg["I"].dropna().unique()):
        b = df_agg[(df_agg["I"] == I) & (df_agg["method"] == "baseline_cost")].sort_values("J")
        d = df_agg[(df_agg["I"] == I) & (df_agg["method"] == "dro")].sort_values("J")
        if len(b) == 0 or len(d) == 0:
            continue
        m = b.merge(d, on=["I", "J"], suffixes=("_b", "_d"))
        pct = 100.0 * (m["cvar_carbon_mean_b"] - m["cvar_carbon_mean_d"]) / m["cvar_carbon_mean_b"]
        ax.plot(m["J"], pct, marker="o", label=f"I={int(I)}")

    ax.set_title("CVaR 0.9 carbon reduction of DRO relative to baseline cost only")
    ax.set_xlabel("Number of jobs (J)")
    ax.set_ylabel("CVaR reduction percent")
    ax.grid(True, alpha=0.3)
    ax.legend()

    _save(fig, outdir, "fig_cvar_reduction_pct.png")


def plot_tradeoff_scatter(df_raw: pd.DataFrame, outdir: str):
    # scatter cost vs cvar carbon for dro vs baseline cost
    sub = df_raw[df_raw["method"].isin(["baseline_cost", "dro"])].copy()
    sub = sub[sub["solve_status"].astype(str).str.startswith("ok")]

    if len(sub) == 0:
        return

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)

    for method in ["baseline_cost", "dro"]:
        m = sub[sub["method"] == method]
        ax.scatter(m["exp_cost"], m["cvar_carbon"], label=method, alpha=0.7)

    ax.set_title("Empirical tradeoff: expected cost vs CVaR 0.9 carbon")
    ax.set_xlabel("Expected cost")
    ax.set_ylabel("CVaR 0.9 carbon")
    ax.grid(True, alpha=0.3)
    ax.legend()

    _save(fig, outdir, "fig_tradeoff_scatter.png")


def plot_runtime_scaling(df_raw: pd.DataFrame, outdir: str, tag: str = "scaling"):
    # runtime vs J, show dro and baselines
    ok = df_raw.copy()
    ok = ok[ok["solve_status"].astype(str).str.startswith("ok")]

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    for method in ["baseline_cost", "baseline_carbon", "baseline_weighted", "baseline_random", "dro"]:
        sub = ok[ok["method"] == method]
        if len(sub) == 0:
            continue
        grp = sub.groupby(["I", "J"])["solve_time_s"].mean().reset_index()
        for I in sorted(grp["I"].unique()):
            g = grp[grp["I"] == I].sort_values("J")
            ax.plot(g["J"], g["solve_time_s"], marker="o", label=f"{method}, I={int(I)}")

    ax.set_title(f"Runtime scaling (mean over seeds), tag={tag}")
    ax.set_xlabel("Number of jobs (J)")
    ax.set_ylabel("Solve time seconds")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)

    _save(fig, outdir, f"fig_runtime_{tag}.png")


def plot_feasibility_rate(df_raw: pd.DataFrame, outdir: str, tag: str = "scaling"):
    # feasibility rate of dro over seeds
    df = df_raw[df_raw["method"] == "dro"].copy()
    if len(df) == 0:
        return

    df["is_ok"] = df["solve_status"].astype(str).str.startswith("ok").astype(int)
    grp = df.groupby(["I", "J"])["is_ok"].mean().reset_index()

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    for I in sorted(grp["I"].unique()):
        g = grp[grp["I"] == I].sort_values("J")
        ax.plot(g["J"], g["is_ok"], marker="o", label=f"I={int(I)}")

    ax.set_title(f"DRO feasibility rate over seeds, tag={tag}")
    ax.set_xlabel("Number of jobs (J)")
    ax.set_ylabel("Fraction solved ok")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()

    _save(fig, outdir, f"fig_feasibility_{tag}.png")


def plot_sensitivity_heatmap(df_sens_raw: pd.DataFrame, outdir: str):
    # expects a sensitivity run with columns eps and cvar_w
    df = df_sens_raw.copy()
    df = df[(df["method"] == "dro") & (df["solve_status"].astype(str).str.startswith("ok"))]
    if len(df) == 0:
        return

    # aggregate over seeds
    grp = df.groupby(["eps", "cvar_w"])["cvar_carbon"].mean().reset_index()
    pivot = grp.pivot(index="eps", columns="cvar_w", values="cvar_carbon").sort_index()

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_title("Sensitivity heatmap: DRO CVaR 0.9 carbon")
    ax.set_xlabel("cvar_w")
    ax.set_ylabel("eps")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{v:g}" for v in pivot.columns], rotation=45, ha="right")

    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{v:.3f}" for v in pivot.index])

    fig.colorbar(im, ax=ax, label="CVaR 0.9 carbon")

    _save(fig, outdir, "fig_sensitivity_heatmap.png")
