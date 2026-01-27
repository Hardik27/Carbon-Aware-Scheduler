from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_illustrative_bars(df_raw: pd.DataFrame, outdir: str):
    # pick one illustrative setting
    sub = df_raw[(df_raw["I"] == 2) & (df_raw["J"] == df_raw["J"].min()) & (df_raw["seed"] == 0)]
    if len(sub) == 0:
        return

    methods = ["baseline_cost", "dro"]
    sub = sub[sub["method"].isin(methods)].copy()

    metrics = ["exp_cost", "exp_carbon", "cvar_carbon"]
    labels = ["Expected Cost", "Expected Carbon", "CVaR0.9 Carbon"]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, m in enumerate(methods):
        row = sub[sub["method"] == m]
        if len(row) == 0:
            continue
        vals = [float(row[k].iloc[0]) for k in metrics]
        ax.bar(x + (i - 0.5) * width, vals, width, label=m)

    ax.set_title("Illustrative comparison (I=2, one seed)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = os.path.join(outdir, "fig_illustrative_bar.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_scaling_lines(df_agg: pd.DataFrame, outdir: str):
    # expects metrics_agg style columns
    # plot CVaR carbon mean vs job count for I=2 and I=4
    if "cvar_carbon_mean" not in df_agg.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    for I in sorted(df_agg["I"].unique()):
        for method in ["baseline_cost", "dro"]:
            sub = df_agg[(df_agg["I"] == I) & (df_agg["method"] == method)].sort_values("J")
            if len(sub) == 0:
                continue
            ax.plot(sub["J"], sub["cvar_carbon_mean"], marker="o", label=f"I={I}, {method}")

    ax.set_title("Scaling behavior: CVaR0.9(carbon) vs job count")
    ax.set_xlabel("Number of jobs (J)")
    ax.set_ylabel("CVaR0.9 carbon (kg)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()

    path = os.path.join(outdir, "fig_scaling_lines.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_sensitivity_heatmap(df_raw: pd.DataFrame, outdir: str):
    # Only if those columns exist (optional)
    # If you later add sweeps over cfg.cvar_w and cfg.eps, wire them into df_raw and this will work.
    if "cvar_w" not in df_raw.columns or "eps" not in df_raw.columns:
        return

    sub = df_raw[df_raw["method"] == "dro"].copy()
    if len(sub) == 0:
        return

    piv = sub.pivot_table(index="eps", columns="cvar_w", values="cvar_carbon", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(piv.values, aspect="auto", origin="lower")
    ax.set_xticks(np.arange(piv.shape[1]))
    ax.set_xticklabels([str(c) for c in piv.columns])
    ax.set_yticks(np.arange(piv.shape[0]))
    ax.set_yticklabels([str(r) for r in piv.index])
    ax.set_xlabel("CVaR weight")
    ax.set_ylabel("eps (DRCC tolerance)")
    ax.set_title("Sensitivity: DRO carbon CVaR")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    path = os.path.join(outdir, "fig_sensitivity.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
