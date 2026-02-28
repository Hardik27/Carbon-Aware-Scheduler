# README v2: Carbon-Aware Scheduling Experiments

This repository contains two Python scripts for running experiments that compare
six cloud scheduling policies — from a simple cost-minimizing greedy to a full
**Distributionally Robust Optimization (DRO)-based carbon-aware scheduler** — and
for collecting real CAISO/WattTime grid data used in the real-data validation
experiment. The codebase supports the JoCC submission:
*"Carbon-Aware Scheduling and Distributionally Robust Optimization for Cloud Systems"*.

---

## Repository Contents

| File | Purpose |
|---|---|
| `scheduler_v4_2.py` | Main experiment script (v4.2, JoCC revision) |
| `watttime_data_collection_v2.py` | Real CAISO + WattTime data fetcher (v2) |
| `README.md` | This file |

---

## 1. Motivation

Cloud data centers consume significant energy whose carbon footprint varies by
region, time of day, and grid generation mix. Traditional schedulers focus solely
on monetary cost without accounting for carbon intensity or market uncertainty.
This framework:

- Incorporates **carbon-aware scheduling** with hard deadline and capacity constraints.
- Models uncertainty via **Distributionally Robust Optimization (DRO)** with adaptive Wasserstein ambiguity sets.
- Quantifies tail-risk exposure via **CVaR₀.₉(Carbon)**.
- Benchmarks DRO against five competing policies (CM, CF, RF, SBS, RO) over 1,000 out-of-distribution stress scenarios.
- Validates the framework on **real CAISO/WattTime data** (August 5, 2025).

---

## 2. scheduler_v4_2.py

### What's New in v4.2 (vs. v4.1)

| Fix | Description |
|---|---|
| FIX-1 | Illustrative experiment now uses S=20 scenarios; CVaR₀.₉₅ column removed (with S=8 the 90th/95th percentiles collapsed) |
| FIX-2 | Integer-relaxation disclaimer printed in results header and written to CSV notes |
| FIX-3 | Real-data validation experiment added using embedded CAISO/WattTime-calibrated traces |
| FIX-4 | Pareto frontier swept on larger config (I=4, T=24, J=60) for a wider, more informative frontier |
| FIX-5 | Battery section clearly labelled as heuristic dispatch; compute-cost table added and written to CSV |

### Six Scheduling Methods

| Method | Description |
|---|---|
| **CM** | Cost-minimizing greedy |
| **CF** | Carbon-first greedy |
| **RF** | Renewable-first greedy |
| **SBS** | Score-based scheduler (balanced linear combination) |
| **RO** | Robust optimization (1.5σ safety margin) |
| **DRO** | Proposed: adaptive Wasserstein ambiguity set + CVaR + TinyLSTM forecasts |

### Experiments Produced

| # | Experiment | Config | Output CSV | Key Figures |
|---|---|---|---|---|
| 1 | Illustrative comparison | I=2, T=8, J=20, S=20 | `metrics_illustrative.csv` | plot4, plot5, plot9, plot11 |
| 2 | Scale sweep (2 regions) | I=2, T=12, J∈{30,60,120} | `metrics_scale_sweep.csv` | plot7 |
| 3 | Scale sweep (4 regions) | I=4, T=12, J∈{30,60,120} | `metrics_scale_sweep.csv` | plot7 |
| 4 | Sensitivity analysis | I=3, T=10, J=50 | `metrics_sensitivity.csv` | plot8 |
| 5 | Six-method benchmark | I=4, T=24, J=100, N=1000 OOD | `metrics_comparative.csv` | plot1, plot2, plot10 |
| 6 | Pareto frontier | I=3, T=12, J=40 | `metrics_pareto.csv` | plot3 |
| 7 | Battery experiment | I=3, T=24, J=80 | `metrics_battery.csv` | plot6, plot12 |
| 8 | Real-data validation | I=3, T=24, J=60, N=100 OOD | `metrics_real_data.csv` | plot13 |
| 9 | Compute cost scaling | 5 configs × 6 methods | `metrics_compute_cost.csv` | plot14 |

### Fourteen Output Figures

| File | Description |
|---|---|
| `plot1_comparative_barplot.png` | Grouped bar chart: all 4 metrics for 6 methods |
| `plot2_cost_carbon_scatter.png` | Cost–carbon trade-off scatter (Pareto structure) |
| `plot3_pareto_frontier.png` | DRO Pareto frontier (cost vs carbon) |
| `plot4_temporal_heatmap.png` | Grid energy heatmap by DC and time slot |
| `plot5_renewable_utilization.png` | Energy mix (grid vs renewable) over time |
| `plot6_battery_dynamics.png` | Battery SoC and charge/discharge profiles |
| `plot7_cvar_analysis.png` | CVaR₀.₉ carbon scaling with job count |
| `plot8_sensitivity_heatmap.png` | Sensitivity of cost and CVaR to w₄ and τ |
| `plot9_uncertainty_quantification.png` | Carbon intensity and price uncertainty bands |
| `plot10_violin_distribution.png` | Violin distributions of carbon across 1000 OOD scenarios |
| `plot11_convergence_analysis.png` | Wasserstein radius convergence over epochs |
| `plot12_job_scheduling_gantt.png` | Gantt chart for first 20 jobs (battery experiment) |
| `plot13_real_data_comparison.png` | Real-data validation: 6 methods on CAISO/WattTime data |
| `plot14_compute_cost.png` | Runtime scaling with job count (log scale) |

### Installation

```bash
pip install numpy pandas cvxpy scipy tqdm tabulate matplotlib torch
```

Optional: install a commercial solver (Gurobi, CPLEX, Mosek) for larger instances.
The default open-source cascade is ECOS → SCS → CLARABEL → OSQP.

### Usage

```bash
# Default run (greedy DRO path, 1000 OOD benchmark scenarios)
python scheduler_v4_2.py

# Full DRO with CVXPY solver and LSTM forecasts
python scheduler_v4_2.py --dro-solver cvxpy --use-lstm

# Custom output directory, fewer benchmark scenarios
python scheduler_v4_2.py --out my_results --n-bench 200 --n-real 50

# Set random seed
python scheduler_v4_2.py --seed 42
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--out` | `out_v4_2` | Output directory for CSVs and figures |
| `--n-bench` | `1000` | Number of OOD scenarios for comparative benchmark |
| `--n-real` | `100` | Number of OOD scenarios for real-data experiment |
| `--use-lstm` | off | Enable TinyLSTM forecasts (falls back to EWMA if PyTorch missing) |
| `--dro-solver` | `greedy` | DRO solve path: `greedy` (fast) or `cvxpy` (exact convex relaxation) |
| `--seed` | `7` | Global random seed |

### Output Structure

```
out_v4_2/
├── metrics_illustrative.csv
├── metrics_scale_sweep.csv
├── metrics_sensitivity.csv
├── metrics_comparative.csv
├── metrics_pareto.csv
├── metrics_battery.csv
├── metrics_real_data.csv
├── metrics_compute_cost.csv
└── plots/
    ├── plot1_comparative_barplot.png
    ├── plot2_cost_carbon_scatter.png
    ├── ... (14 figures total)
    └── plot14_compute_cost.png
```

---

## 3. watttime_data_collection_v2.py

This script downloads the real grid data used in the FIX-3 real-data validation
experiment (Section 9.4 of the paper). It requires no API key for CAISO OASIS but
requires a WattTime free-tier token for carbon intensity.

### Data Sources

| Signal | Source | Region/Node | Resolution |
|---|---|---|---|
| Carbon intensity (MOER) | WattTime API | `CAISO_NORTH` | 5-min → hourly |
| Electricity price (LMP) | CAISO OASIS API | `TH_NP15_GEN-APND` | Hourly |
| Renewable fraction | Derived from CAISO OASIS (solar + wind share) | CAISO balancing area | Hourly |

All three data centers are modelled within the **CAISO NP15 pricing zone** with
small calibrated signal offsets (±2–4%) to simulate intra-zone variation.

### Setup

1. Obtain a free WattTime token:
   ```bash
   curl -u your_username:your_password https://api.watttime.org/login
   ```
2. Paste the token into the script:
   ```python
   WATTTIME_TOKEN = "your_token_here"
   ```
3. Install dependencies:
   ```bash
   pip install requests pandas numpy
   ```

### Usage

```bash
python watttime_data_collection_v2.py
```

### Output

The script prints three ready-to-paste NumPy arrays (carbon, price, renewable)
and a LaTeX footnote for the paper. Paste the arrays into the
`run_real_data_experiment()` function in `scheduler_v4_2.py` to replace the
embedded fallback traces.

### Target Date

`August 5, 2025` — a representative summer peak-demand weekday on the CAISO grid.

---

## 4. Known Limitations

- The DRO solver path (`--dro-solver cvxpy`) solves a convex relaxation of the MILP
  and projects via greedy one-hot assignment; global integer optimality is not guaranteed.
- Battery experiments use the carbon-threshold heuristic (Remark 1 in the paper),
  not the full joint DRO optimization over scheduling + storage dispatch simultaneously.
- Synthetic traces use AR(1) dynamics; real-world variability is richer.
- The EWMA adaptive refinement assumes approximate stationarity; abrupt grid
  regime changes (e.g., large new renewable capacity) may temporarily degrade performance.
- Real-data experiment covers CAISO only; broader validation (PJM, ERCOT) is left for future work.

---

## 5. Reproducibility

All experiments use fixed random seeds (`np.random.seed(42)` globally; per-experiment
seeds controllable via `--seed`). Results reported in the paper use:

```bash
python scheduler_v4_2.py --dro-solver cvxpy --use-lstm --n-bench 1000 --n-real 100 --seed 7
```

---

## 6. Citation

If you use this code in academic work, please cite:

> Ruparel, H. (2025). *Carbon-Aware Scheduling and Distributionally Robust Optimization
> for Cloud Systems*. Journal of Cloud Computing (JoCC). Springer Nature.

---

## 7. License

This code is intended for academic and research use.
