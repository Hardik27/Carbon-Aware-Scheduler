# README: Carbon-Aware Scheduling Experiments

This repository contains a self-contained Python script for running experiments that compare a **baseline cost-only cloud scheduler** with a **distributionally robust optimization (DRO)-based carbon-aware scheduler**. The purpose is to evaluate trade-offs between cost, carbon footprint, and risk-aware scheduling metrics such as **Conditional Value-at-Risk (CVaR)**.

---

## 1. Motivation

Cloud computing consumes significant energy, and its carbon footprint depends on the regional energy mix. Traditional schedulers focus primarily on minimizing monetary costs, without explicitly accounting for carbon intensity or uncertainty in energy markets. This repository provides an experimental framework to:

- Incorporate **carbon-aware scheduling** into decision-making.
- Model uncertainty using **Distributionally Robust Optimization (DRO)**.
- Quantify not just expected carbon emissions, but also **tail-risk exposure** via CVaR.
- Compare outcomes against a simple greedy baseline scheduler.

The code is designed as a bridge between theory and reproducible experimental evaluation for academic and applied research.

---

## 2. Features

The script implements the following capabilities:

1. **Synthetic Data Generation**
   - Multiple cloud regions (data centers).
   - Time slots with stochastic variations in price, carbon intensity, and renewable energy availability.
   - Job workloads with release times, deadlines, and processing demands.
   - Latency and placement penalties.

2. **Schedulers Implemented**
   - **Baseline Greedy Scheduler**: Minimizes expected cost with mild penalties for carbon and placement. Uses a simple heuristic.
   - **DRO Scheduler**: Uses convex optimization (via CVXPY) to:
     - Enforce capacity and job deadline constraints.
     - Model uncertainty in carbon intensity using a conservative DRO surrogate.
     - Incorporate CVaR (tail-risk control) for carbon emissions.
     - Balance multiple objectives (cost, carbon, placement, CVaR).

3. **Metrics Reported**
   - **Expected Cost**: Average monetary expenditure on energy.
   - **Expected Carbon**: Average carbon emissions across scenarios.
   - **CVaR₀.₉(Carbon)**: Conditional Value-at-Risk at 90%, measuring the worst-case tail behavior of carbon emissions.

4. **Experiments Included**
   - **Illustrative Case**: Small-scale example comparing baseline and DRO.
   - **Scale Sweep**: Varying the number of jobs and regions to test scalability and robustness.
   - **Sensitivity Analysis**: Varying CVaR weight and conservatism parameter (τ) to explore trade-offs between cost and carbon tail-risk.

---

## 3. Installation

Requirements:
- Python 3.10+
- Required packages: numpy, pandas, cvxpy, scipy, tqdm, tabulate

Install dependencies with:

pip install numpy pandas cvxpy scipy tqdm tabulate

Optional: Install a commercial solver (e.g., Gurobi, CPLEX, Mosek) to handle larger problem sizes more efficiently. The included setup defaults to open-source solvers ECOS and OSQP, which may be slower.

---

## 4. Usage

Run the main script:

python carbon_aware_dro_experiments.py

This will automatically execute the following experiments:

1. **Illustrative Comparison**
   - Runs a baseline vs. DRO comparison on a small instance (20 jobs, 2 regions, 8 time slots).
   - Produces a CSV file: `out/metrics_illustrative.csv`
   - Prints a table to the console comparing expected cost, expected carbon, and CVaR.

2. **Scale Sweep**
   - Runs experiments with job counts {30, 60, 120} and regions {2, 4}.
   - Evaluates performance across multiple random seeds.
   - Produces a CSV file: `out/metrics_scale_sweep.csv`
   - Prints aggregated results (mean values across seeds).

3. **Sensitivity Analysis**
   - Tests how increasing the CVaR weight and DRO conservatism parameter τ affect outcomes.
   - Produces a CSV file: `out/metrics_sensitivity.csv`
   - Prints mean results across seeds.

---

## 5. Output Files

All results are saved under the `out/` directory:

- `metrics_illustrative.csv`: Results for the illustrative baseline vs DRO experiment.
- `metrics_scale_sweep.csv`: Results across varying scales of jobs and regions.
- `metrics_sensitivity.csv`: Results for parameter sensitivity experiments.

Each CSV contains the following fields:
- exp: experiment type (baseline or DRO).
- I: number of regions.
- J: number of jobs.
- exp_cost: expected monetary cost.
- exp_carbon: expected carbon emissions.
- cvar90_carbon: 90% CVaR of carbon emissions.

---

## 6. Expected Results (Qualitative)

- **Illustrative Experiment**: DRO achieves lower carbon tail-risk (CVaR) than baseline with a modest cost increase.
- **Scale Sweep**: DRO continues to show reduced tail-risk at larger scales, while remaining tractable.
- **Sensitivity Analysis**:
  - Increasing the CVaR weight results in progressively lower CVaR values.
  - Increasing τ (distributional robustness conservatism) leads to more conservative (safer) solutions, but at higher expected cost.

These qualitative outcomes align with the theoretical expectations of distributionally robust optimization.

---

## 7. Extensions

You can extend this repository in the following ways:

- **Larger-scale experiments**: Increase the number of jobs, regions, and scenarios to simulate real-world conditions.
- **Visualization**: Add plotting (e.g., matplotlib) for CVaR vs cost trade-offs.
- **Integration with real-world data**: Replace synthetic carbon and price traces with real datasets (e.g., Google cluster traces, electricity grid carbon intensity APIs).
- **Hybrid heuristics**: Compare DRO against reinforcement learning, metaheuristics, or rule-based schedulers.

---

## 8. Limitations

- Current experiments use synthetic data; real-world variability is richer and more complex.
- The DRO model is convex-relaxed (x ∈ [0,1]), so solutions may need rounding for strict integrality.
- Computational performance may degrade for very large problem sizes without a commercial solver.
- Renewable energy dynamics are simplified.

---

## 9. Citation

If you use this framework in academic work, please cite it as part of the associated research paper on **carbon-aware cloud scheduling using DRO**.

---

## 10. License

This code is intended for academic and research use. Please adapt and extend as needed for your own work.

