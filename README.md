# 🦪 Network Analysis for Oyster Reef Restoration

This repository contains the full experimental and modeling framework for **Oyster Reef Site Selection Optimization** — integrating biological modeling, heuristic algorithms, and mixed-integer quadratic programming (MIQP).  
Developed under the supervision of **Dr. Rex Kincaid** (William & Mary), the project combines **ODE-based ecological simulations** and **network optimization** to inform data-driven coastal restoration planning.

---

## 🌍 Background and Motivation

The Chesapeake Bay once hosted over **10 billion oysters**, but overharvesting, habitat loss, and pollution have reduced populations by over **99%**.  
Restoration projects are costly, and only a limited number of degraded reefs can be rebuilt. The challenge:  
> **Which subset of reef sites should be restored to maximize long-term oyster population recovery and ecosystem services?**

Because oyster larvae disperse via **ocean currents**, each reef’s success depends not only on its own productivity but also on its **connectivity** with others. This creates a **network optimization problem** — selecting reefs that maximize both **local survival** and **system-wide larval retention**.

---

## 🧠 Project Overview

This repository provides a modular, reproducible pipeline for solving the **Oyster Reef Site Selection Problem (ORSSP)**.  
It integrates three modeling layers:

1. **ODE Connectivity Model (`src/model/jars_ode.py`)**  
   - Simulates larval dispersal among reef sites.  
   - Produces matrices for external input (`P₀`) and internal connectivity (`P₁`).  
   - Models four coupled life stages: juveniles, adults, shell/reef, and sediment.

2. **Heuristic Optimization (`src/opt/`)**  
   - Fast search algorithms to find near-optimal site combinations:
     - Forward **Greedy Selection**
     - **1-Swap Hill Climb**
     - Reverse **Stingy Backward** removal

3. **Exact Optimization via AMPL (`/ampl/`)**  
   - Quadratic programming models to find provably optimal site sets under:
     - Base network retention (MIQP)
     - Community coverage constraints
     - Reef sizing and total budget limits

---

## 🧩 Repository Structure

```
NetworkAnalysisOysters/
│
├── ampl/                          # AMPL models and data files
│   ├── oyster_quad.mod/dat         # Base MIQP (select K sites)
│   ├── oyster_comm.mod/dat         # MIQP + community constraints
│   ├── oyster_size.mod/dat         # MIQP + reef sizing (budget)
│   ├── oyster_comm_size.mod/dat    # MIQP + reef sizing + community constraints
│
├── data/
│   └── nk_All_060102final_56sites_Model.xlsx  # Larval connectivity and site metadata
│
├── figures/                       # Auto-generated figures
│   ├── fig_strategy_comparison.png # Comparison of optimization strategies
│   ├── fig_reef_sizes_miqp_size.png # Reef sizes under MIQP sizing model
│
├── runs/                          # Results and logs from all experiments
│   ├── *_sites.csv                 # Selected sites (IDs and sizes)
│   ├── *_summary.txt               # Objective summaries and solver logs
│   └── miqp_validated_with_ode.txt # Validation of MIQP results via ODE
│
├── scripts/                       # Entry-point scripts for each experiment
│   ├── run_greedy.py               # Greedy heuristic
│   ├── run_backward.py             # Stingy backward selection
│   ├── run_miqp.py                 # Base MIQP
│   ├── run_miqp_comm.py            # MIQP + communities
│   ├── run_miqp_size.py            # MIQP + reef sizing
│   ├── run_miqp_comm_size.py       # MIQP + community + sizing
│   ├── prepare_miqp_data.py        # Converts Excel to AMPL data format
│   ├── make_figures.py             # Generates plots and figures
│   ├── validate_miqp_with_ode.py   # Runs ODE to validate MIQP outcomes
│
├── src/                           # Core algorithms and biological model
│   ├── model/jars_ode.py           # Coupled ODE life stage model (Professor Leah Shaw's Work, she provided the math lab code i just converted to python)
│   ├── opt/                        # Optimization heuristics
│   │   ├── greedy.py               # Forward selection
│   │   ├── local_search.py         # Swap & tabu-based refinement
│   │   ├── backward.py             # Reverse elimination
│   │   └── evaluator.py            # Scoring functions
│   ├── utils/io_utils.py           # Input-output handlers
│   └── viz/plots.py                # Visualization and analysis
│
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation (this file)
```

---

## ⚙️ Installation and Setup

1. **Clone the repository**  
   ```bash
   git clone https://github.com/maroufpaul/NetworkAnalysisOysters.git
   cd NetworkAnalysisOysters
   ```

2. **Create a virtual environment**  
   ```bash
   conda create -n oysters python=3.10
   conda activate oysters
   pip install -r requirements.txt
   ```

3. **Configure AMPL**  
   - Ensure AMPL binaries and Gurobi are available in PATH.  
   - Test AMPL integration:
     ```bash
     python -m scripts.check_ampl_env
     ```

---

## 🚀 Running the Models

| Command | Model | Description |
|----------|--------|-------------|
| `python -m scripts.run_miqp` | Base MIQP | Maximizes internal larval retention (no constraints) |
| `python -m scripts.run_miqp_comm` | MIQP + Communities | Enforces minimum reefs per geographic community |
| `python -m scripts.run_miqp_size` | MIQP + Reef Sizing | Allocates reef area under total acreage budget |
| `python -m scripts.run_miqp_comm_size` | Hybrid | Combines equity and reef sizing constraints |

Each model saves outputs to `/runs/` including selected sites and solver summaries.

---

## 🔬 Algorithmic Experiments

Heuristic and metaheuristic algorithms serve as dynamic approximations to the MIQP.

| Algorithm | Method | Purpose |
|------------|---------|----------|
| Greedy Forward | Adds sites incrementally based on marginal gain | Myopic/ ode takes time|
| 1-Swap Hill Climb | Local improvement of Greedy solution | Enhances optimality |
| Stingy Backward | Removes least valuable sites until K remain | Validates local optimum |
| MIQP (AMPL) | Exact quadratic optimization | Global benchmark |
| MIQP + Community/Sizing | Adds realism & policy constraints | Equity & feasibility |

---

## 📊 Outputs and Figures

All results are automatically saved in `/runs/` and `/figures/`.

**Example output (`miqp_size_summary.txt`):**
```
MIQP (with sizing)
Objective: 52556.03
Selected Sites: 22
Reef Sizes (acres): 5–50 range, total = 1000
Interpretation: Optimizer assigns larger reefs to highly connected hubs.
```

**Example figures:**
- `fig_strategy_comparison.png` — compares greedy, backward, MIQP models.  
- `fig_reef_sizes_miqp_size.png` — visualizes reef size distribution.  

---

## 🧪 Experimental Workflow

1. **ODE Simulation** → Generate `P₀` and `P₁` connectivity matrices.  
2. **Greedy + Local Search** → Identify promising site clusters.  
3. **MIQP Formulation** → Optimize mathematically with quadratic model.  
4. **Add Constraints** → Enforce fairness (community) and resource limits (sizing).  
5. **Validate via ODE** → Compare surrogate vs. biological outcomes.

---

## 📈 Key Results & Insights

- **Heuristic Search** quickly converges to near-optimal solutions.  
- **MIQP** matches greedy + swap results, confirming surrogate validity.  
- **Community Constraints** distribute reefs more evenly across regions.  
- **Sizing Model** allocates reef area efficiently (larger reefs at network hubs).  
- Across all formulations, a **9-site backbone** appears consistently — a robust core for restoration investment.

---

## 🧭 Reproducibility

To reproduce all experiments and figures:

```bash
python -m scripts.prepare_miqp_data
python -m scripts.run_miqp
python -m scripts.run_miqp_comm
python -m scripts.run_miqp_size
python -m scripts.make_figures
```

Outputs are versioned in `/runs/` and `/figures/` for full traceability.

---

## 📚 References

- Gurobi Optimization LLC. *Gurobi Optimizer Reference Manual.*  
---

## 👩‍🔬 Authors and Acknowledgements

**Research Developer:** Marouf Paul  
**Advisor:** Dr. Rex Kincaid, 
**Collaborators:** Dr. Leah Shaw, Dr. Rom Lipcius,  
**Affiliation:** William & Mary – Virginia Institute of Marine Science (VIMS)

This project was conducted as part of a research assistantship focusing on **computational optimization for marine ecosystem restoration**.  

For questions or collaboration inquiries:  
📧 mmarouf@wm.edu | 🧠 [rexkincaid.wm.edu](https://www.wm.edu/as/mathematics/faculty-directory/kincaid_r.php)

---
