# 🦪 Network Analysis for Oyster Reef Restoration

This repository contains the full experimental and modeling framework for the **Oyster Reef Site Selection Problem (ORSSP)** — combining biological ODE simulation, ODE-driven heuristics, and mixed-integer quadratic programming (MIQP). Developed under the supervision of **Prof. Rex Kincaid** (William & Mary), with biological collaborators **Prof. Leah Shaw** (W&M) and **Prof. Rom Lipcius** (VIMS), the project integrates ecological modeling with network optimization to inform data-driven coastal restoration planning.

> A full technical handover report is available in [`docs/Oyster_Project_Report.pdf`](docs/Oyster_Project_Report.pdf) (LaTeX source in `docs/Oyster_Project_Report.tex`). The report includes the full methods, all 18 final designs, network centrality analysis, and an errata appendix listing five corrections to earlier drafts.

---

## 🌍 Background and Motivation

The Chesapeake Bay historically supported more than **10 billion oysters**; today the population sits below **1%** of that baseline due to overharvesting, habitat loss, and disease. Restoration is expensive, so the question is not whether to restore but **where**:

> Among 49 candidate reef sites distributed across five ecological management zones in the bay, which subset of **K = 25** sites should be restored to maximize long-run adult oyster biomass?

Because oyster larvae disperse via tidal currents, each reef's value depends not only on its own productivity but on the larvae it receives from, and sends to, the other selected reefs. The problem is **combinatorial and non-additive**: clusters of well-connected reefs interact super-linearly, so an isolated high-productivity reef can be worse than three weaker but well-connected ones.

---

## 🧠 Project Overview

The pipeline has three modeling layers:

1. **JARS metapopulation ODE** (`src/model/jars_ode.py`)
   Four coupled life stages — juveniles, adults, reef shell, sediment — with larval transport between sites driven by an empirical connectivity matrix `P₁` and an external supply vector `P₀`. The nonlinear larval-production term `|A|^1.72` makes retentive clusters disproportionately valuable. (Original MATLAB by Prof. Leah Shaw; Python port by Marouf Paul.)

2. **ODE-driven heuristics** (`src/opt/`)
   - **Greedy forward** — fast but myopic
   - **1-for-1 local swap** — refines greedy
   - **Stingy backward elimination** — independent path to the same local optimum (validation)

3. **MIQP surrogates via AMPL + Gurobi** (`ampl/`)
   Quadratic surrogate of the JARS dynamics with four constraint variants:
   - Base (cardinality only)
   - Community minimums (regional equity)
   - Variable reef sizing (continuous acreage)
   - Community + sizing (combined real-world constraints)

---

## 🧩 Repository Structure

```
NetworkAnalysisOysters/
│
├── ampl/                              # AMPL models and auto-generated data
│   ├── oyster_quad.mod / .dat         # Base MIQP (select K sites)
│   ├── oyster_comm.mod / .dat         # MIQP + community minimums
│   ├── oyster_size.mod / .dat         # MIQP + reef sizing (budget)
│   └── oyster_comm_size.mod / .dat    # MIQP + sizing + community
│
├── data/                              # Connectivity matrices
│   ├── nk_All_060102final_56sites_Model.xlsx   # Matrix 1
│   └── nk_All_060103final_56sites_Model.xlsx   # Matrix 2
│
├── figures/                           # Generated PNGs (strategy comparison, centrality, …)
│
├── runs/                              # All CSV / TXT outputs
│   ├── *_sites_matrix{1,2}.csv        # Selected sites (and reef sizes where applicable)
│   ├── *_summary_matrix{1,2}.txt      # Objective values and solver logs
│   ├── network_metrics_*.csv          # Per-site centrality metrics
│   ├── oyster_index_mapping_matrix{1,2}.csv   # AMPL index ↔ site label table
│   └── validation_summary.csv         # Produced by validate.py
│
├── scripts/                           # Entry-point CLIs
│   ├── prepare_miqp_data.py           # Excel → AMPL .dat
│   ├── run_greedy.py                  # Greedy heuristic
│   ├── run_greedy_then_local.py       # Greedy + 1-swap local search
│   ├── run_backward.py                # Stingy backward elimination
│   ├── run_miqp.py                    # Base MIQP
│   ├── run_miqp_comm.py               # MIQP + communities
│   ├── run_miqp_size.py               # MIQP + reef sizing
│   ├── run_miqp_comm_size.py          # MIQP + community + sizing
│   ├── network_core_analysis.py       # Centrality on the backbone sites
│   └── make_figures.py                # Plots
│
├── src/                               # Core algorithms and biological model
│   ├── model/jars_ode.py              # JARS ODE, CANDIDATE_SITES, P0 setter
│   ├── opt/
│   │   ├── evaluator.py               # ODE scoring of any site set
│   │   ├── greedy.py                  # Forward selection
│   │   ├── local_search.py            # 1-for-1 swap improver
│   │   └── backward.py                # Reverse elimination
│   ├── utils/io_utils.py              # CSV/TXT writers
│   └── viz/plots.py                   # Figure helpers
│
├── docs/
│   ├── Oyster_Project_Handover_Report.tex   # Full technical report (LaTeX source)
│   └── Oyster_Project_Handover_Report.pdf   # Compiled report
│
├── validate.py                        # Independent re-validation harness
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

---

## 🔢 Indexing: Site Labels vs AMPL Indices (Read This First)

There are **two coexisting indexing schemes** in this project, and confusing them is the easiest way to get wrong results.

### Biological site labels
The original integer site IDs from the Excel matrices: `1, 3, 4, 5, …, 60` (sites `66`–`72` are dropped per biological advice; several others in `1`–`60` are absent from the raw data, leaving **49 candidate sites**).

> **Every CSV under `runs/` and every results table in this README and in the report uses these labels.**

The canonical candidate list lives in `src/model/jars_ode.py` as `CANDIDATE_SITES`:

```python
CANDIDATE_SITES = [
     1,  3,  4,  5,  6,  7,  9, 10, 11, 12,
    15, 16, 17, 18, 19, 20, 21, 24, 26, 27,
    28, 29, 30, 31, 32, 33, 35, 36, 37, 38,
    39, 40, 41, 42, 44, 47, 48, 49, 50, 51,
    52, 53, 54, 55, 56, 57, 58, 59, 60
]
```

### AMPL indices
Inside the AMPL data files (`ampl/oyster_*.dat`), the index set `N` is `0, 1, …, 48` — i.e., positions in the `CANDIDATE_SITES` array, **not** the biological labels.

| AMPL index | Site label |
| ---: | ---: |
| 0  | 1  |
| 1  | 3  |
| 2  | 4  |
| …  | …  |
| 25 | 33 |
| 33 | 42 |
| 48 | 60 |

The full table for each matrix is auto-saved to `runs/oyster_index_mapping_matrix{1,2}.csv` whenever `prepare_miqp_data.py` is run.

### Where this matters
- **AMPL `.dat` files** use indices. For example, in `oyster_comm.dat`:
  ```ampl
  set C1 := 33 36 44 ;   # these are INDICES, not site labels
  ```
  Looking up indices `33, 36, 44` in the mapping CSV shows that `C1` actually contains site labels `{42, 48, 56}`.
- **Python runner scripts** (`scripts/run_miqp*.py`) translate AMPL indices back to site labels before writing CSVs, so the output you read is always in biological labels.
- **The validation harness** (`validate.py`) operates entirely in site labels.

**Rule of thumb:** if you're looking at a `.dat` file, the integers are indices. Everywhere else, they're site labels.

---

## ⚙️ Installation and Setup

```bash
git clone https://github.com/maroufpaul/NetworkAnalysisOysters.git
cd NetworkAnalysisOysters

conda create -n oysters python=3.10
conda activate oysters
pip install -r requirements.txt
```

AMPL + Gurobi need to be on PATH for the MIQP scripts. Verify with:

```bash
python -m scripts.check_ampl_env
```

The ODE heuristics (greedy, swap, stingy) work with just Python — no AMPL needed.

---

## 🚀 Running the Models

All scripts accept `--matrix 1` or `--matrix 2` to choose the connectivity matrix. Outputs are written to `runs/` and figures to `figures/`.

### ODE-driven heuristics (constant P₀)

```bash
python -m scripts.run_greedy
python -m scripts.run_greedy_then_local
python -m scripts.run_backward
```

### MIQP variants (constant P₀)

| Command | Model |
| --- | --- |
| `python -m scripts.run_miqp           --matrix 1` | Base MIQP |
| `python -m scripts.run_miqp_comm      --matrix 1` | + community minimums |
| `python -m scripts.run_miqp_size      --matrix 1` | + variable reef sizing |
| `python -m scripts.run_miqp_comm_size --matrix 1` | + community + sizing |

Each MIQP variant solves to proven optimality in well under a second; greedy / stingy take ~4–5 minutes; greedy + local swap takes ~15–30 minutes (the ODE is the bottleneck).

### Network centrality and figures

```bash
python -m scripts.network_core_analysis --matrix 1
python -m scripts.make_figures
```

---

## 🧪 Algorithmic Summary

| Algorithm | Method | Purpose |
| --- | --- | --- |
| **Greedy forward** | Add the site with the largest marginal ODE gain | Fast first pass; myopic |
| **1-swap hill climb** | Swap one in / one out until no improving swap | Refines greedy by 1–2% |
| **Stingy backward** | Drop the site whose removal hurts F least, repeat | Independent local-optimum check |
| **MIQP (AMPL/Gurobi)** | Quadratic surrogate, exact | Global optimum on the surrogate |
| **MIQP + community / sizing** | Adds equity and acreage realism | Real-world planning model |

---

## 📊 Key Results

### Validation
All ODE scores reported in the handover were independently re-derived with `validate.py`. The ten heuristic scores match the original logs **exactly to six decimal places**.

### Heuristic scores under constant P₀
| Matrix | Greedy | Greedy + Swap | Stingy |
| ---: | ---: | ---: | ---: |
| Matrix 1 | 1.846640 | **1.880054** | **1.880054** |
| Matrix 2 | 1.692871 | **1.733033** | **1.733033** |

On both matrices, greedy + swap and stingy converge to the same 25-site set with the same score — strong evidence of a stable local optimum.

### MIQP objectives under constant P₀
| Model | Matrix 1 | Matrix 2 |
| --- | ---: | ---: |
| Base MIQP | 13,692.07 | 15,806.74 |
| + Comm | 12,782.91 | 14,587.09 |
| + Size | 58,564.19 | 63,607.36 |
| + Comm + Size | 56,442.63 | 64,447.23 |

The sizing models always exhaust the 1,000-acre budget and allocate 40–50 acres to network hubs, 5–15 acres to weaker neighbors.

### Backbone sites
Across all 18 model runs (2 matrices × 4 MIQP variants + 3 heuristics + realistic P₀):

- **Global backbone (every model, every matrix, both P₀ regimes):**
  `{10, 31, 37, 40, 41, 49, 53}` — **7 sites**
- **Cross-matrix constant-P₀ backbone:**
  `{10, 15, 31, 32, 37, 40, 41, 49, 51, 52, 53}` — **11 sites**
- **Per-matrix constant-P₀ backbone:**
  Matrix 1: 14 sites; Matrix 2: 14 sites

The 7-site backbone is the practical takeaway: these reefs sit in dense subnetworks (high eigenvector centrality, high PageRank) or act as bridges (high betweenness), and every optimization method we ran selects them. **Any restoration design that omits them is structurally weaker.**

---

## ✅ Reproducing Everything

```bash
# 1. Generate AMPL data from Excel
python -m scripts.prepare_miqp_data --matrix 1
python -m scripts.prepare_miqp_data --matrix 2

# 2. ODE heuristics
python -m scripts.run_greedy
python -m scripts.run_greedy_then_local
python -m scripts.run_backward

# 3. All MIQP variants for both matrices
for m in 1 2; do
  python -m scripts.run_miqp           --matrix $m
  python -m scripts.run_miqp_comm      --matrix $m
  python -m scripts.run_miqp_size      --matrix $m
  python -m scripts.run_miqp_comm_size --matrix $m
done

# 4. Network centrality + figures
python -m scripts.network_core_analysis --matrix 1
python -m scripts.network_core_analysis --matrix 2
python -m scripts.make_figures

# 5. Independent re-validation (re-runs ODE on every reported design,
#    recomputes all backbone intersections from scratch)
python validate.py
```

`validate.py` writes `runs/validation_summary.csv` and `runs/validation_network_core_matrix{1,2}.csv`. Every number in the handover report came from that script — if you're reviewing or extending the work, run it first.

---

## 🧭 Limitations and Next Steps

- **Equilibrium-only objective.** `F(S)` is the sum of adult biomass at `t = 1000`. Transients and resilience to shocks are not considered.
- **Fixed JARS parameters.** No robustness analysis over parameter uncertainty.
- **Surrogate uses a fixed A\*.** A state-dependent surrogate would tighten alignment with the ODE, at the cost of solvability.
- **Exogenous communities.** The five regions come from geography, not from the connectivity graph. Data-driven community detection on `P₁` is a natural extension.

Suggested next experiments: robust/stochastic MIQP, multi-objective Pareto frontiers, phased multi-period restoration, and coupling with non-larval ecosystem services.

See §7 of the handover report for the full discussion.

---

## 📚 References

- Gurobi Optimization, LLC. *Gurobi Optimizer Reference Manual.*
- AMPL: Fourer, R., Gay, D.M., Kernighan, B.W. *AMPL: A Modeling Language for Mathematical Programming.*
- JARS dynamics: based on metapopulation ODE work by Prof. Leah Shaw (W&M).

---

## 👩‍🔬 Authors and Acknowledgements

**Research Developer:** Marouf Paul
**Advisor:** Prof. Rex Kincaid (Mathematics, William & Mary)
**Collaborators:** Prof. Leah Shaw (W&M), Prof. Rom Lipcius (Virginia Institute of Marine Science)

This project was conducted as part of an M.S. research in Computational Operations Research at William & Mary, focused on computational optimization for marine ecosystem restoration.

**Contact:**
📧 mmarouf@wm.edu / maroufpaul2@gmail.com (post-graduation)
🔗 [Prof. Kincaid's faculty page](https://www.wm.edu/as/mathematics/faculty-directory/kincaid_r.php)

---
