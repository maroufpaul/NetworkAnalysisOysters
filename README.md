# 🦪 Network Analysis for Oyster Reef Restoration

This repository contains the full experimental and modeling framework for the **Oyster Reef Site Selection Problem (ORSSP)** — combining biological ODE simulation, ODE-driven heuristics, and mixed-integer quadratic programming (MIQP). Developed under the supervision of **Prof. Rex Kincaid** (William & Mary), with biological collaborators **Prof. Leah Shaw** (W&M) and **Prof. Rom Lipcius** (VIMS), the project integrates ecological modeling with network optimization to inform data-driven coastal restoration planning.

The repository is **config-driven** — every parameter lives in [`config.py`](config.py) — and **self-proving** — every result is checked against [`references.json`](references.json) on each run.

> A full technical report is available in [`docs/Oyster_Project_Report.pdf`](docs/Oyster_Project_Report.pdf) (LaTeX source in `docs/Oyster_Project_Report.tex`). It includes the full methods, all 18 final designs, the network-centrality analysis, the equilibrium-convergence justification for the `t = 1000` horizon, and a self-recruitment sensitivity check.

---

## 🌍 Background and Motivation

The Chesapeake Bay historically supported more than **10 billion oysters**; today the population sits below **1%** of that baseline due to overharvesting, habitat loss, and disease. Restoration is expensive, so the question is not whether to restore but **where**:

> Among 49 candidate reef sites distributed across five ecological management zones in the bay, which subset of **K = 25** sites should be restored to maximize long-run adult oyster biomass?

Because oyster larvae disperse via tidal currents, each reef's value depends not only on its own productivity but on the larvae it receives from, and sends to, the other selected reefs. The problem is **combinatorial and non-additive**: clusters of well-connected reefs interact super-linearly, so an isolated high-productivity reef can be worse than three weaker but well-connected ones.

---

## 🧠 Project Overview

The pipeline has three modeling layers:

1. **JARS metapopulation ODE** (`src/model/jars_ode.py`)
   Four coupled life stages — juveniles, adults, reef shell, sediment — with larval transport between sites driven by an empirical connectivity matrix `P₁` and an external supply vector `P₀`. The internal larval sum includes each reef's own contribution (self-recruitment). The nonlinear larval-production term `|A|^1.72` makes retentive clusters disproportionately valuable. Model time is in **years** (the JARS rate parameters are annual; Lipcius et al. 2021, Tables 1–2). *(The single-reef JARS model and its parameters are from Lipcius et al. 2021, Front. Mar. Sci. 8:677640, which itself revises Jordan-Cooley et al. 2011; the metapopulation extension with larval connectivity follows the USACE CESU 2024 report. Original MATLAB by Prof. Leah Shaw; Python port by Marouf Paul.)*

2. **ODE-driven heuristics** (`src/opt/`)
   - **Greedy forward** — fast but myopic
   - **1-for-1 local swap** — refines greedy
   - **Stingy backward elimination** — independent path to the same local optimum (validation)

3. **MIQP surrogates via AMPL + Gurobi** (`ampl/`)
   Quadratic surrogate of the JARS dynamics — with self-recruitment (the connectivity diagonal) retained to match the ODE — and four constraint variants:
   - Base (cardinality only)
   - Community minimums (regional equity)
   - Variable reef sizing (continuous acreage)
   - Community + sizing (combined real-world constraints)

### Top-level drivers

Every parameter is read from `config.py`, and every objective / backbone is validated against `references.json`. The project is reproduced through **two** root-level drivers plus a convergence script:

| Script | Role |
| --- | --- |
| `run_everything.py` | Main driver — re-runs all heuristics (constant + realistic `P₀`) and all 8 MIQP variants on both matrices, parallelized, and **self-validates** every objective and the backbone against `references.json`. Supports `--update-refs` to record fresh numbers. |
| `run_extra_experiments.py` | Robustness experiments **E1–E5** (surrogate fidelity, null baseline, budget nesting, A\* / `P₁` sensitivity, self-recruitment sensitivity). |
| `scripts/settling_time.py` | Equilibrium-convergence check: how fast adult biomass settles and why `t = 1000` is on the steady-state plateau. |

---

## 🧩 Repository Structure

```
NetworkAnalysisOysters/
│
├── config.py                          # Single source of truth for every parameter
├── references.json                    # Golden expected values (validated each run)
├── run_everything.py                  # Main driver: all heuristics + all MIQP, parallel, self-validating
├── run_extra_experiments.py           # Robustness experiments E1–E5
├── requirements.txt
├── README.md                          # This file
│
├── ampl/                              # AMPL models + AUTO-GENERATED data (via scripts/prepare_data.py)
│   ├── oyster_quad.mod / .dat         # Base MIQP (select K sites)
│   ├── oyster_comm.mod / .dat         # MIQP + community minimums (rmin1..5, from config)
│   ├── oyster_size.mod / .dat         # MIQP + reef sizing (budget)
│   └── oyster_comm_size.mod           # MIQP + sizing + community (reads comm.dat + size.dat)
│
├── data/                              # INPUTS
│   ├── nk_All_060102final_56sites_Model.xlsx   # Matrix 1 (M1, dry year 2002)
│   ├── nk_All_060103final_56sites_Model.xlsx   # Matrix 2 (M2, high-flow year 2003)
│   └── communitiesJune2002.xlsx               # Community membership (partition of the 49 sites)
│
├── docs/
│   ├── Oyster_Project_Report.tex      # Full technical report (LaTeX source)
│   └── Oyster_Project_Report.pdf      # Compiled report
│
├── figures/                           # Generated PNGs (strategy comparison, centrality, …)
│
├── runs/                              # Generated CSV / TXT / JSON outputs (gitignored)
│   ├── oyster_index_mapping_matrix{1,2}.csv   # AMPL index ↔ site label table (INPUT — tracked)
│   ├── oyster_data_preview_matrix{1,2}.csv    # W preview in site labels
│   ├── *_sites_matrix{1,2}.csv        # Selected sites (+ reef sizes where applicable)
│   ├── *_summary_matrix{1,2}.txt      # Objective values and solver logs
│   ├── network_ranking_matrix{1,2}_selfloops_{on,off}.txt   # Per-site centrality
│   ├── run_everything.json            # written by run_everything.py
│   └── extra_experiments.json         # written by run_extra_experiments.py
│
├── scripts/                           # Entry-point + auxiliary CLIs
│   ├── prepare_data.py                # Excel + config → ALL AMPL .dat (partition-checked, self-recruitment kept)
│   ├── run_greedy.py                  # Greedy heuristic
│   ├── run_greedy_then_local.py       # Greedy + 1-swap local search
│   ├── run_backward.py                # Stingy backward elimination
│   ├── run_miqp.py                    # Base MIQP
│   ├── run_miqp_comm.py               # MIQP + communities
│   ├── run_miqp_size.py               # MIQP + reef sizing
│   ├── run_miqp_comm_size.py          # MIQP + community + sizing
│   ├── network_core_analysis.py       # Centrality on the candidate / backbone sites
│   ├── settling_time.py               # Equilibrium-convergence check (t = 1000 horizon)
│   ├── make_figures.py                # Paper figures
│   ├── compare_greedy_backward.py     # Cross-check greedy vs stingy
│   ├── plot_greedy_run.py             # Plot a greedy run
│   └── check_ampl_env.py              # Verify AMPL + Gurobi on PATH
│
├── presentation/                      # Slide deck + slide builders
│   ├── Oyster_presentation_updated.pptx
│   ├── generate_presentation.py
│   └── generate_presentation_figures.py
│
└── src/                               # Core algorithms + biological model
    ├── __init__.py                    # puts the repo root on sys.path so `import config` works
    ├── model/jars_ode.py              # JARS ODE; re-exports CANDIDATE_SITES / P0 setter from config
    ├── opt/
    │   ├── evaluator.py               # ODE scoring of any site set
    │   ├── greedy.py                  # Forward selection
    │   ├── local_search.py            # 1-for-1 swap improver
    │   └── backward.py                # Reverse elimination
    ├── utils/                         # IO helpers
    └── viz/                           # Figure helpers
```

---

## 🔢 Indexing: Site Labels vs AMPL Indices (Read This First)

There are **two coexisting indexing schemes** in this project, and confusing them is the easiest way to get wrong results. (This is exactly the bug that once left the community file broken — see below.)

### Biological site labels
The original integer site IDs from the Excel matrices: `1, 3, 4, 5, …, 60` (sites `66`–`72` are dropped per biological advice; several others in `1`–`60` are absent from the raw data, leaving **49 candidate sites**).

> **Every CSV under `runs/` and every results table in this README and in the report uses these labels.**

The canonical candidate list lives in `config.py` as `CANDIDATE_SITES` (re-exported from `src/model/jars_ode.py` for backward compatibility):

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

The full table for each matrix is auto-saved to `runs/oyster_index_mapping_matrix{1,2}.csv` whenever `scripts/prepare_data.py` is run.

### Where this matters
- **AMPL `.dat` files** use indices. For example, in the generated `oyster_comm.dat`:
  ```ampl
  set C1 :=
    0  13  33  36  44  45  46 ;   # these are INDICES, not site labels
  ```
  Looking up those indices in the mapping CSV shows that `C1` actually contains site labels `{1, 18, 42, 48, 56, 57, 58}`.
- `oyster_comm.dat` is **generated by `prepare_data.py`** from `data/communitiesJune2002.xlsx`, which does the label→index translation and **refuses to run unless the five communities partition all 49 candidates** (sizes 7 / 19 / 10 / 5 / 8 = 49). It can no longer silently go stale or untranslated.
- **Python runner scripts** (`scripts/run_miqp*.py`) translate AMPL indices back to site labels before writing CSVs, so the output you read is always in biological labels.
- **The drivers** (`run_everything.py`, `run_extra_experiments.py`) operate in site labels throughout.

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

The ODE heuristics (greedy, swap, stingy) and the convergence check work with just Python — no AMPL needed. **Run every command from the repository root** so `import config` resolves.

---

## 🚀 Running the Models

Everything is parameterized by `config.py`. Regenerate the AMPL data first, then run any model. All MIQP/centrality scripts accept `--matrix 1` or `--matrix 2`. Outputs are written to `runs/` and figures to `figures/`.

### Generate the AMPL data (from `config.py` + the spreadsheets)

```bash
python -m scripts.prepare_data --matrix both      # or --matrix 1 / --matrix 2
```

This writes `ampl/oyster_quad.dat` (matrix-dependent), `ampl/oyster_comm.dat`, and `ampl/oyster_size.dat`, plus the index-mapping CSVs. `oyster_quad.dat` is overwritten per matrix, so for a standalone single-matrix MIQP run, prepare that matrix first (the `run_everything.py` driver does this automatically).

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

Each MIQP variant solves to proven global optimality (nonconvex bilinear objective, `nonconvex=2`, MIP gap `1e-9`) in well under a second; greedy / stingy take ~4–5 minutes; greedy + local swap takes ~15–30 minutes (the ODE is the bottleneck). `run_everything.py` parallelizes the per-candidate ODE evaluations across CPU cores and runs everything in one pass.

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

> The MIQP surrogate retains self-recruitment (the connectivity diagonal), matching the JARS ODE. Zeroing it is reported only as a sensitivity check; it leaves the base/size selections unchanged and the cross-matrix and 7-site backbones unaffected.

### Validation
`run_everything.py` re-derives every heuristic and MIQP objective and validates it against `references.json` (prints `OK / MISMATCH / no-ref` per design and a `PASSED / FAILED` summary). The ten heuristic scores (6 constant-`P₀` + 4 realistic-`P₀`) match the original logs **exactly to six decimal places**.



### Backbone sites
Across all 18 model runs (2 matrices × 4 MIQP variants + 6 constant-P₀ heuristics + 4 realistic-P₀ heuristics):

- **Global backbone (every model, every matrix, both P₀ regimes):**
  `{10, 31, 37, 40, 41, 49, 53}` — **7 sites** (unchanged after the community correction)


The 7-site backbone is the practical takeaway: these reefs sit in dense subnetworks (high eigenvector centrality, high PageRank) or act as bridges (high betweenness), and every optimization method we ran selects them. **Any restoration design that omits them is structurally weaker.**

---

## ✅ Reproducing Everything

```bash
# 0. Setup + (for MIQP) verify the solver. Optionally edit config.py.
pip install -r requirements.txt
python -m scripts.check_ampl_env

# 1. Generate AMPL data from config.py + Excel (self-recruitment / diagonal kept,
#    communities partition-checked and translated to AMPL indices).
python -m scripts.prepare_data --matrix both

# 2. One-pass: all heuristics (constant + realistic P0) + all 8 MIQP, both
#    matrices, parallelized, self-validated against references.json.
#    Writes runs/run_everything.json and prints PASSED / FAILED.
python run_everything.py

# 3. Robustness experiments E1-E5 (surrogate fidelity, null baseline, budget
#    nesting, A* / P1 sensitivity, self-recruitment sensitivity).
#    Writes runs/extra_experiments.json.
python run_extra_experiments.py

# 4. Equilibrium-convergence check (justifies the t = 1000 horizon)
python -m scripts.settling_time

# 5. Network centrality + figures
python -m scripts.network_core_analysis --matrix 1
python -m scripts.network_core_analysis --matrix 2
python -m scripts.make_figures
```

For granular runs, the individual `scripts/run_*` entry points above produce the same per-model CSVs in `runs/`. `run_everything.py` accepts `--skip-miqp` / `--skip-heur` to run only one layer, `--workers N` to cap CPU usage, and `--update-refs` to record a fresh run's numbers into `references.json`.

**Validation flow at a glance:** `run_everything.py` checks every heuristic and MIQP objective + the backbone against `references.json`; `run_extra_experiments.py` checks the robustness claims (E1–E5); `scripts/settling_time.py` checks equilibrium convergence. If you're reviewing or extending the work, run those three first.

---

## 🧭 Limitations and Next Steps

- **Equilibrium-only objective.** `F(S)` is the sum of adult biomass at `t = 1000` (model years). Transients and resilience to shocks are not considered.
- **Fixed JARS parameters.** No robustness analysis over parameter uncertainty.
- **Surrogate uses a fixed A\*.** A state-dependent surrogate would tighten alignment with the ODE, at the cost of solvability. (A\* sets only the objective scale, not the selection — see E4.)
- **Exogenous communities.** The five regions come from geography, not from the connectivity graph. Data-driven community detection on `P₁` is a natural extension.

Suggested next experiments: robust/stochastic MIQP, multi-objective Pareto frontiers, phased multi-period restoration, and coupling with non-larval ecosystem services.

See the discussion section of the technical report for the full treatment.

---

## 📚 References

- Lipcius, R. N., Zhang, Y., Zhou, J., Shaw, L. B., & Shi, J. (2021). Modeling oyster reef restoration: larval supply and reef geometry jointly determine population resilience and performance. Frontiers in Marine Science, 8:677640. https://doi.org/10.3389/fmars.2021.677640 (Source of the JARS model and parameter table.)
- Lipcius, R. N., Shen, J., Shaw, L. B., & Shi, J. (2024). Oyster larval transport / hydrodynamic modeling at Tangier/Pocomoke Sound, Virginia. U.S. Army Corps of Engineers, Chesapeake Watershed CESU W912HZ-23-02-0015, Final Report. (Source of the connectivity matrices and the metapopulation JARS extension.)
- Gurobi Optimization, LLC. *Gurobi Optimizer Reference Manual.*
- AMPL: Fourer, R., Gay, D. M., Kernighan, B. W. *AMPL: A Modeling Language for Mathematical Programming.*

---

## 👩‍🔬 Authors and Acknowledgements

**Research Developer:** Marouf Paul
**Advisor:** Prof. Rex Kincaid (Mathematics, William & Mary)
**Collaborators:** Prof. Leah Shaw (W&M), Prof. Rom Lipcius (Virginia Institute of Marine Science)

This project was conducted as part of an M.S. in Computational Operations Research at William & Mary, focused on computational optimization for marine ecosystem restoration.

**Contact:**
📧 mmarouf@wm.edu / maroufpaul2@gmail.com (post-graduation)
🔗 [Prof. Kincaid's faculty page](https://www.wm.edu/as/mathematics/faculty-directory/kincaid_r.php)

---