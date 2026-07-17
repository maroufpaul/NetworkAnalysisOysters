# 🦪 Network Analysis for Oyster Reef Restoration

This repository contains the full experimental and modeling framework for the **Oyster Reef Site Selection Problem (ORSSP)** — combining biological ODE simulation, ODE-driven heuristics, and mixed-integer quadratic programming (MIQP). Developed under the supervision of **Prof. Rex Kincaid** (William & Mary), with biological collaborators **Prof. Leah Shaw** (W&M) and **Prof. Rom Lipcius** (VIMS), the project integrates ecological modeling with network optimization to inform data-driven coastal restoration planning.

The repository is **config-driven** — every parameter lives in [`config.py`](config.py) — and **self-proving** — every result is checked against [`references.json`](references.json) on each run.

> A full technical report is available in [`docs/Oyster_Project_Report.pdf`](docs/Oyster_Project_Report.pdf) (LaTeX source in `docs/Oyster_Project_Report.tex`). It includes the full methods, all 16 post-search and exact designs, the network-centrality analysis, the equilibrium-convergence justification for the `t = 1000` horizon, and the self-recruitment sensitivity check.

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

   Under **constant** external supply the surrogate freezes every larval source at a single reference density `A*`. Under **site-specific** supply that assumption fails, and `scripts/run_iterated.py` replaces `A*` with per-source densities refreshed from a small number of ODE evaluations.

### Top-level drivers

Every parameter is read from `config.py`, and every objective / backbone is validated against `references.json`.

| Script | Role |
| --- | --- |
| `run_everything.py` | **Main driver** — all heuristics (constant + realistic `P₀`) and all 8 MIQP variants on both matrices, parallelized, and **self-validates** all 22 results against `references.json`. Supports `--update-refs`. |
| `run_extra_experiments.py` | Robustness experiments **E1–E5** (surrogate fidelity, null baseline, budget nesting, A\* / `P₁` sensitivity, self-recruitment sensitivity). |
| `scripts/run_iterated.py` | Site-specific `P₀`: the A\* sensitivity sweep and the iterated surrogate. |
| `scripts/size_sweep.py` | Reef-size budget sweep under uniform bounds. |
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
├── ampl/                              # AMPL models (.mod, hand-written) + data (.dat, GENERATED)
│   ├── oyster_quad.mod                # Base MIQP                       (objective: score)
│   ├── oyster_comm.mod                # + community minimums            (objective: Larvae)
│   ├── oyster_size.mod                # + reef sizing                   (objective: Larvae)
│   ├── oyster_comm_size.mod           # + sizing + community            (objective: TotalLarvae)
│   ├── oyster_quad_matrix1.dat        # GENERATED per matrix by prepare_data
│   ├── oyster_quad_matrix2.dat        #   (one file each — see note below)
│   ├── oyster_comm.dat                # GENERATED (matrix-independent)
│   ├── oyster_size.dat                # GENERATED (matrix-independent)
│   └── oyster_iter.dat                # GENERATED per pass by run_iterated
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
├── figures/                           # Generated PNGs
│
├── runs/                              # Generated CSV / TXT / JSON outputs
│   ├── oyster_index_mapping_matrix{1,2}.csv   # AMPL index ↔ site label table
│   ├── oyster_data_preview_matrix{1,2}.csv    # W preview in site labels
│   ├── run_everything.json            # every heuristic + MIQP + backbone, with pass/fail
│   ├── miqp_results.csv               # written by scripts/run_miqp.py
│   ├── astar_sweep.csv                # written by run_iterated --exp astar
│   ├── iterated_table.csv             # written by run_iterated --exp iterated
│   ├── size_sweep_matrix{1,2}.csv     # written by scripts/size_sweep.py
│   ├── network_ranking_matrix{1,2}_selfloops_{on,off}.txt
│   └── extra_experiments.json         # written by run_extra_experiments.py
│
├── scripts/                           # Entry-point CLIs
│   ├── prepare_data.py                # Excel + config → ALL AMPL .dat (partition-checked)
│   ├── run_miqp.py                    # ALL FOUR MIQP variants (--model base|comm|size|comm+size)
│   ├── run_iterated.py                # A* sweep + iterated surrogate (site-specific P₀)
│   ├── size_sweep.py                  # Reef-size budget sweep
│   ├── network_core_analysis.py       # Centrality on the candidate / backbone sites
│   ├── settling_time.py               # Equilibrium-convergence check (t = 1000 horizon)
│   ├── check_ampl_env.py              # Verify AMPL + Gurobi on PATH
│   └── make_figures.py                # ⚠ stale — see Known Issues
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
    │   ├── evaluator.py               # evaluate_subset(): THE ODE scoring path for every script
    │   ├── miqp.py                    # solve(): THE AMPL/Gurobi call for every script
    │   ├── greedy.py                  # Forward selection
    │   ├── local_search.py            # 1-for-1 swap improver
    │   └── backward.py                # Reverse elimination
    ├── utils/                         # IO helpers
    └── viz/                           # Figure helpers
```

Only `config.py`, `data/`, `ampl/*.mod` and `references.json` are hand-maintained. Everything under `runs/` and every `ampl/*.dat` is generated — delete them and re-run.

**Note on the per-matrix `.dat` files.** The surrogate data used to be a single `ampl/oyster_quad.dat` that `prepare_data` overwrote for each matrix, so only M2 survived `--matrix both`; a standalone MIQP run then read that file while using `--matrix` only to pick the (identical) label mapping, and silently solved the wrong matrix. Each matrix now has its own file, so `--matrix both` is safe and is the default.

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
- **`src/opt/miqp.py`** translates AMPL indices back to site labels before returning, so every CSV you read is in biological labels.
- **The drivers** (`run_everything.py`, `run_extra_experiments.py`, `scripts/run_iterated.py`) operate in site labels throughout.

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

The ODE heuristics and the convergence check work with just Python — no AMPL needed. **Run every command from the repository root** so `import config` resolves.

---

## 🚀 Running the Models

Everything is parameterized by `config.py`. Regenerate the AMPL data first, then run any model. Outputs go to `runs/`, figures to `figures/`.

### 1. Generate the AMPL data (from `config.py` + the spreadsheets)

```bash
python -m scripts.prepare_data                    # both matrices (default)
python -m scripts.prepare_data --matrix 1         # just M1
```

Writes `ampl/oyster_quad_matrix{1,2}.dat`, `ampl/oyster_comm.dat`, `ampl/oyster_size.dat`, and the index-mapping CSVs. **Re-run this after changing anything that affects the surrogate weights** (`A_STAR`, `P1SCALING`, `CONST_P0`, `K`, `SIZE`, `COMMUNITY_MINS`).

### 2. Everything, self-validated

```bash
python run_everything.py
```

All heuristics (constant + realistic `P₀`) and all 8 MIQP variants, both matrices, parallelized, checked against `references.json`. ~35 min. Ends with `PASSED 22  FAILED 0  ALL_PASS=True`.

### 3. MIQP variants standalone

```bash
python -m scripts.run_miqp                          # all 4 models, both matrices
python -m scripts.run_miqp --matrix 1 --model base
python -m scripts.run_miqp --model comm+size
python -m scripts.run_miqp --selfcheck              # baseline only, verify vs the report
```

| `--model` | Model |
| --- | --- |
| `base` | Base MIQP (cardinality only) |
| `comm` | + community minimums |
| `size` | + variable reef sizing |
| `comm+size` | + community + sizing |
| `all` | all four (default) |

Each variant solves to proven global optimality (nonconvex bilinear objective, `nonconvex=2`, MIP gap `1e-9`) in well under a second. Greedy / stingy take ~4–5 minutes; greedy + local swap ~15–30 minutes — the ODE is the bottleneck, which is why `run_everything.py` parallelizes the per-candidate evaluations across cores.

### 4. Site-specific external supply (A\* sweep + iterated surrogate)

```bash
python -m scripts.run_iterated                                  # both experiments
python -m scripts.run_iterated --exp astar                      # A* sweep only
python -m scripts.run_iterated --exp iterated --matrix 1
python -m scripts.run_iterated --exp iterated --fallback network
```

### 5. Reef-size budget sweep, centrality, convergence

```bash
python -m scripts.size_sweep                                    # both matrices
python -m scripts.size_sweep --matrix 1 --budgets 300 500 750 1000
python -m scripts.network_core_analysis --matrix 1
python -m scripts.network_core_analysis --matrix 2 --self-loops off
python -m scripts.settling_time
```

### 6. Robustness experiments

```bash
python run_extra_experiments.py                                 # E1–E5
```

---

## 🎛 Flags

| Script | Flag | Meaning |
| --- | --- | --- |
| `prepare_data.py` | `--matrix 1\|2\|both` | which matrix's data to write (default `both`) |
| `run_everything.py` | `--skip-miqp` | heuristics + backbone only |
| | `--skip-heur` | MIQP only (fast — skips the ~30 min of ODE work) |
| | `--workers N` | cap CPU usage (default: all cores) |
| | `--update-refs` | rewrite `references.json` with this run's numbers |
| `run_miqp.py` | `--matrix 1\|2\|both` | which connectivity matrix |
| | `--model base\|comm\|size\|comm+size\|all` | which formulation |
| | `--selfcheck` | verify the baseline objective against the report and exit |
| `run_iterated.py` | `--exp astar\|iterated\|all` | which experiment |
| | `--matrix 1\|2\|both` | which connectivity matrix |
| | `--fallback sticky\|isolated\|network\|all` | how to value **unselected** candidates between passes |
| `size_sweep.py` | `--matrix 1\|2\|both` | which connectivity matrix |
| | `--budgets 300 500 …` | total reef-area budgets to sweep |
| `network_core_analysis.py` | `--matrix 1\|2\|both` | which connectivity matrix |
| | `--self-loops on\|off` | keep or drop self-recruitment in the centrality graph |

> **`--update-refs` is the dangerous one.** It overwrites the golden values that everything else is checked against. Use it only when you *intend* to change a published number.

> **`--fallback` is the consequential one.** `sticky` keeps a candidate's last value, so one never selected keeps `A⁰` forever and the iteration can return its first selection and halt (worst case 81.3%). `isolated` reverts it to its solitary equilibrium (98.3%). `network` re-integrates it with the inflow it would receive from the current selected set (99.8%, and the only rule that returns a single design per matrix regardless of `A⁰`). Default in `config.ITER`.

---

## 🧩 Config Knobs

| Change | Edit |
| --- | --- |
| candidate sites, `K`, `TMAX`, `A_STAR`, `P1SCALING`, `MU`, `IC` | `config.py` top block |
| which MIQP models exist | `config.MIQP_MODELS` |
| reef-size bounds / total budget | `config.SIZE` |
| budget sweep values | `config.SIZE_SWEEP` |
| community minimums | `config.COMMUNITY_MINS` |
| iterated surrogate starts + fallback rule | `config.ITER` |
| A\* sweep values | `config.A_STAR_SWEEP` |
| solver options | `config.GUROBI_OPTIONS` |

---

## 🧪 Algorithmic Summary

| Algorithm | Method | Purpose |
| --- | --- | --- |
| **Greedy forward** | Add the site with the largest marginal ODE gain | Fast first pass; myopic |
| **1-swap hill climb** | Swap one in / one out until no improving swap | Refines greedy by 1–2% |
| **Stingy backward** | Drop the site whose removal hurts F least, repeat | Independent local-optimum check |
| **MIQP (AMPL/Gurobi)** | Quadratic surrogate, exact | Global optimum on the surrogate |
| **MIQP + community / sizing** | Adds equity and acreage realism | Real-world planning model |
| **Iterated surrogate** | Per-source densities refreshed from one ODE per pass | Restores surrogate accuracy under site-specific `P₀` |

---

## 📊 Key Results

> The MIQP surrogate retains self-recruitment (the connectivity diagonal), matching the JARS ODE. Zeroing it is reported only as a sensitivity check; it leaves the base/size selections unchanged and the cross-matrix and 7-site backbones unaffected.

### Validation
`run_everything.py` re-derives every heuristic and MIQP objective and validates it against `references.json` (prints `OK / MISMATCH / no-ref` per design and a `PASSED / FAILED` summary). The ten heuristic scores (6 constant-`P₀` + 4 realistic-`P₀`) match the original logs **exactly to six decimal places**.

Last full run — everything reproduced:

| Result | Status |
| --- | --- |
| 22/22 in `run_everything.py` | ✅ `ALL_PASS=True` (1977 s) |
| A\* sweep, all 18 cells + 4/6 distinct designs + 14.1/47.1 pt spread | ✅ exact |
| Iterated surrogate, all 30 cells | ✅ exact |
| Reef-size sweep, network centrality | ✅ exact |

### Backbone sites
Across all 16 post-search and exact designs (2 matrices × 4 MIQP variants + 4 swap + 4 stingy heuristics across both `P₀` regimes):

- **Global backbone:** `{10, 31, 37, 40, 41, 49, 53}` — **7 sites**

These reefs sit in dense subnetworks (high eigenvector centrality, high PageRank) or act as bridges (high betweenness), and every optimization method we ran selects them. **Site 37 is the interesting one:** only moderate standalone centrality (out-strength rank 31/49), selected by every design, and held at the 5-acre floor at every budget in the size sweep. Its value is in receiving and relaying, and acreage cannot buy that — the clearest evidence that a site-by-site centrality screen misses what the quadratic objective captures.

---

## ✅ Reproducing Everything

```bash
# 0. Setup + (for MIQP) verify the solver. Optionally edit config.py.
pip install -r requirements.txt
python -m scripts.check_ampl_env

# 1. Generate AMPL data from config.py + Excel (self-recruitment / diagonal kept,
#    communities partition-checked and translated to AMPL indices).
python -m scripts.prepare_data

# 2. One-pass: all heuristics (constant + realistic P0) + all 8 MIQP, both
#    matrices, parallelized, self-validated against references.json.
#    Writes runs/run_everything.json and prints PASSED / FAILED.
python run_everything.py

# 3. Site-specific P0: A* sensitivity sweep + iterated surrogate.
#    Writes runs/astar_sweep.csv and runs/iterated_table.csv.
python -m scripts.run_iterated

# 4. Reef-size budget sweep under uniform bounds.
#    Writes runs/size_sweep_matrix{1,2}.csv.
python -m scripts.size_sweep

# 5. Robustness experiments E1-E5 (surrogate fidelity, null baseline, budget
#    nesting, A* / P1 sensitivity, self-recruitment sensitivity).
python run_extra_experiments.py

# 6. Equilibrium-convergence check (justifies the t = 1000 horizon)
python -m scripts.settling_time

# 7. Network centrality
python -m scripts.network_core_analysis
```

**Validation flow at a glance:** `run_everything.py` checks every heuristic and MIQP objective + the backbone against `references.json`; `run_extra_experiments.py` checks the robustness claims (E1–E5); `scripts/settling_time.py` checks equilibrium convergence. If you're reviewing or extending the work, run those three first.

---

## 📚 References

- Lipcius, R. N., Zhang, Y., Zhou, J., Shaw, L. B., & Shi, J. (2021). Modeling oyster reef restoration: larval supply and reef geometry jointly determine population resilience and performance. Frontiers in Marine Science, 8:677640. https://doi.org/10.3389/fmars.2021.677640 (Source of the JARS model and parameter table.)
- Lipcius, R. N., Shen, J., Shaw, L. B., & Shi, J. (2024). Oyster larval transport / hydrodynamic modeling at Tangier/Pocomoke Sound, Virginia. U.S. Army Corps of Engineers, Chesapeake Watershed CESU W912HZ-23-02-0015, Final Report. (Source of the connectivity matrices and the metapopulation JARS extension.)
- Jordan-Cooley, W. C., Lipcius, R. N., Shaw, L. B., Shen, J., & Shi, J. (2011). Bistability in a differential equation model of oyster reef height and sediment accumulation. Journal of Theoretical Biology, 289, 1–11.
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