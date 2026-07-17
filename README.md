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

2. **ODE-driven heuristics** 
   - **Greedy forward** — fast but myopic
   - **1-for-1 local swap** — refines greedy
   - **Stingy backward elimination** — independent path to the same local optimum (validation)

3. **MIQP surrogates via AMPL + Gurobi** (`ampl/`)
   Quadratic surrogate of the JARS dynamics — with self-recruitment (the connectivity diagonal) retained to match the ODE — and four constraint variants:
   - Base (cardinality only)
   - Community minimums (regional equity)
   - Variable reef sizing (continuous acreage)
   - Community + sizing (combined real-world constraints)


## ⚡ Quickstart
 
```bash
pip install -r requirements.txt          # needs a working AMPL + Gurobi license
python -m scripts.prepare_data           # writes AMPL data for both matrices
python run_everything.py                 # ~35 min. Ends with ALL_PASS=True
```
 
Then the rest:
 
```bash
python -m scripts.run_iterated           # Tables 6 and 7
python -m scripts.size_sweep             # Table 5
python -m scripts.network_core_analysis  # Table 8
python run_extra_experiments.py          # Sec 5.7 (E1–E5)
python -m scripts.settling_time          # Sec 4 timescales
```

---

## 📁 Directory structure
 
```
.
├── config.py                     ← every parameter. Start here.
├── references.json               ← golden values; run_everything.py checks against these
├── requirements.txt
│
├── run_everything.py             ← heuristics + all 8 MIQPs + backbones  [VALIDATES]
├── run_extra_experiments.py      ← E1–E5 (fidelity, null, K-sweep, params, self-recruit)
│
├── scripts/
│   ├── prepare_data.py           ← xlsx + config  ->  AMPL .dat files
│   ├── run_miqp.py               ← all 4 MIQP variants standalone (--model, --matrix)
│   ├── run_iterated.py           ← site-specific Pe: A* sweep + iterated surrogate
│   ├── size_sweep.py             ← reef-size budget sweep, uniform bounds
│   ├── network_core_analysis.py  ← centrality of all 49 sites
│   ├── settling_time.py          ← equilibrium-convergence check for t=1000
│   ├── check_ampl_env.py         ← verify AMPL/Gurobi is wired up
│   └── make_figures.py           ← ⚠ stale, see Known issues
│
├── src/
│   ├── model/jars_ode.py         ← JARS metapopulation ODE (J, A, R, S)
│   ├── opt/
│   │   ├── evaluator.py          ← evaluate_subset(): THE ODE scoring path
│   │   ├── miqp.py               ← solve(): THE AMPL/Gurobi call
│   │   ├── greedy.py             ← greedy additive
│   │   ├── backward.py           ← stingy deletion
│   │   └── local_search.py       ← swap improvement
│   ├── utils/io_utils.py
│   └── viz/plots.py
│
├── ampl/
│   ├── oyster_quad.mod           ← baseline surrogate        (objective: score)
│   ├── oyster_comm.mod           ← + community coverage      (objective: Larvae)
│   ├── oyster_size.mod           ← + variable reef size      (objective: Larvae)
│   ├── oyster_comm_size.mod      ← + both                    (objective: TotalLarvae)
│   ├── oyster_quad_matrix1.dat   ← generated per matrix by prepare_data
│   ├── oyster_quad_matrix2.dat
│   ├── oyster_comm.dat           ← generated (matrix-independent)
│   ├── oyster_size.dat           ← generated (matrix-independent)
│   └── oyster_iter.dat           ← generated per pass by run_iterated
│
├── data/                         ← raw connectivity + community xlsx (INPUTS)
├── runs/                         ← all generated results
├── figures/
├── docs/                         ← paper source + PDF
└── presentation/
```


Only `config.py`, `data/`, `ampl/*.mod` and `references.json` are hand-maintained. Everything in `runs/` and every `ampl/*.dat` is generated.

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

## 🔬 Paper → script → command → output
 
| Paper | Experiment | Command | Output |
|---|---|---|---|
| **Table 2** | heuristics, constant Pe | `python run_everything.py` | `runs/run_everything.json` |
| **Table 3** | heuristics, realistic Pe | same run | same |
| **Table 4** | 8 integer programs | same run | same |
| **Table 4** | *(standalone)* | `python -m scripts.run_miqp` | `runs/miqp_results.csv` |
| **Table 5** | reef-size budget sweep | `python -m scripts.size_sweep` | `runs/size_sweep_matrix{1,2}.csv` |
| **Table 6** | A\* sensitivity | `python -m scripts.run_iterated --exp astar` | `runs/astar_sweep.csv` |
| **Table 7** | iterated surrogate | `python -m scripts.run_iterated --exp iterated` | `runs/iterated_table.csv` |
| **Table 8** | network centrality | `python -m scripts.network_core_analysis` | `runs/network_ranking_matrix*.txt` |
| **Table 9** | realistic greedy seeds | `python run_everything.py` | `runs/run_everything.json` |
| **backbone ×3** | within / cross / global | same run | same |
| **§5.7** | surrogate fidelity (E1) | `python run_extra_experiments.py` | `runs/extra_experiments.json` |
| **§5.7** | null baseline (E2) | same run | same |
| **§5.7** | budget nesting, K sweep (E3) | same run | same |
| **§5.7** | A\*/P1 invariance (E4) | same run | same |
| **§5.7** | self-recruitment (E5) | same run | same |
| **§4** | settling time / t=1000 | `python -m scripts.settling_time` | stdout |
 
---
 
## ✅ Verifying the numbers
 
`run_everything.py` is the check. It ends with:
 
```
PASSED 22  FAILED 0  ALL_PASS=True
```
 
If you change anything and it still says `ALL_PASS=True`, the paper still holds. Use `--update-refs` **only** when you mean to change a published number.
 
`run_miqp.py` checks its 8 objectives against paper Table 4 the same way:
 
```bash
python -m scripts.run_miqp --selfcheck   # baseline only, both matrices, fast
python -m scripts.run_miqp               # all 8, OK/MISMATCH per row
```
 
Last full run, everything reproduced:
 
| Result | Status |
|---|---|
| 22/22 in `run_everything.py` | ✅ `ALL_PASS=True` (1977 s) |
| Table 6, all 18 cells + 4/6 distinct designs + 14.1/47.1 spread | ✅ exact |
| Table 7, all 30 cells | ✅ exact |
| Table 5 size sweep, Table 8 centrality | ✅ exact |
| E4 invariance (Jaccard 1.0 across the A\*×P1 grid) | ✅ |
| E5 (7.39% / 3.89% drop, selections unchanged) | ✅ |
| `\|F(1000) − F(2000)\|` = 6.1e-06 / 5.6e-06 | ✅ under the stated 1e-5 |
 
---
 
## 🎛 Flags
 
```bash
python -m scripts.prepare_data --matrix 1
 
python -m scripts.run_miqp --matrix 1 --model base
python -m scripts.run_miqp --model comm+size
python -m scripts.run_miqp --selfcheck
 
python -m scripts.run_iterated --exp astar
python -m scripts.run_iterated --exp iterated --matrix 1 --fallback network
 
python -m scripts.size_sweep --matrix 1 --budgets 300 500 750 1000
python -m scripts.network_core_analysis --matrix 2 --self-loops off
 
python run_everything.py --skip-heur      # MIQP + backbone only (fast)
python run_everything.py --update-refs    # rewrite references.json
```
 
---
 
## 🧩 Config knobs
 
| Change | Edit |
|---|---|
| candidate sites, `K`, `TMAX`, `A_STAR`, `P1SCALING`, `MU`, `IC` | `config.py` top block |
| which MIQP models exist | `config.MIQP_MODELS` |
| reef-size bounds / total budget | `config.SIZE` |
| budget sweep values | `config.SIZE_SWEEP` |
| community minimums | `config.COMMUNITY_MINS` |
| iterated surrogate starts + fallback rule | `config.ITER` |
| A\* sweep values | `config.A_STAR_SWEEP` |
| solver options | `config.GUROBI_OPTIONS` |
 
**Re-run `python -m scripts.prepare_data` after changing anything that affects the surrogate weights** (`A_STAR`, `P1SCALING`, `CONST_P0`, `K`, `SIZE`, `COMMUNITY_MINS`).
 
### The iterated surrogate's fallback rule
 
`config.ITER["fallback"]` decides what an **unselected** candidate is worth at the next pass. This is the consequential choice in that experiment:
 
| Rule | What it does | Worst over 5 starts |
|---|---|---|
| `sticky` | keep its last value — a candidate never selected keeps `A⁰` forever, so the iteration can return its first selection and halt | **81.3%** |
| `isolated` | revert to `Ā_l`, its equilibrium integrated alone | 98.3% |
| `network` | one-step lookahead: re-integrate it alone with external supply raised by the inflow it would get from the current selected set | **99.8%** ← default |
 
Only `network` returns a single design per matrix regardless of `A⁰`.
 


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