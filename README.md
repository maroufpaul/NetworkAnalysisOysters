# 🦪 Network Analysis for Oyster Reef Restoration

This repository contains the full experimental and modeling framework for the **Oyster Reef Site Selection Problem (ORSSP)** — combining biological ODE simulation, ODE-driven heuristics, and mixed-integer quadratic programming (MIQP). Developed under the supervision of **Prof. Rex Kincaid** (William & Mary), with biological collaborators **Prof. Leah Shaw** (W&M) and **Prof. Rom Lipcius** (VIMS), the project integrates ecological modeling with network optimization to inform data-driven coastal restoration planning.

Everything is driven by [`config.py`](config.py). Five scripts do the work — one job each — and each talks to AMPL directly, with no wrapper layers.

> A full technical report is in [`docs/Oyster_Project_Report.pdf`](docs/Oyster_Project_Report.pdf) (LaTeX source alongside it). It has the full methods, all 16 post-search and exact designs, the network-centrality analysis, the equilibrium-convergence justification for the `t = 1000` horizon, and the self-recruitment sensitivity check.

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

2. **ODE-driven heuristics** (`scripts/run_heuristics.py`)
   - **Greedy forward** — fast but myopic
   - **1-for-1 local swap** — refines greedy
   - **Stingy backward elimination** — independent path to the same local optimum (validation)

   Every score is a full ODE solve, so this is the slow, honest layer. It's parallelized across CPU cores.

3. **MIQP surrogates via AMPL + Gurobi** (`ampl/`)
   Quadratic surrogate of the JARS dynamics — with self-recruitment (the connectivity diagonal) retained to match the ODE — and four constraint variants:
   - Base (cardinality only)
   - Community minimums (regional equity)
   - Variable reef sizing (continuous acreage)
   - Community + sizing (combined real-world constraints)

   Under **constant** external supply the surrogate freezes every larval source at a single reference density `A*`, and the selection is provably invariant to it. Under **site-specific** supply that assumption fails, and `scripts/run_iterated.py` replaces `A*` with per-source densities refreshed from a small number of ODE evaluations.

### The five scripts

| Script | Role |
| --- | --- | 
| `scripts/prepare_data.py` | Excel + `config.py` → AMPL data. Verifies the communities partition the 49 sites. | 
| `scripts/run_miqp.py` | MIQP, **constant `P₀`**. base / comm / size / comm+size × both matrices. 
| `scripts/run_heuristics.py` | ODE search: greedy / swap / stingy × both matrices × constant + realistic `P₀`. Parallel. Takes 10-15 min to run|
| `scripts/run_iterated.py` | MIQP, **realistic `P₀`**: the frozen-`A*` failure, then the ODE-feedback fix. |
| `scripts/run_extra.py` | Backbone, surrogate fidelity, K sweep, self-recruitment. Reads what the others wrote. |

Plus three utilities: `size_sweep.py` (reef area vs budget), `network_core_analysis.py` (centrality), `settling_time.py` (why `t = 1000`), and `check_ampl_env.py` (is the solver wired up).

---

## 🧩 Repository Structure

```
NetworkAnalysisOysters/
│
├── config.py                          # Single source of truth for every parameter
├── requirements.txt
├── README.md                          # This file
│
├── scripts/                           # The five drivers + three utilities
│   ├── prepare_data.py                # Excel + config -> AMPL .dat (partition-checked)
│   ├── run_miqp.py                    # MIQP, constant P0 (--model base|comm|size|comm+size)
│   ├── run_heuristics.py              # greedy/swap/stingy, parallel (--method, --p0)
│   ├── run_iterated.py                # MIQP, realistic P0: the problem + the ODE fix
│   ├── run_extra.py                   # backbone, fidelity, K sweep, self-recruitment
│   ├── size_sweep.py                  # reef area vs budget, uniform bounds
│   ├── network_core_analysis.py       # centrality of all 49 sites
│   ├── settling_time.py               # equilibrium-convergence check (t = 1000)
│   └── check_ampl_env.py              # is AMPL + Gurobi actually on PATH?
│
├── src/
│   ├── model/jars_ode.py              # the JARS ODE itself
│   └── opt/evaluator.py               # evaluate_subset(): THE ODE scoring path
│
├── ampl/
│   ├── oyster_quad.mod                # base MIQP                    (objective: score)
│   ├── oyster_comm.mod                # + community minimums         (objective: Larvae)
│   ├── oyster_size.mod                # + reef sizing                (objective: Larvae)
│   ├── oyster_comm_size.mod           # + both                       (objective: TotalLarvae)
│   │
│   ├── oyster_quad_M1_constant.dat    # GENERATED - one per matrix x P0 mode
│   ├── oyster_quad_M1_realistic.dat   #   (see the note below)
│   ├── oyster_quad_M2_constant.dat
│   ├── oyster_quad_M2_realistic.dat
│   ├── oyster_comm.dat                # GENERATED (same for both matrices)
│   ├── oyster_size.dat                # GENERATED (same for both matrices)
│   └── oyster_iter.dat                # GENERATED per solve by run_iterated
│
├── data/                              # INPUTS - the only things you can't regenerate
│   ├── nk_All_060102final_56sites_Model.xlsx   # M1, dry year (2002)
│   ├── nk_All_060103final_56sites_Model.xlsx   # M2, high-flow year (2003)
│   └── communitiesJune2002.xlsx               # the 5 regions
│
├── runs/                              # GENERATED results
│   ├── oyster_index_mapping_matrix{1,2}.csv   # AMPL index <-> site label (tracked)
│   ├── miqp_<model>.txt / .csv                # run_miqp
│   ├── heuristics_<method>.txt / .csv         # run_heuristics
│   ├── astar_sweep.csv                        # run_iterated --exp astar
│   ├── iterated.txt / .csv                    # run_iterated --exp iterated
│   ├── extra_<exp>.txt                        # run_extra
│   ├── size_sweep.csv, size_sweep_summary.txt # size_sweep
│   └── network_ranking_matrix{1,2}_*.txt      # network_core_analysis
│
├── figures/                           # size_sweep_matrix{1,2}.png + presentation figures
├── docs/                              # report source + PDF
└── presentation/                      # slide deck + builders
```

Hand-maintained: `config.py`, `data/`, `ampl/*.mod`, the `.py` files. **Everything in `runs/` and every `ampl/*.dat` is generated** — delete and re-run.

** The `.dat` files are split by matrix and `P₀` mode.** 

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
- **Every script converts indices back to site labels** before writing anything, so every CSV you read is in biological labels.

**Rule of thumb:** if you're looking at a `.dat` file, the integers are indices. Everywhere else, they're site labels.

---


## ⚙️ Installation and Setup

### Get the repository

Choose one of the following options:

#### Option 1: Clone the repository

> **Contribution note:** Please do not commit or push changes directly to the `main` branch. Create a separate branch for your work.

#### Option 2: Fork the repository

If you plan to modify the code and do not have write access to this repository, fork it on GitHub and clone your fork:


#### Option 3: Download the ZIP file

If you only want to run the project and do not need Git version control:

1. Select **Code** on the GitHub repository page.
2. Select **Download ZIP**.
3. Extract the downloaded file.
4. Open a terminal inside the extracted `NetworkAnalysisOysters` directory.


```bash
cd NetworkAnalysisOysters
```

### Python `venv` (THIS IS RECOMMENDED NOT NECESSARY)

`venv` is included with Python and does not require Conda.

#### Windows PowerShell

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 🔑 AMPL + Gurobi License

The MIQP scripts (`run_miqp`, `run_iterated`, `size_sweep`, `run_extra`) need
**AMPL** and **Gurobi**. Both are commercial but **free for academics**. (The
`run_heuristics` layer needs neither.)

- **Gurobi** — get an academic license key from the [Gurobi User Portal](https://portal.gurobi.com/),
  then run `grbgetkey <your-key>`. This writes `gurobi.lic` to your home folder,
  where the solver finds it automatically. Set `GRB_LICENSE_FILE` if you put it
  elsewhere.
- **AMPL** — `pip install amplpy` ships a community edition that covers a model
  this size. Academic AMPL licenses are often temporary, so check the expiry.

### Verify AMPL and Gurobi

The MIQP scripts require **AMPL** and a licensed **Gurobi** installation available on `PATH`. Verify the solver setup with:

```bash
python -m scripts.check_ampl_env
```

> Run every command from the repository root so that imports such as `import config` resolve correctly.

The MIQP scripts need **AMPL + a licensed Gurobi on PATH**. Verify with:

```bash
python -m scripts.check_ampl_env
```

The ODE heuristics, the settling-time check, and the centrality analysis are pure Python — no solver needed. **Run every command from the repository root** so `import config` resolves.

---

## 🚀 Running the Models

Everything is parameterized by `config.py`. Generate the AMPL data first, then run any layer. `run_extra --exp backbone` reads what the other scripts wrote into `runs/`, so **run it last**.

### The whole thing, in order

```bash
python -m scripts.prepare_data      # ~10 sec  writes the AMPL data
python -m scripts.run_miqp          # ~5 sec   MIQP, constant P0, 4 models x 2 matrices
python -m scripts.run_heuristics    # ~15 min  greedy/swap/stingy x 2 matrices x 2 P0 modes
python -m scripts.run_iterated      # ~10 min  MIQP, realistic P0 + the ODE fix
python -m scripts.size_sweep        # ~1 min   reef area vs budget
python -m scripts.run_extra         # ~2 min   backbone, fidelity, K sweep, self-recruitment
```

### 1. Generate the AMPL data (from `config.py` + the spreadsheets)

```bash
python -m scripts.prepare_data                     # both matrices, both P0 modes
python -m scripts.prepare_data --matrix 1
python -m scripts.prepare_data --p0 realistic
```

Writes `ampl/oyster_quad_M{1,2}_{constant,realistic}.dat`, `ampl/oyster_comm.dat`, `ampl/oyster_size.dat`, and the index-mapping CSVs. **Re-run this after changing anything that affects the surrogate weights** (`A_STAR`, `P1SCALING`, `CONST_P0`, `K`, `SIZE`, `COMMUNITY_MINS`).

### 2. MIQP, constant external supply

```bash
python -m scripts.run_miqp                         # all 4 models, both matrices
python -m scripts.run_miqp --model size
python -m scripts.run_miqp --model comm+size --matrix 2
```

Constant `P₀ = 170` means every reef gets the same external subsidy, so any difference between designs comes purely from the larval network. The `comm` reports break the selection down by region and flag which minimums are binding; the `size` reports give the per-site acreage. Each model solves to proven global optimality (nonconvex bilinear objective, `nonconvex=2`, MIP gap `1e-9`) in under a second.

### 3. ODE heuristics (constant + realistic `P₀`)

```bash
python -m scripts.run_heuristics                   # everything
python -m scripts.run_heuristics --method greedy
python -m scripts.run_heuristics --matrix 1 --p0 constant
python -m scripts.run_heuristics --workers 4       # leave some cores free
```

`--method swap` runs greedy first — swap starts from greedy's answer.

### 4. MIQP, site-specific external supply (A\* sweep + iterated surrogate)

```bash
python -m scripts.run_iterated                     # both experiments
python -m scripts.run_iterated --exp astar         # just the frozen-A* problem
python -m scripts.run_iterated --exp iterated --fallback network
python -m scripts.run_iterated --trace --show-sites
```

See [The iterated surrogate](#-the-iterated-surrogate) for what this is actually doing.

### 5. Reef-size budget sweep, centrality, convergence

```bash
python -m scripts.size_sweep                       # both matrices, size model
python -m scripts.size_sweep --matrix 1 --budgets 300 500 750 1000
python -m scripts.size_sweep --model comm+size     # with the equity constraint
python -m scripts.network_core_analysis --matrix 1
python -m scripts.network_core_analysis --matrix 2 --self-loops off
python -m scripts.settling_time
```

### 6. Robustness experiments

```bash
python -m scripts.run_extra                        # backbone + fidelity + ksweep + selfrecruit
python -m scripts.run_extra --exp backbone
python -m scripts.run_extra --exp fidelity
```

---

## 🎛 Flags

| Script | Flag | Meaning |
| --- | --- | --- |
| `prepare_data.py` | `--matrix 1\|2\|both` | which matrix (default `both`) |
| | `--p0 constant\|realistic\|both` | which supply mode (default `both`) |
| `run_miqp.py` | `--model base\|comm\|size\|comm+size\|all` | which formulation |
| | `--matrix 1\|2\|both` | which connectivity matrix |
| `run_heuristics.py` | `--method greedy\|swap\|stingy\|all` | which search |
| | `--matrix 1\|2\|both` | which connectivity matrix |
| | `--p0 constant\|realistic\|both` | which supply mode |
| | `--workers N` | CPU workers (default: all cores) |
| `run_iterated.py` | `--exp astar\|iterated\|all` | the problem, the fix, or both |
| | `--matrix 1\|2\|both` | which connectivity matrix |
| | `--fallback sticky\|isolated\|network\|all` | how to value **unpicked** sites between rounds |
| | `--trace` | print each round's ODE score |
| | `--show-sites` | print the final site sets |
| `run_extra.py` | `--exp backbone\|fidelity\|ksweep\|selfrecruit\|all` | which experiment |
| | `--matrix 1\|2\|both` | which connectivity matrix |
| `size_sweep.py` | `--matrix 1\|2\|both` | which connectivity matrix |
| | `--model size\|comm+size` | with or without the equity constraint |
| | `--budgets 300 500 …` | override `config.SIZE_SWEEP['budgets']` |
| | `--no-plot` | skip the PNG |
| `network_core_analysis.py` | `--matrix 1\|2\|both` | which connectivity matrix |
| | `--self-loops on\|off` | keep or drop self-recruitment in the graph |

---

## 🧩 Config Knobs

| Change | Edit |
| --- | --- |
| candidate sites, `K`, `TMAX`, `A_STAR`, `P1SCALING`, `MU`, `IC` | `config.py` top block |
| which MIQP models exist | `config.MIQP_MODELS` |
| reef-size bounds / total budget | `config.SIZE` |
| budget sweep values | `config.SIZE_SWEEP` |
| community minimums / names | `config.COMMUNITY_MINS`, `config.COMM_NAMES` |
| iterated surrogate starts + fallback | `config.ITER` |
| A\* sweep values | `config.A_STAR_SWEEP` |
| solver options | `config.GUROBI_OPTIONS` |

After editing anything that affects the surrogate weights, re-run `python -m scripts.prepare_data`.

---

## 🔄 The Iterated Surrogate

`run_iterated.py` is the one script whose *point* isn't obvious from its name, so:

**The problem.** The MIQP weights every larval link as `W[l,k] = P₁[l,k] × A_l^1.72` — how many larvae site `l` sends depends on how many adults sit on `l`. So it needs a source density for all 49 candidates before it can solve, and it uses one number, `A*`, for all of them.

Under constant `P₀` that's fine: every reef settles near the same density, and `A*` just rescales the objective (the selection doesn't move — `run_extra --exp ksweep` and the report's E4 confirm it). Under realistic `P₀` it's false — the `P₀=400` reefs settle near 0.058, the `P₀=200` ones near 0.046, and most `P₀=100` ones don't establish at all. `--exp astar` sweeps `A*` and shows the answer swinging between **51% and 99%** of the best heuristic with no findable pattern.

**The fix.** Give each source its own density. You don't know them up front, so: guess → solve → run the ODE once on the 25 you picked → update → re-solve. Stops in 2–3 rounds, when two solves return the same set.

**The catch, and what `--fallback` controls.** The ODE only tells you about the 25 you picked. The other 24 still need a density for the next round:

| `--fallback` | What an unpicked site is worth | Worst case |
| --- | --- | --- |
| `sticky` | leave it at the last guess | **81.3%** — a site never picked keeps its initial guess forever, so the loop hands back its first answer |
| `isolated` | what it'd reach alone in the bay | **98.3%** — a real number, wrong question, and it's 0 for the 17 reefs that can't establish alone |
| `network` | what it'd reach **if it joined your current 25** | **99.8%** — the actual question, and the only rule that returns the same design regardless of the starting guess |

`network` is the default (`config.ITER["fallback"]`). It costs one extra single-reef ODE per unpicked site per round — nothing next to the metapopulation solve you're already paying for. The five starting guesses (`config.ITER["starts"]`) are one isolated-density start plus four flat constants (0.02, 0.05675, 0.20, 0.50) chosen as stress tests; "Designs = 1" in the report's table means all five converge to the same set.

---

## 🧪 Algorithmic Summary

| Algorithm | Method | Purpose |
| --- | --- | --- |
| **Greedy forward** | Add the site with the largest marginal ODE gain | Fast first pass; myopic |
| **1-swap hill climb** | Swap one in / one out until nothing improves | Refines greedy by 1–2% |
| **Stingy backward** | Drop the site you'd miss least, repeat | Independent local-optimum check |
| **MIQP (AMPL/Gurobi)** | Quadratic surrogate, exact | Global optimum on the surrogate |
| **MIQP + community / sizing** | Adds equity and acreage realism | Real-world planning model |
| **Iterated surrogate** | Per-source densities refreshed from one ODE per round | Restores accuracy under realistic `P₀` |

---

## 📊 Key Results

> The MIQP surrogate retains self-recruitment (the connectivity diagonal), matching the JARS ODE. Zeroing it is reported only as a sensitivity check: it leaves the base/size selections unchanged and the cross-matrix and 7-site backbones unaffected.

### Numbers to check against

There is no automatic reference checking — compare by eye:

| Where | Expect |
| --- | --- |
| `runs/miqp_all.csv` | M1 base **14785.03**, M1 comm **13863.50**, M1 size **32620.49** (1000 ac) |
| | M2 base **16446.59**, M2 comm **15294.18**, M2 size **36695.67** (1000 ac) |
| `runs/heuristics_all.csv` | M1 greedy constant **1.846640**, M1 swap constant **1.880054** |
| | M1 swap realistic **1.861968**, M2 swap realistic **1.793457** |
| `runs/extra_backbone.txt` | picked by everything: **7 sites** `[10, 31, 37, 40, 41, 49, 53]` |
| | per matrix: **16** / **16**, both matrices: **11** |
| `runs/iterated.csv` | `network` worst **99.8%**, one design per matrix; `sticky` worst **81.3%** |
| `settling_time` | `\|F(1000) − F(2000)\|` ≈ 6e-06 |

### Backbone sites

Across all 16 post-search and exact designs (2 matrices × 4 MIQP variants + 4 swap + 4 stingy across both `P₀` regimes):

- **Global backbone:** `{10, 31, 37, 40, 41, 49, 53}` — **7 sites**

These sit in dense subnetworks (high eigenvector centrality, high PageRank) or act as bridges (high betweenness), and every method picks them. **Site 37 is the interesting one:** only moderate standalone centrality (out-strength rank 31/49), picked by every design, and held at the 5-acre floor at every budget in the size sweep. Its value is in receiving and relaying, and acreage can't buy that — the clearest evidence that a site-by-site centrality screen misses what the quadratic objective captures.

---

## ✅ Reproducing Everything

```bash
# 0. Setup + (for MIQP) verify the solver. Optionally edit config.py.
pip install -r requirements.txt
python -m scripts.check_ampl_env

# 1. Generate AMPL data from config.py + Excel (self-recruitment kept,
#    communities partition-checked and translated to AMPL indices).
python -m scripts.prepare_data

# 2. MIQP under constant external supply, all four models, both matrices.
#    Writes runs/miqp_all.txt / .csv.
python -m scripts.run_miqp

# 3. ODE heuristics: greedy/swap/stingy, both matrices, both P0 modes.
#    Writes runs/heuristics_all.txt / .csv.
python -m scripts.run_heuristics

# 4. Site-specific P0: A* sweep + iterated surrogate.
#    Writes runs/astar_sweep.csv and runs/iterated.txt / .csv.
python -m scripts.run_iterated

# 5. Reef-size budget sweep under uniform bounds.
#    Writes runs/size_sweep.csv, runs/size_sweep_summary.txt.
python -m scripts.size_sweep

# 6. Robustness: backbone, surrogate fidelity, K sweep, self-recruitment.
#    (Reads the CSVs from steps 2-3, so run it after them.)
python -m scripts.run_extra

# 7. Equilibrium-convergence check (justifies the t = 1000 horizon).
python -m scripts.settling_time

# 8. Network centrality.
python -m scripts.network_core_analysis
```

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