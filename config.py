# config.py
#
# Single source of truth for the Oyster Reef Site Selection pipeline.
# Every script and model parameter is read from here. To run a new experiment,
# change a value in this file and re-run `python -m scripts.prepare_data` (which
# regenerates the AMPL .dat files) followed by `python run_everything.py`.
#
# Nothing in this file imports from `src/`, so it can be imported anywhere
# without circular-import problems.

from pathlib import Path
import numpy as np

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
AMPL_DIR = ROOT / "ampl"
RUNS_DIR = ROOT / "runs"
FIG_DIR  = ROOT / "figures"

# --------------------------------------------------------------------------- #
# Candidate set
# --------------------------------------------------------------------------- #
# Sites dropped on biological advice (unsuitable habitat). Removing 66-72 from
# the 56 simulated sites, together with the labels never present in the raw
# matrices, leaves the 49 candidates below.
UNWANTED = [66, 67, 68, 69, 70, 71, 72]

# The canonical candidate list (biological site labels), in AMPL-index order:
# position i in this array == AMPL index i in the .dat files.
CANDIDATE_SITES = np.array([
    1,  3,  4,  5,  6,  7,  9, 10, 11, 12,
    15, 16, 17, 18, 19, 20, 21, 24, 26, 27,
    28, 29, 30, 31, 32, 33, 35, 36, 37, 38,
    39, 40, 41, 42, 44, 47, 48, 49, 50, 51,
    52, 53, 54, 55, 56, 57, 58, 59, 60
], dtype=int)

# --------------------------------------------------------------------------- #
# ODE / surrogate parameters
# --------------------------------------------------------------------------- #
TMAX      = 1000      # integration horizon, in model YEARS
P1SCALING = 0.5       # scaling applied to raw connectivity -> internal supply P1
CONST_P0  = 170.0     # constant external supply (constant-forcing runs)
A_STAR    = 0.05675   # median single-reef equilibrium adult density
ALPHA     = 1.72      # gamete/fertilisation density exponent
MU        = 0.4       # adult mortality rate used by the evaluator
K         = 25        # number of reefs to select

# Initial conditions for the JARS integration (generous start -> high-density
# restored equilibrium).
IC = {"J": 0.0, "A": 0.2, "R": 0.3, "S": 0.0}

# --------------------------------------------------------------------------- #
# Connectivity matrices
# --------------------------------------------------------------------------- #
MATRICES = {
    "M1": DATA_DIR / "nk_All_060102final_56sites_Model.xlsx",  # dry year 2002
    "M2": DATA_DIR / "nk_All_060103final_56sites_Model.xlsx",  # high-flow 2003
}
DEFAULT_MATRIX = "M1"

# Map a CLI "--matrix 1|2" to the dict key above.
def matrix_key(matrix_id: str) -> str:
    return {"1": "M1", "2": "M2", "M1": "M1", "M2": "M2"}[str(matrix_id)]

# --------------------------------------------------------------------------- #
# Realistic external supply
# --------------------------------------------------------------------------- #
# VOSARA low/moderate/high classes -> {100, 200, 400}. Indexed by (label - 1).
P0_REALISTIC = np.array([
    400, 400, 400, 400, 400, 400, 200, 200, 200, 100, 100, 100,
    100, 100, 100, 100, 100, 400, 400, 200, 200, 400, 400, 400,
    200, 100, 200, 100, 100, 200, 200, 100, 100, 100, 400, 200,
    200, 400, 400, 100, 100, 400, 400, 100, 400, 400, 200, 400,
    200, 100, 100, 100, 200, 100, 400, 400, 400, 100, 100, 400,
    100, 400, 400, 400, 100, 400, 400, 400, 400, 400, 200, 400,
], dtype=int)

# --------------------------------------------------------------------------- #
# Communities (equity constraints)
# --------------------------------------------------------------------------- #
# The community membership is read from this spreadsheet (one row per community:
# comm_number | size | label_1 label_2 ...). prepare_data verifies it partitions
# all 49 candidates before writing ampl/oyster_comm.dat.
COMMUNITIES_XLSX = DATA_DIR / "communitiesJune2002.xlsx"

# Minimum number of selected sites per community (the equity requirement).
COMMUNITY_MINS = {1: 2, 2: 5, 3: 3, 4: 2, 5: 3}

# --------------------------------------------------------------------------- #
# Variable reef sizing
# --------------------------------------------------------------------------- #
# Upper bound is tiered by AMPL index: the first U_SPLIT indices get U_FIRST,
# the rest get U_REST. (This reproduces the original size testbed exactly.)
SIZE = {
    "L":           5.0,     # min reef area (acres) when a site is built
    "U_FIRST":     50.0,    # max area for AMPL indices [0, U_SPLIT)
    "U_REST":      40.0,    # max area for AMPL indices [U_SPLIT, n)
    "U_SPLIT":     26,
    "TotReefSize": 1000.0,  # total area budget (acres)
    "Sbar":        20.0,    # size-normalisation constant
}

# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #
GUROBI_OPTIONS = "nonconvex=2 mipgap=1e-9"

# --------------------------------------------------------------------------- #
# Reef-size budget-sweep experiment  (scripts/size_sweep.py)
# --------------------------------------------------------------------------- #
# UNIFORM bounds on purpose: the canonical SIZE dict above tiers the ceiling by
# index (40 vs 50), which is an assumption, not biology. For the experiment we
# give every site the SAME L and U so any area concentration is driven by
# connectivity, not by which ceiling a site was handed.
#
# Sweep the budget T across the BINDING band  L*K < T < U*K.
# With K=25, L=5, U=50 that band is 125 < T < 1250:
#   T >= 1250 -> every site maxed (no discrimination)
#   T <= 125  -> every site at the floor (no discrimination)
#   in between -> budget binds, optimizer must rank sites -> the interesting case.
SIZE_SWEEP = {
    "L": 5.0,            # uniform lower bound (acres) when a site is built
    "U": 50.0,           # uniform upper bound (acres)
    "Sbar": 20.0,
    "budgets": [300, 500, 750, 1000],   # slack = T/(K*U) = 0.24 / 0.40 / 0.60 / 0.80
}