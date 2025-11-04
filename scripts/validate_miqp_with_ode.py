# scripts/validate_miqp_with_ode.py

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# we only need odesys from your model file
from src.model.jars_ode import odesys

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"

EXCEL_PATH = DATA_DIR / "nk_All_060102final_56sites_Model.xlsx"
MIQP_CSV = RUNS_DIR / "miqp_sites.csv"

# this is the same list we filtered out everywhere
UNWANTED = [66, 67, 68, 69, 70, 71, 72]


def load_connectivity(path: Path):
    """Load excel and return (connectivity_matrix, labels) with 66–72 removed."""
    raw = pd.read_excel(path, header=None).values
    labels = raw[0, 1:].astype(int)
    P = raw[1:, 1:].astype(float)

    mask = ~np.isin(labels, UNWANTED)
    labels = labels[mask]
    P = P[np.ix_(mask, mask)]
    return P, labels


def run_jars_subset(selected_sites, connectivity, labels,
                    tmax=1000, P1scaling=0.5, consP0=170.0):
    """
    Minimal evaluator: given a list of site labels,
    prune the connectivity to those sites and run the 4N ODE, return sum of A.
    """
    # map site labels to indices in "labels"
    idx = []
    for s in selected_sites:
        where = np.where(labels == s)[0]
        if len(where):
            idx.append(where[0])
    idx = np.array(idx, dtype=int)
    if len(idx) == 0:
        return 0.0

    # prune connectivity and scale
    P1 = P1scaling * connectivity[np.ix_(idx, idx)]
    key_subset = labels[idx]
    Npatch = len(key_subset)

    # external larvae (constant P0 like in your scripts)
    P0 = consP0 * np.ones(Npatch)

    # mortality
    mu = 0.4 * np.ones(Npatch)

    # initial conditions
    v0 = np.zeros(4 * Npatch)
    # J, A, R, S
    for i in range(Npatch):
        v0[i] = 0.0         # J
        v0[Npatch + i] = 0.2   # A
        v0[2 * Npatch + i] = 0.3  # R
        v0[3 * Npatch + i] = 0.0  # S

    sol = solve_ivp(lambda t, v: odesys(t, v, P0, P1, mu),
                    [0, tmax], v0, method="RK45", rtol=1e-6)

    v_final = sol.y[:, -1]
    A_final = v_final[Npatch:2 * Npatch]

    return float(np.sum(A_final))


def main():
    # 1) read MIQP sites
    miqp_df = pd.read_csv(MIQP_CSV)
    sites = miqp_df["site_id"].astype(int).tolist()
    print("[validate] MIQP sites:", sites)

    # 2) load connectivity
    connectivity, labels = load_connectivity(EXCEL_PATH)

    # 3) evaluate
    total_A = run_jars_subset(sites, connectivity, labels,
                              tmax=1000, P1scaling=0.5, consP0=170.0)

    print(f"[validate] ∑A(t=1000) for MIQP set = {total_A:.6f}")

    # optional: write a little note
    out_txt = RUNS_DIR / "miqp_validated_with_ode.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"MIQP set validated with ODE\n")
        f.write(f"Sites: {sites}\n")
        f.write(f"Sum A (t=1000): {total_A:.6f}\n")
    print(f"[validate] wrote {out_txt}")


if __name__ == "__main__":
    main()
