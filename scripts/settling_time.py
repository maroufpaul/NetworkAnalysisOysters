"""
scripts/settling_time.py

Reproduces the equilibrium-convergence numbers cited in the paper's
"t = 1000" discussion (Sections 4-5). For each connectivity matrix it:

  1. Validates the pipeline by reproducing the published F-value of the
     constant-forcing swap design (M1 -> 1.880054, M2 -> 1.733033).
  2. Measures how long adult biomass (the quantity F(S) reports) takes to
     settle, at 1% and 0.1% tolerance, for the 7-site backbone and the
     25-site design.
  3. Reports the slow sediment-compartment settling time.
  4. Checks horizon stability: how much total adult biomass moves between
     t = 1000 and t = 2000 (the justification for both the horizon and the
     six-decimal reporting).

Model time is in YEARS (the JARS rate parameters are annual rates).

Run from the repository root:
    python -m scripts.settling_time
    python -m scripts.settling_time --matrix data/nk_All_060102final_56sites_Model.xlsx
"""

import argparse
import numpy as np
from scipy.integrate import solve_ivp

# Single source of truth: reuse the model code already in the repo.
from src.model.jars_ode import (
    load_connectivity,
    sitetoindex,
    setP0,
    odesys,
)

# Constant-forcing swap/stingy designs and the global backbone (from the paper).
DESIGNS = {
    "M1": {
        "file_hint": "060102",  # dry year 2002
        "swap": [10, 12, 15, 16, 17, 20, 21, 26, 28, 29, 31, 32, 33, 36, 37,
                 40, 41, 44, 47, 49, 51, 52, 53, 54, 59],
        "published_F": 1.880054,
    },
    "M2": {
        "file_hint": "060103",  # moderate-to-high flow year 2003
        "swap": [10, 11, 12, 15, 16, 17, 19, 20, 26, 29, 31, 32, 33, 36, 37,
                 40, 41, 44, 47, 49, 51, 52, 53, 54, 59],
        "published_F": 1.733033,
    },
}
BACKBONE = [10, 31, 37, 40, 41, 49, 53]


def integrate(site_labels, connectivity, key_all, P0_mode="constant",
              consP0=170.0, P1scaling=0.5, mu_val=0.4, tmax=1000, npts=4001):
    """Integrate JARS on a subset; return (solution, n_sites) with a dense
    time grid so trajectories can be inspected."""
    idx = sitetoindex(key_all, np.array(site_labels, dtype=int))
    if len(idx) == 0:
        raise ValueError("none of the requested sites are in this matrix")
    P1 = P1scaling * connectivity[np.ix_(idx, idx)]
    key_subset = key_all[idx]
    n = len(key_subset)
    if P0_mode == "constant":
        P0 = consP0 * np.ones(n)
    elif P0_mode == "realistic":
        P0 = setP0(key_subset)
    else:
        P0 = np.zeros(n)
    mu = mu_val * np.ones(n)
    v0 = np.zeros(4 * n)
    v0[n:2 * n] = 0.2      # A0  (generous initial conditions)
    v0[2 * n:3 * n] = 0.3  # R0
    t_eval = np.linspace(0, tmax, npts)
    sol = solve_ivp(lambda t, v: odesys(t, v, P0, P1, mu),
                    [0, tmax], v0, method="RK45", rtol=1e-6, t_eval=t_eval)
    return sol, n


def settling_time(sol, n, comp="A", tol=1e-3):
    """Last time (years) at which the summed `comp` biomass is still more than
    `tol` (relative) away from its final value. comp in {J, A, R, S}."""
    ci = {"J": 0, "A": 1, "R": 2, "S": 3}[comp]
    total = sol.y[ci * n:(ci + 1) * n, :].sum(axis=0)
    final = total[-1]
    rel = np.abs(total - final) / max(abs(final), 1e-12)
    above = np.where(rel > tol)[0]
    return (0.0 if len(above) == 0 else float(sol.t[above[-1]])), float(final)


def report_matrix(name, connectivity, key_all):
    spec = DESIGNS[name]
    print(f"\n===== Matrix {name} =====")

    # 1. validation against the published F-value
    sol, n = integrate(spec["swap"], connectivity, key_all, P0_mode="constant")
    _, f_swap = settling_time(sol, n, "A")
    print(f"  swap-design sum A (constant P0=170): {f_swap:.6f}   "
          f"(published {spec['published_F']:.6f})")

    # 2. adult settling times
    for label, sites in [("7-site backbone", BACKBONE),
                         ("25-site design", spec["swap"])]:
        sol, n = integrate(sites, connectivity, key_all, P0_mode="constant")
        t1, finA = settling_time(sol, n, "A", tol=1e-2)
        t01, _ = settling_time(sol, n, "A", tol=1e-3)
        tS, _ = settling_time(sol, n, "S", tol=1e-3)
        print(f"  {label:15s}: adults settle ~{t1:5.0f} yr (1%) / "
              f"~{t01:5.0f} yr (0.1%) | sediment ~{tS:4.0f} yr | "
              f"final A = {finA:.6f}")

    # 3. horizon stability (justifies the t=1000 choice and the precision)
    sol, n = integrate(spec["swap"], connectivity, key_all,
                       P0_mode="constant", tmax=2000, npts=8001)
    total = sol.y[n:2 * n, :].sum(axis=0)
    vals = {T: total[np.argmin(np.abs(sol.t - T))] for T in (1000, 2000)}
    print(f"  horizon check: sum A at t=1000 = {vals[1000]:.6f}, "
          f"t=2000 = {vals[2000]:.6f}  (|delta| = {abs(vals[1000]-vals[2000]):.2e})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", default=None,
                    help="path to one connectivity .xlsx; default runs both M1 and M2")
    args = ap.parse_args()

    if args.matrix is not None:
        connectivity, key_all = load_connectivity(args.matrix)
        name = "M1" if DESIGNS["M1"]["file_hint"] in args.matrix else \
               "M2" if DESIGNS["M2"]["file_hint"] in args.matrix else "M1"
        report_matrix(name, connectivity, key_all)
    else:
        # default file names in data/ — edit here if yours differ
        files = {
            "M1": "data/nk_All_060102final_56sites_Model.xlsx",
            "M2": "data/nk_All_060103final_56sites_Model.xlsx",
        }
        for name, path in files.items():
            connectivity, key_all = load_connectivity(path)
            report_matrix(name, connectivity, key_all)

    print("\n(Model time is in years; F(S) is the steady-state adult biomass.)")


if __name__ == "__main__":
    main()