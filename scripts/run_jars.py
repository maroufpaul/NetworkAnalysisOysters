"""
Run the JARS biology simulation on any set of reefs, straight from the terminal.

This is the honest scorer (the full ODE), exposed as a command so you don't have
to write Python. Pass site LABELS (10, 37, ...), not the 0-48 AMPL indices.

By default it runs every matrix x supply-mode combination (M1/M2 x
constant/realistic = 4 results). Narrow it with --matrix and/or --p0.

    # one reef, all 4 combos (M1/M2 x constant/realistic)
    python -m scripts.run_jars --sites 37

    # a chosen set, all 4 combos
    python -m scripts.run_jars --sites 10 31 37 40 41 49 53

    # all 49 candidates, all 4 combos
    python -m scripts.run_jars --all

    # narrow down with flags
    python -m scripts.run_jars --sites 10 40 41 --matrix 2
    python -m scripts.run_jars --sites 10 40 41 --p0 constant
    python -m scripts.run_jars --sites 10 40 41 --matrix 2 --p0 constant

    # per-reef densities, and save to CSV
    python -m scripts.run_jars --all --densities --csv runs/my_jars_run.csv
    
    # just per-reef densities
    python -m scripts.run_jars --all --densities 
    
    # every listed reef ALONE (isolated), batched
    python -m scripts.run_jars --each --sites 10 31 37 40 41 49 53

    # all 49, each alone
    python -m scripts.run_jars --each --all

    # still respects --matrix / --p0 (defaults to all 4 combos)
    python -m scripts.run_jars --each --sites 10 37 --matrix 1 --p0 realistic

Flags:
    --sites N [N ...]         reef labels to simulate
    --all                     use all 49 candidates (instead of --sites)
    --matrix 1|2|both         which matrix        (default both)
    --p0 constant|realistic|both   external supply (default both)
    --densities               print each reef's equilibrium adult density
    --csv PATH                write per-site densities (tagged by matrix + p0)
"""
import argparse

import pandas as pd

import config
from src.model.jars_ode import load_connectivity
from src.opt.evaluator import evaluate_subset


def run_each(sites, matrix_id, p0_mode):
    """Every reef simulated ALONE (isolated), one matrix + one supply mode.
    This is the batch version of calling --sites with a single reef, repeated."""
    conn, key = load_connectivity(config.MATRICES[config.matrix_key(matrix_id)])

    print("=" * 56)
    print(f"M{matrix_id}  |  supply={p0_mode}  |  {len(sites)} reef(s), each ALONE")
    print("=" * 56)
    print("  reef        F (solo)")
    rows = []
    for site in sorted(sites):
        # one reef by itself -> F is just that reef's own adult density
        f = evaluate_subset([site], conn, key, P0_mode=p0_mode,
                            consP0=config.CONST_P0)
        dead = "   (dies alone)" if f <= 1e-6 else ""
        print(f"  {site:>3}     {f:.6f}{dead}")
        rows.append({"matrix": f"M{matrix_id}", "p0": p0_mode, "site": site,
                     "F_solo": round(f, 6)})
    print()
    return rows


def run_one(sites, matrix_id, p0_mode, show_densities):
    """One matrix + one supply mode. Returns rows for the optional CSV."""
    conn, key = load_connectivity(config.MATRICES[config.matrix_key(matrix_id)])
    F, dens = evaluate_subset(sites, conn, key, P0_mode=p0_mode,
                              consP0=config.CONST_P0, return_densities=True) # this evaluates K number of sites and calls real JARS code

    print("=" * 56)
    print(f"M{matrix_id}  |  supply={p0_mode}  |  {len(sites)} reef(s)")
    print("=" * 56)
    print(f"  sites: {sorted(sites)}")
    print(f"  F (total adult biomass at t={config.TMAX}) = {F:.6f}")
    if show_densities:
        print("  reef   adult density")
        for site, d in sorted(dens.items()):
            print(f"  {site:>3}   {d:.6f}")
    print()

    return [{"matrix": f"M{matrix_id}", "p0": p0_mode, "site": s,
             "adult_density": round(dens[s], 6)} for s in sorted(dens)]


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sites", type=int, nargs="+", help="reef labels, e.g. 10 40 41")
    g.add_argument("--all", action="store_true", help="use all 49 candidate reefs")
    ap.add_argument("--matrix", choices=["1", "2", "both"], default="both")
    ap.add_argument("--p0", choices=["constant", "realistic", "both"], default="both")
    ap.add_argument("--each", action="store_true",
                    help="run every listed reef ALONE (isolated), not together")
    ap.add_argument("--densities", action="store_true", help="print per-reef density")
    ap.add_argument("--csv", default=None, help="save per-reef densities here")
    args = ap.parse_args()

    sites = config.CANDIDATE_SITES.tolist() if args.all else args.sites

    # catch a bad label early with a clear message instead of a confusing crash
    valid = set(int(s) for s in config.CANDIDATE_SITES)
    bad = [s for s in sites if s not in valid]
    if bad:
        raise SystemExit(f"not candidate reefs: {bad}\nvalid labels: {sorted(valid)}")

    mats = ["1", "2"] if args.matrix == "both" else [args.matrix]
    modes = ["constant", "realistic"] if args.p0 == "both" else [args.p0]

    all_rows = []
    for m in mats:
        for mode in modes:
            if args.each:
                all_rows += run_each(sites, m, mode)
            else:
                all_rows += run_one(sites, m, mode, args.densities)

    # summary line so a multi-combo run is easy to read at a glance.
    # for --each there is no single F (each reef is separate), so skip it.
    if not args.each and len(mats) * len(modes) > 1:
        print("summary (F by matrix x supply):")
        for m in mats:
            for mode in modes:
                F = sum(r["adult_density"] for r in all_rows
                        if r["matrix"] == f"M{m}" and r["p0"] == mode)
                print(f"  M{m} {mode:<10} F = {F:.6f}")

    if args.csv:
        config.RUNS_DIR.mkdir(exist_ok=True)
        pd.DataFrame(all_rows).to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()