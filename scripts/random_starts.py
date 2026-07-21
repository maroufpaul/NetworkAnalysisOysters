"""
Initialization-robustness check for the network-aware iterated surrogate.

Instead of the four hand-picked constant starts (0.02, 0.05675, 0.2, 0.5), this
runs the SAME network-aware iteration from many RANDOM initial density vectors,
each entry drawn uniformly from the sweep range [0.01, 0.5], and reports how many
distinct final designs come out. If they all converge to one design, the result
does not depend on the starting values, which is the point the four fixed starts
were meant to make.

Reuses scripts.run_iterated.iterate / load / alone_densities unchanged (real
MIQP via Gurobi), so it must be run where AMPL/Gurobi are available.

    python -m scripts.random_starts --matrix 1 --starts 50
    python -m scripts.random_starts --matrix 2 --starts 50 --seed 0

Writes runs/random_starts_M<>.csv.
"""
import argparse
import numpy as np
import pandas as pd

import config
from scripts.run_iterated import load, alone_densities, iterate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="both",
                    help="1, 2, or both (default: both)")
    ap.add_argument("--starts", type=int, default=50, help="number of random initializations")
    ap.add_argument("--lo", type=float, default=0.01, help="low end of the A0 range")
    ap.add_argument("--hi", type=float, default=0.5, help="high end of the A0 range")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    mats = ["1", "2"] if a.matrix == "both" else [a.matrix]
    for mi, mat in enumerate(mats):
        run_matrix(mat, a)


def run_matrix(matrix, a):
    conn, key, labels, P1, Pe = load(matrix)
    alone = alone_densities(P1, Pe)
    n = len(labels)
    rng = np.random.default_rng(a.seed)
    mtag = f"M{config.matrix_num(matrix)}"

    print(f"\n=== network-aware iteration from {a.starts} random starts | {mtag} "
          f"| realistic Pe | A0 entries ~ U[{a.lo}, {a.hi}]^{n} ===")

    rows = []
    designs = {}          # frozenset(sites) -> F
    for i in range(a.starts):
        A0 = rng.uniform(a.lo, a.hi, size=n)          # random per-reef start
        res = iterate(matrix, A0, "network", alone, labels, P1, Pe, conn, key)
        fs = frozenset(res["sites"])
        designs.setdefault(fs, res["F"])
        rows.append(dict(start=i, F=res["F"], updates=res["updates"],
                         status=res["status"], n_sites=len(res["sites"])))
        print(f"  start {i:3d}: F={res['F']:.6f}  updates={res['updates']}  "
              f"{res['status']}  (distinct designs so far: {len(designs)})")

    df = pd.DataFrame(rows)
    config.RUNS_DIR.mkdir(exist_ok=True)
    out = config.RUNS_DIR / f"random_starts_{mtag}.csv"
    df.to_csv(out, index=False)

    print("\n===== summary =====")
    print(f"  random starts run:        {a.starts}")
    print(f"  distinct final designs:   {len(designs)}")
    print(f"  all converged to one set: {len(designs) == 1}")
    print(f"  final F values:           {sorted(round(v,6) for v in designs.values())}")
    print(f"  all reached fixed point:  {(df['status']=='fixed_point').all()}")
    if len(designs) == 1:
        only = sorted(next(iter(designs)))
        print(f"  the single design:        {only}")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()