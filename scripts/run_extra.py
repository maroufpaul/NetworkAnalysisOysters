"""
The leftover experiments from the paper that don't need their own script.

    python -m scripts.run_extra                # all of them
    python -m scripts.run_extra --exp backbone
    python -m scripts.run_extra --exp fidelity

  backbone    Which sites does EVERY method pick? Reads whatever's already in
              runs/ (heuristics + miqp csvs), intersects the designs. This is
              the headline result, so run the other scripts first.

  fidelity    The MIQP optimizes a quadratic stand-in, not the real ODE. So: take
              what the MIQP picked, score it under the actual ODE, compare to the
              best heuristic. Tells you how much the shortcut costs you.

  ksweep      Re-solve base for K = 15, 20, 25, 30. Are the smaller answers just
              subsets of the bigger ones? If yes the surrogate is ranking reefs
              consistently rather than jumping around.

  selfrecruit Reefs feed themselves (the diagonal of the connectivity matrix).
              Zero it out and see if the answer changes. It shouldn't, much.

Writes runs/extra_<exp>.txt
"""
import argparse

import numpy as np
import pandas as pd
from amplpy import AMPL

import config
from src.model.jars_ode import load_connectivity
from src.opt.evaluator import evaluate_subset
from scripts.run_iterated import load


def solve_base(matrix_id, labels, P1, Pe, K, zero_diag=False):
    """Base MIQP with knobs for K and whether reefs feed themselves."""
    W = P1 * (config.A_STAR ** config.ALPHA)
    if zero_diag:
        W = W.copy()
        np.fill_diagonal(W, 0.0)
    n = len(labels)
    lines = ["# written by run_extra.py", "", "set N :="]
    lines += [f"  {i}" for i in range(n)]
    lines += [";", "", f"param K := {K};", "", "param Pe :="]
    lines += [f"  {i} {Pe[i]:.6f}" for i in range(n)]
    lines += [";", "", "param W :="]
    lines += [f"  [{i}, {j}] {W[i, j]:.6f}"
              for i in range(n) for j in range(n) if W[i, j] != 0]
    lines += [";", ""]
    dat = config.AMPL_DIR / "oyster_extra.dat"
    dat.write_text("\n".join(lines), encoding="utf-8")

    ampl = AMPL()
    ampl.eval("option solver gurobi;")
    ampl.eval(f"option gurobi_options '{config.GUROBI_OPTIONS}';")
    ampl.eval("option solver_msg 0;")
    ampl.read(str(config.AMPL_DIR / "oyster_quad.mod"))
    ampl.readData(str(dat))
    ampl.eval("solve;")
    site_labels = pd.read_csv(config.mapping_csv(matrix_id))["site_id"].tolist()
    picked = sorted(site_labels[int(r[0])]
                    for r in ampl.getVariable("x").getValues().to_list()
                    if float(r[1]) > 0.5)
    obj = float(ampl.getObjective("score").value())
    ampl.close()
    return picked, obj


def exp_backbone(L):
    """Intersect every design we've got sitting in runs/."""
    L("=" * 70)
    L("BACKBONE -- sites that every single method picks")
    L("=" * 70)

    # two rules here, on purpose:
    #   global backbone -> skip greedy. It's only the starting point for swap,
    #     and its raw answer isn't one you'd use. (On M1 realistic it misses
    #     site 37, so leaving it in would knock 37 out of the core for a silly
    #     reason.)
    #   per-matrix      -> keep greedy. It's every constant-Pe run we did.
    post = {}       # swap + stingy + miqp
    allruns = {}    # + greedy

    h = config.RUNS_DIR / "heuristics_all.csv"
    if h.exists():
        for _, r in pd.read_csv(h).iterrows():
            key = f"{r['matrix']} {r['method']} {r['p0']}"
            sites = set(int(x) for x in str(r["sites"]).split())
            allruns[key] = sites
            if r["method"] != "greedy":
                post[key] = sites

    m = config.RUNS_DIR / "miqp_all.csv"
    if m.exists():
        for _, r in pd.read_csv(m).iterrows():
            key = f"{r['matrix']} MIQP {r['model']}"
            sites = set(int(x) for x in str(r["sites"]).split())
            allruns[key] = sites
            post[key] = sites

    if not post:
        L("  Nothing in runs/ yet. Run run_heuristics.py and run_miqp.py first.")
        return

    L(f"  {len(post)} designs (greedy left out -- it just seeds swap):")
    for k in sorted(post):
        L(f"    {k}")
    core = sorted(set.intersection(*post.values()))
    L("")
    L(f"  PICKED BY EVERYTHING ({len(core)} sites): {core}")
    L("")

    # per matrix, constant Pe only, greedy included this time
    L("  per matrix (constant Pe runs only, greedy included):")
    per = {}
    for mat in ("M1", "M2"):
        sub = [v for k, v in allruns.items()
               if k.startswith(mat) and ("constant" in k or "MIQP" in k)]
        if sub:
            per[mat] = set.intersection(*sub)
            L(f"    {mat}: {len(per[mat])} sites  {sorted(per[mat])}")
    if len(per) == 2:
        cross = sorted(per["M1"] & per["M2"])
        L(f"    both matrices: {len(cross)} sites  {cross}")


def exp_fidelity(L, mats):
    """How much does the quadratic shortcut actually cost?"""
    L("=" * 70)
    L("FIDELITY -- score the MIQP's picks under the real ODE")
    L("=" * 70)
    csv = config.RUNS_DIR / "miqp_all.csv"
    if not csv.exists():
        L("  Need runs/miqp_all.csv first: python -m scripts.run_miqp")
        return
    df = pd.read_csv(csv)
    heur = config.RUNS_DIR / "heuristics_all.csv"
    best_by_matrix = {}
    if heur.exists():
        hd = pd.read_csv(heur)
        for mat in hd.matrix.unique():
            sub = hd[(hd.matrix == mat) & (hd.p0 == "constant")]
            best_by_matrix[mat] = sub.F.max()

    for mat in sorted(df.matrix.unique()):
        conn, key = load_connectivity(config.MATRICES[mat])
        best = best_by_matrix.get(mat)
        L(f"\n  {mat}" + (f"   best heuristic F = {best:.6f}" if best else ""))
        for _, r in df[df.matrix == mat].iterrows():
            sites = [int(x) for x in str(r["sites"]).split()]
            f = evaluate_subset(sites, conn, key, P0_mode="constant",
                                consP0=config.CONST_P0)
            pct = f"   {100*f/best:5.1f}% of best" if best else ""
            L(f"    {r['model']:10s} ODE F = {f:.6f}{pct}")


def exp_ksweep(L, mats):
    """Does a bigger budget just add sites, or does it rethink everything?"""
    L("=" * 70)
    L("K SWEEP -- are smaller answers subsets of bigger ones?")
    L("=" * 70)
    for m in mats:
        conn, key, labels, P1, Pe_real = load(m)
        Pe = np.full(len(labels), config.CONST_P0)
        L(f"\n  M{m}")
        picks = {}
        for K in (15, 20, 25, 30):
            sites, obj = solve_base(m, labels, P1, Pe, K)
            picks[K] = set(sites)
            L(f"    K={K:<3} obj={obj:10.2f}  {sorted(sites)}")
        ks = [15, 20, 25, 30]
        nested = all(picks[ks[i]] < picks[ks[i + 1]] for i in range(3))
        L(f"    nested (each one inside the next)? {nested}")


def exp_selfrecruit(L, mats):
    """Reefs feeding themselves -- does it change who we pick?"""
    L("=" * 70)
    L("SELF-RECRUITMENT -- what if reefs didn't feed themselves?")
    L("=" * 70)
    for m in mats:
        conn, key, labels, P1, Pe_real = load(m)
        Pe = np.full(len(labels), config.CONST_P0)
        kept, obj_kept = solve_base(m, labels, P1, Pe, config.K, zero_diag=False)
        zeroed, obj_zero = solve_base(m, labels, P1, Pe, config.K, zero_diag=True)
        L(f"\n  M{m}")
        L(f"    diagonal kept   : obj={obj_kept:10.2f}")
        L(f"    diagonal zeroed : obj={obj_zero:10.2f}   "
          f"({100*(obj_kept-obj_zero)/obj_kept:.1f}% lower)")
        L(f"    same sites picked? {kept == zeroed}")
        if kept != zeroed:
            L(f"      only in kept   : {sorted(set(kept) - set(zeroed))}")
            L(f"      only in zeroed : {sorted(set(zeroed) - set(kept))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["backbone", "fidelity", "ksweep",
                                      "selfrecruit", "all"], default="all")
    ap.add_argument("--matrix", choices=["1", "2", "both"], default="both")
    args = ap.parse_args()
    mats = ["1", "2"] if args.matrix == "both" else [args.matrix]

    out = []
    def L(line=""):
        print(line)
        out.append(line)

    if args.exp in ("backbone", "all"):
        exp_backbone(L); L()
    if args.exp in ("fidelity", "all"):
        exp_fidelity(L, mats); L()
    if args.exp in ("ksweep", "all"):
        exp_ksweep(L, mats); L()
    if args.exp in ("selfrecruit", "all"):
        exp_selfrecruit(L, mats); L()

    config.RUNS_DIR.mkdir(exist_ok=True)
    p = config.RUNS_DIR / f"extra_{args.exp}.txt"
    p.write_text("\n".join(out), encoding="utf-8")
    print(f"-> runs/extra_{args.exp}.txt")


if __name__ == "__main__":
    main()