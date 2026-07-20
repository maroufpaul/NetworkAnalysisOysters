"""
Does the network-aware ITERATION recover what the frozen surrogate loses?

calibrate_real.py showed the one-shot frozen-density surrogate has a heavy tail
of large gaps under realistic Pe -- but that is the paper's PREMISE, not its
contribution. The contribution is the network-aware iterated surrogate. This
script tests it directly: on induced subinstances small enough to enumerate the
true optimum F*, it runs the SAME iteration loop as scripts.run_iterated (its F,
alone_densities, and network_fallback_densities are imported unchanged; only the
Gurobi MIQP is replaced by exact enumeration of the identical source-specific
quadratic, which is exact on these small instances), and reports:

  frozen gap   = 100 * (F* - F(frozen-surrogate set))   / F*
  iterated gap = 100 * (F* - F(network-aware iter set))  / F*

If the iteration works, iterated gap << frozen gap and lands near 0.

Realistic Pe only: under constant Pe the surrogate selection is invariant to the
source density (A_* only rescales), so the iteration is a no-op there and is
skipped with a note.

    python -m scripts.calibrate_iterated --n 10 --seeds 20
    python -m scripts.calibrate_iterated --n 12 --seeds 15 --matrix 2

Writes runs/calibrate_iterated.csv (appended per subinstance).
"""
import argparse, itertools, time
import numpy as np, pandas as pd
import config
from src.model.jars_ode import load_connectivity, setP0
from src.opt.evaluator import evaluate_subset
# run_iterated imports amplpy at load time for its Gurobi MIQP path, which we
# never call here (we enumerate instead). Stub it so the pure numpy functions we
# DO reuse import cleanly without an AMPL/Gurobi license.
import sys, types
if "amplpy" not in sys.modules:
    _stub = types.ModuleType("amplpy")
    _stub.AMPL = _stub.OutputHandler = object
    sys.modules["amplpy"] = _stub
# reuse the paper's iteration components verbatim
from scripts.run_iterated import F as F_real, alone_densities, network_fallback_densities
from scripts.calibrate_real import auto_K, frozen_astar, surrogate_opt

FAST = dict(method="LSODA", rtol=1e-5)
MAX_PASSES = config.ITER.get("max_passes", 12) if hasattr(config, "ITER") else 12


def Fscore(sites, conn, key):
    return evaluate_subset(list(sites), conn, key, P0_mode="realistic", **FAST)


def miqp_enum(sublabels, P1, Pe, A, K):
    """Exact optimum of the source-specific MIQP by enumeration -- identical
    objective to run_iterated.solve_miqp (W row-scaled by source density),
    just solved without Gurobi."""
    W = P1 * (np.maximum(A, 0.0) ** config.ALPHA)[:, None]   # W[l,k]
    n = len(sublabels)
    best, bestS = -np.inf, None
    for S in itertools.combinations(range(n), K):
        S = list(S)
        val = Pe[S].sum() + W[np.ix_(S, S)].sum()
        if val > best:
            best, bestS = val, S
    return sorted(int(sublabels[i]) for i in bestS)


def iterate_enum(sublabels, P1, Pe, conn, key, A0, alone, K):
    """run_iterated.iterate with solve_miqp -> miqp_enum; network fallback."""
    pos = {int(l): i for i, l in enumerate(sublabels)}
    A = np.array(A0, float)
    history, score_by_set = [], {}
    for _ in range(MAX_PASSES):
        picked = miqp_enum(sublabels, P1, Pe, A, K)
        pk = tuple(picked)
        if pk in history:
            if pk == history[-1]:
                return list(pk), score_by_set[pk], "fixed_point"
            cyc = history[history.index(pk):]
            best_key = max(cyc, key=lambda s: score_by_set[s])
            return list(best_key), score_by_set[best_key], "cycle"
        history.append(pk)
        score, dens = evaluate_subset(picked, conn, key, P0_mode="realistic",
                                      return_densities=True, **FAST)
        score_by_set[pk] = score
        A, _ = network_fallback_densities(sublabels, P1, Pe, picked, dens)
    best_key = max(history, key=lambda s: score_by_set[s])
    return list(best_key), score_by_set[best_key], "max_passes"


def run(matrix, n, K, seeds, seed_start):
    conn, key = load_connectivity(config.MATRICES[config.matrix_key(matrix)])
    mtag = f"M{config.matrix_num(matrix)}"
    cand = config.CANDIDATE_SITES.tolist()
    print(f"\n=== ITERATED vs FROZEN vs F*  |  {mtag}, n={n}, K={K}, realistic Pe "
          f"|  C(n,K)={np.math.comb(n,K) if hasattr(np,'math') else __import__('math').comb(n,K)} x {seeds} ===")
    rows = []
    for seed in range(seed_start, seed_start + seeds):
        t = time.time()
        rng = np.random.default_rng(1000 + seed)
        pool = sorted(rng.choice(cand, size=n, replace=False).tolist())
        sub = np.array(pool)
        idx = np.array([np.where(key == l)[0][0] for l in pool])
        P1 = conn[np.ix_(idx, idx)] * config.P1SCALING
        Pe = setP0(sub).astype(float)

        # true optimum
        bestF = -np.inf
        for S in itertools.combinations(pool, K):
            f = Fscore(S, conn, key)
            bestF = max(bestF, f)
        if bestF <= 1e-9:
            print(f"  seed {seed}: degenerate, skipped"); continue

        # frozen one-shot surrogate (paper's construction on the subinstance)
        astar = frozen_astar(pool, conn, key, "subpool")
        frozenS = surrogate_opt(pool, conn, key, K, "realistic", astar)
        frozenF = Fscore(frozenS, conn, key)

        # network-aware iteration (paper's loop, enumerated MIQP)
        alone = alone_densities(P1, Pe)
        iterS, iterF, status = iterate_enum(sub, P1, Pe, conn, key, alone, alone, K)

        fg, ig = 100*(bestF-frozenF)/bestF, 100*(bestF-iterF)/bestF
        row = dict(matrix=mtag, seed=seed, n=n, K=K, Fstar=bestF,
                   frozen_gap_pct=fg, iterated_gap_pct=ig,
                   iter_status=status, improved=int(ig < fg - 1e-9),
                   iter_optimal=int(bestF - iterF <= 1e-6*max(1.0, bestF)))
        rows.append(row)
        p = config.RUNS_DIR / "calibrate_iterated.csv"
        config.RUNS_DIR.mkdir(exist_ok=True)
        pd.DataFrame([row]).to_csv(p, mode="a", header=not p.exists(), index=False)
        print(f"  seed {seed:2d}: F*={bestF:.4f}  frozen gap={fg:6.2f}%  "
              f"iterated gap={ig:6.2f}%  ({status}, {time.time()-t:.0f}s)")
    return rows


def summarize(rows):
    if not rows: return
    df = pd.DataFrame(rows)
    print("\n===== frozen vs network-aware iterated (gap to true optimum) =====")
    for c, nm in [("frozen_gap_pct", "frozen one-shot"),
                  ("iterated_gap_pct", "network-aware iter")]:
        v = df[c]
        print(f"  {nm:20s} mean {v.mean():6.2f}%  median {v.median():5.2f}%  "
              f"worst {v.max():6.2f}%")
    print(f"  iteration improved on frozen in {int(df.improved.sum())}/{len(df)} "
          f"subinstances; reached true optimum in {int(df.iter_optimal.sum())}/{len(df)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="1")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed-start", type=int, default=0)
    a = ap.parse_args()
    K = auto_K(a.n, a.K)
    summarize(run(a.matrix, a.n, K, a.seeds, a.seed_start))


if __name__ == "__main__":
    main()