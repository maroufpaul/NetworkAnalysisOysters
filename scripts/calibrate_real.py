"""
Optimality-gap calibration on exhaustively enumerated INDUCED SUBINSTANCES of
the TSPS connectivity network.

Motivation
----------
On the full instance the true optimum F* of the JARS objective is not
computable (C(49,25) ~ 6.3e13 ODE solves), so the paper reports scores relative
to the best heuristic found -- a self-referential denominator. Here we take
induced subinstances small enough to enumerate: draw n of the 49 real candidate
sites, keep the induced sub-matrix of the real connectivity network, hold the
selection ratio K/n at the full-problem value (~25/49), and enumerate all
C(n,K) subsets under the SAME JARS F. That yields the true F* for the
subinstance, so every method is scored as a real optimality gap, not a gap to
another heuristic.

What is measured, per subinstance (all under the identical ODE F):
  * F*  : true optimum by exhaustive enumeration,
  * greedy, greedy+swap, stingy  (shared impl, src/opt/heuristics.py),
  * the frozen-density surrogate optimum, then re-scored under F.

The surrogate optimum is found by exhaustive enumeration of the SAME quadratic
the paper's MIQP maximizes; on an enumerable subinstance with no side
constraints this returns the identical global optimum Gurobi would, so no AMPL/
Gurobi call is needed. A_* is frozen exactly as in the paper: the median
positive equilibrium density of the all-restored subinstance under CONSTANT
forcing (default), or the paper's fixed value via --astar paper. Under constant
Pe the A_* choice cannot change the surrogate selection (it only rescales the
objective); it matters only under realistic Pe.

    python -m scripts.calibrate_real --n 12                 # K auto = 6
    python -m scripts.calibrate_real --n 10 --seeds 50 --p0 realistic
    python -m scripts.calibrate_real --n 14 --matrix 2 --seed-start 0 --seeds 15

Writes runs/calibrate_real.csv (appended per subinstance).
"""
import argparse, math, time, itertools
import numpy as np, pandas as pd
import config
from src.model.jars_ode import load_connectivity
from src.opt.evaluator import evaluate_subset
from src.opt import heuristics as H

RATIO = 25 / 49                      # full-problem selection ratio
FAST = dict(method="LSODA", rtol=1e-5)

# F cache keyed by (matrix, p0, frozenset(sites)). F depends only on those, so
# it is valid to reuse across seeds/heuristics/enumeration within a run.
_CACHE = {}


def auto_K(n, K=None):
    return K if K is not None else int(np.floor(RATIO * n + 0.5))


def make_scorer(C, key, p0, matrix_tag):
    """score_one / score_many closures with memoisation."""
    def score_one(sites):
        fs = frozenset(int(s) for s in sites)
        ck = (matrix_tag, p0, fs)
        v = _CACHE.get(ck)
        if v is None:
            v = evaluate_subset(list(fs), C, key, tmax=config.TMAX,
                                P1scaling=config.P1SCALING, P0_mode=p0,
                                consP0=config.CONST_P0, **FAST)
            _CACHE[ck] = v
        return v
    def score_many(sets):
        return [score_one(s) for s in sets]
    return score_one, score_many


def frozen_astar(pool, C, key, mode):
    """A_* as the paper freezes it: median positive equilibrium of the
    all-restored (sub)instance under CONSTANT forcing. --astar paper uses the
    fixed full-49 value instead."""
    if mode == "paper":
        return config.A_STAR
    _, dens = evaluate_subset(list(pool), C, key, tmax=config.TMAX,
                              P1scaling=config.P1SCALING, P0_mode="constant",
                              consP0=config.CONST_P0, return_densities=True, **FAST)
    pos = np.array([v for v in dens.values() if v > 1e-9])
    return float(np.median(pos)) if len(pos) else config.A_STAR


def surrogate_opt(pool, C, key, K, p0, astar):
    """Exhaustive optimum of the paper's frozen-density surrogate on the
    subinstance:  Ftilde(S) = sum_k Pe_k + A_*^ALPHA * sum_{l,k in S} P1_lk."""
    idx = {lab: int(np.where(key == lab)[0][0]) for lab in pool}
    if p0 == "constant":
        pe = {lab: config.CONST_P0 for lab in pool}
    else:
        pe = {lab: float(config.P0_REALISTIC[lab - 1]) for lab in pool}
    W = config.P1SCALING * (astar ** config.ALPHA)      # scalar (frozen density)
    best, bestS = -np.inf, None
    for S in itertools.combinations(pool, K):
        ext = sum(pe[k] for k in S)
        internal = sum(C[idx[l], idx[k]] for l in S for k in S) * W
        val = ext + internal
        if val > best:
            best, bestS = val, list(S)
    return bestS


def run(matrix, n, K, seeds, seed_start, p0, astar_mode):
    C, key = load_connectivity(config.MATRICES[config.matrix_key(matrix)])
    mtag = f"M{config.matrix_num(matrix)}"
    cand = config.CANDIDATE_SITES.tolist()
    ncomb = math.comb(n, K)
    print(f"\n=== induced subinstances of {mtag} | n={n} of 49, K={K} "
          f"(ratio {K/n:.3f}, target {RATIO:.3f}) | Pe={p0} | "
          f"A_*={astar_mode} | {ncomb} subsets x {seeds} ===")
    rows = []
    for seed in range(seed_start, seed_start + seeds):
        t = time.time()
        rng = np.random.default_rng(1000 + seed)
        pool = sorted(rng.choice(cand, size=n, replace=False).tolist())
        score_one, score_many = make_scorer(C, key, p0, mtag)

        bestF, bestS = -np.inf, None
        for S in itertools.combinations(pool, K):
            f = score_one(S)
            if f > bestF:
                bestF, bestS = f, set(S)
        if bestF <= 1e-9:
            print(f"  seed {seed}: degenerate (F*<=0), skipped"); continue

        g_s, g, _ = H.greedy(score_many, pool, K)
        sw_s, sw, _ = H.swap(score_many, g_s, pool, K)
        st_s, st, _ = H.stingy(score_many, pool, K)
        astar = frozen_astar(pool, C, key, astar_mode)
        surrS = surrogate_opt(pool, C, key, K, p0, astar)
        surrF = score_one(surrS)

        tol = 1e-6 * max(1.0, abs(bestF))
        gap = lambda x: 100 * (bestF - x) / bestF
        row = dict(
            matrix=mtag, seed=seed, n=n, K=K, p0=p0, astar_mode=astar_mode,
            astar=round(astar, 6), Fstar=bestF,
            greedy_gap_pct=gap(g), swap_gap_pct=gap(sw),
            stingy_gap_pct=gap(st), surrogate_gap_pct=gap(surrF),
            greedy_opt=int(bestF - g <= tol),
            swap_opt=int(bestF - sw <= tol),
            stingy_opt=int(bestF - st <= tol),
            surr_opt=int(bestF - surrF <= tol),
            swap_same_set=int(set(sw_s) == bestS),
            surr_same_set=int(set(surrS) == bestS),
            pool=" ".join(map(str, pool)),
        )
        rows.append(row)
        config.RUNS_DIR.mkdir(exist_ok=True)
        p = config.RUNS_DIR / "calibrate_real.csv"
        pd.DataFrame([row]).to_csv(p, mode="a", header=not p.exists(), index=False)
        print(f"  seed {seed:2d}: F*={bestF:.4f}  gaps%  greedy={gap(g):5.2f} "
              f"swap={gap(sw):5.2f} stingy={gap(st):5.2f} surr={gap(surrF):6.2f}  "
              f"({time.time()-t:.0f}s, cache={len(_CACHE)})")
    return rows


def summarize(rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    print("\n===== optimality gaps (0.00% == found the true optimum) =====")
    pairs = [("greedy_gap_pct", "greedy", "greedy_opt"),
             ("swap_gap_pct", "greedy+swap", "swap_opt"),
             ("stingy_gap_pct", "stingy", "stingy_opt"),
             ("surrogate_gap_pct", "frozen surrogate", "surr_opt")]
    for col, nm, optcol in pairs:
        v = df[col]
        print(f"  {nm:18s} mean {v.mean():5.2f}%  median {v.median():5.2f}%  "
              f"worst {v.max():6.2f}%  optimal {int(df[optcol].sum())}/{len(df)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="1")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--K", type=int, default=None,
                    help="default: round(25/49 * n) to preserve selection ratio")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--p0", choices=["constant", "realistic"], default="realistic")
    ap.add_argument("--astar", choices=["subpool", "paper"], default="subpool",
                    help="subpool = paper's construction on the subinstance; "
                         "paper = fixed full-49 A_* value")
    a = ap.parse_args()
    K = auto_K(a.n, a.K)
    rows = run(a.matrix, a.n, K, a.seeds, a.seed_start, a.p0, a.astar)
    summarize(rows)


if __name__ == "__main__":
    main()