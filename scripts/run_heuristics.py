"""
ODE heuristics. These search over site sets by actually simulating JARS, so they
optimize the real thing instead of the MIQP's quadratic approximation. Every
score is a full ODE solve, which is why this is slow -- greedy alone is ~900 of
them. It runs them across your CPU cores, otherwise it'd take hours.

    python -m scripts.run_heuristics                    # everything, ~15 min
    python -m scripts.run_heuristics --method greedy
    python -m scripts.run_heuristics --matrix 1 --p0 constant
    python -m scripts.run_heuristics --workers 4        

Methods:
    greedy   start empty, keep adding whichever site helps most. Fast, myopic.
    swap     take greedy's answer, try swapping one out for one in, keep the best
             swap, repeat. Usually squeezes out another 1-2%.
    stingy   start with all 49, keep deleting whichever you miss least. Comes at
             the problem backwards, so if it lands where swap did, that's a
             decent sign you're at a real local optimum.

swap starts from greedy's answer, so asking for swap runs greedy too.

Pe modes:
    constant   everyone gets 170. Differences come only from the network.
    realistic  the actual low/moderate/high classes (100/200/400).

Writes runs/heuristics_<method>.txt and .csv
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import config

# Each worker process loads the connectivity matrix once and reuses it. Globals
# are ugly but this is how you avoid re-reading a 56x56 xlsx 900 times.
_CONN = None
_KEY = None


def _setup(matrix_path):
    global _CONN, _KEY
    from src.model.jars_ode import load_connectivity
    _CONN, _KEY = load_connectivity(matrix_path)


def _score(job):
    """Runs in a worker. job = (sites, p0_mode) -> F"""
    from src.opt.evaluator import evaluate_subset
    sites, p0_mode = job
    return evaluate_subset(list(sites), _CONN, _KEY, tmax=config.TMAX,
                           P1scaling=config.P1SCALING, P0_mode=p0_mode,
                           consP0=config.CONST_P0)


def score_many(pool, list_of_site_sets, p0_mode):
    """Score a bunch of candidate sets at once, spread over the workers."""
    return list(pool.map(_score, [(tuple(s), p0_mode) for s in list_of_site_sets]))


def score_one(pool, sites, p0_mode):
    return score_many(pool, [sites], p0_mode)[0]


def greedy(pool, p0_mode, tag):
    chosen, left = [], config.CANDIDATE_SITES.tolist()
    best = 0.0
    for step in range(1, config.K + 1):
        t = time.time()
        scores = score_many(pool, [chosen + [c] for c in left], p0_mode)
        i = int(np.argmax(scores))                 # ties -> first one
        chosen.append(left.pop(i))
        best = scores[i]
        print(f"    [{tag}] greedy {step:2d}/{config.K}  +{chosen[-1]:<3} "
              f"F={best:.6f}  ({time.time()-t:.0f}s)", flush=True)
    return sorted(chosen), best


def swap(pool, start, p0_mode, tag, max_passes=50):
    cur = list(start)
    score = score_one(pool, cur, p0_mode)
    for p in range(1, max_passes + 1):
        t = time.time()
        outside = [s for s in config.CANDIDATE_SITES.tolist() if s not in cur]
        # build every possible 1-for-1 swap, score them all at once
        moves = [(out, inn) for out in cur for inn in outside]
        trials = [[inn if s == out else s for s in cur] for out, inn in moves]
        scores = score_many(pool, trials, p0_mode)
        i = int(np.argmax(scores))
        if scores[i] <= score + 1e-9:
            print(f"    [{tag}] swap pass {p}: no swap helps, done  "
                  f"({time.time()-t:.0f}s)", flush=True)
            break
        out, inn = moves[i]
        cur = trials[i]
        gain, score = scores[i] - score, scores[i]
        print(f"    [{tag}] swap pass {p}: {out} -> {inn}  +{gain:.6f}  "
              f"F={score:.6f}  ({time.time()-t:.0f}s)", flush=True)
    return sorted(cur), score


def stingy(pool, p0_mode, tag):
    S = config.CANDIDATE_SITES.tolist()
    score = score_one(pool, S, p0_mode)
    while len(S) > config.K:
        t = time.time()
        scores = score_many(pool, [[s for s in S if s != d] for d in S], p0_mode)
        i = int(np.argmax(scores))                 # whichever we miss least
        gone = S.pop(i)
        score = scores[i]
        print(f"    [{tag}] stingy {len(S):2d}/{config.K}  -{gone:<3} "
              f"F={score:.6f}  ({time.time()-t:.0f}s)", flush=True)
    return sorted(S), score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["greedy", "swap", "stingy", "all"], default="all")
    ap.add_argument("--matrix", choices=["1", "2", "both"], default="both")
    ap.add_argument("--p0", choices=["constant", "realistic", "both"], default="both")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    mats = ["1", "2"] if args.matrix == "both" else [args.matrix]
    modes = ["constant", "realistic"] if args.p0 == "both" else [args.p0]

    blocks, rows = [], []
    t_all = time.time()
    for m in mats:
        path = str(config.MATRICES[config.matrix_key(m)])
        # one pool per matrix, so the workers load that matrix once and keep it
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_setup, initargs=(path,)) as pool:
            for mode in modes:
                tag = f"M{m} {mode}"
                print(f"\n===== {tag}  ({args.workers} workers) =====", flush=True)
                got = {}
                if args.method in ("greedy", "swap", "all"):
                    got["greedy"] = greedy(pool, mode, tag)
                if args.method in ("swap", "all"):
                    got["swap"] = swap(pool, got["greedy"][0], mode, tag)
                if args.method in ("stingy", "all"):
                    got["stingy"] = stingy(pool, mode, tag)

                keep = ["greedy", "swap", "stingy"] if args.method == "all" else [args.method]
                for meth in keep:
                    sites, score = got[meth]
                    blocks.append("\n".join([
                        "=" * 70,
                        f"M{m}  |  {meth}  |  Pe = {mode}",
                        "=" * 70,
                        f"  F (adult biomass at equilibrium) : {score:.6f}",
                        f"  sites ({len(sites)}) : {sites}", ""]))
                    rows.append({"matrix": f"M{m}", "p0": mode, "method": meth,
                                 "F": round(score, 6), "sites": " ".join(map(str, sites))})
                    print(f"  {tag} {meth:7s} F={score:.6f}")

    config.RUNS_DIR.mkdir(exist_ok=True)
    (config.RUNS_DIR / f"heuristics_{args.method}.txt").write_text("\n".join(blocks),
                                                                   encoding="utf-8")
    pd.DataFrame(rows).to_csv(config.RUNS_DIR / f"heuristics_{args.method}.csv", index=False)
    print(f"\ndone in {(time.time()-t_all)/60:.1f} min -> "
          f"runs/heuristics_{args.method}.txt, .csv")


if __name__ == "__main__":
    main()