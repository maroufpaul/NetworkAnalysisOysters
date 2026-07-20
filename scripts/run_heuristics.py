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
from src.opt import heuristics as H

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


# The search logic now lives in src/opt/heuristics.py (one canonical copy,
# shared with scripts.calibrate_real). These wrappers just bind the parallel
# scorer to that core and keep the console progress line per step.

def greedy(pool, p0_mode, tag):
    t = time.time()
    sm = lambda sets: score_many(pool, sets, p0_mode)
    sites, best, _ = H.greedy(sm, config.CANDIDATE_SITES.tolist(), config.K)
    print(f"    [{tag}] greedy  F={best:.6f}  ({time.time()-t:.0f}s)", flush=True)
    return sites, best


def swap(pool, start, p0_mode, tag, max_passes=50):
    t = time.time()
    sm = lambda sets: score_many(pool, sets, p0_mode)
    sites, score, trace = H.swap(sm, start, config.CANDIDATE_SITES.tolist(),
                                 config.K, max_passes=max_passes)
    for (out, inn), sc in trace:
        print(f"    [{tag}] swap: {out} -> {inn}  F={sc:.6f}", flush=True)
    print(f"    [{tag}] swap done  F={score:.6f}  ({time.time()-t:.0f}s)", flush=True)
    return sites, score


def stingy(pool, p0_mode, tag):
    t = time.time()
    sm = lambda sets: score_many(pool, sets, p0_mode)
    sites, score, _ = H.stingy(sm, config.CANDIDATE_SITES.tolist(), config.K)
    print(f"    [{tag}] stingy  F={score:.6f}  ({time.time()-t:.0f}s)", flush=True)
    return sites, score


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