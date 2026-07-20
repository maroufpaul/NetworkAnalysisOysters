# src/opt/heuristics.py
#
# Single canonical implementation of the three ODE-driven search heuristics.
# Both scripts.run_heuristics (parallel, full 49-site problem) and
# scripts.calibrate_real (serial + cached, small induced subinstances) call
# these, so the search logic lives in exactly one place.
#
# Each function takes a `score_many(list_of_site_sets) -> list_of_floats`
# callable. The caller decides how sets are scored (parallel workers, a cached
# serial evaluator, etc.); the algorithm here is agnostic to that.

import numpy as np


def greedy(score_many, pool, K):
    """Start empty; repeatedly add the site with the largest marginal gain."""
    chosen, left = [], list(pool)
    best = 0.0
    trace = []
    for _ in range(K):
        scores = score_many([chosen + [c] for c in left])
        i = int(np.argmax(scores))                 # ties -> first
        chosen.append(left.pop(i)); best = scores[i]
        trace.append((chosen[-1], best))
    return sorted(chosen), best, trace


def swap(score_many, start, pool, K, max_passes=50, tol=1e-9):
    """Local search from `start`: best improving 1-for-1 exchange each pass."""
    cur = list(start)
    score = score_many([cur])[0]
    trace = []
    for _ in range(max_passes):
        outside = [s for s in pool if s not in cur]
        moves = [(o, i) for o in cur for i in outside]
        trials = [[i if s == o else s for s in cur] for o, i in moves]
        scores = score_many(trials)
        j = int(np.argmax(scores))
        if scores[j] <= score + tol:
            break
        cur = trials[j]; score = scores[j]
        trace.append((moves[j], score))
    return sorted(cur), score, trace


def stingy(score_many, pool, K):
    """Start with the whole pool; repeatedly delete the least-missed site."""
    S = list(pool)
    score = score_many([S])[0]
    trace = []
    while len(S) > K:
        scores = score_many([[s for s in S if s != d] for d in S])
        i = int(np.argmax(scores))                 # whichever we miss least
        trace.append((S[i], scores[i]))
        S.pop(i); score = scores[i]
    return sorted(S), score, trace