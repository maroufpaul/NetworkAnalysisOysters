"""
Realistic-supply MIQP experiments with JARS density feedback.

Default output is intentionally compact. Detailed site sets and full run data
are always written to CSV/TXT. Use --trace to print round-by-round JARS scores
and --show-sites to print final site sets in the console.

# Clean  output
python -m scripts.run_iterated

# Show each iteration's JARS score
python -m scripts.run_iterated --trace

# Show final selected site sets
python -m scripts.run_iterated --show-sites

# Only run the main network-aware experiment
python -m scripts.run_iterated --exp iterated --fallback network
"""

import argparse
import time
from typing import Any

import numpy as np
import pandas as pd
from amplpy import AMPL, OutputHandler
from scipy.integrate import solve_ivp

import config
from src.model.jars_ode import load_connectivity, odesys, setP0
from src.opt.evaluator import evaluate_subset


TMP_DAT = config.AMPL_DIR / "oyster_iter.dat"


class QuietAMPL(OutputHandler):
    """Discard routine AMPL and solver chatter; errors still raise normally."""

    def output(self, kind, msg):  # noqa: D401, ARG002
        pass


def load(matrix_id: str):
    """Load one connectivity matrix and retain the 49 candidate reefs."""
    conn, key = load_connectivity(config.MATRICES[config.matrix_key(matrix_id)])
    keep = ~np.isin(key, config.UNWANTED)
    labels = key[keep]
    P1 = conn[np.ix_(keep, keep)] * config.P1SCALING
    Pe = setP0(labels).astype(float)
    return conn, key, labels, P1, Pe


def F(sites, conn, key, want_densities: bool = False):
    """Evaluate a selected set with the realistic-supply JARS model."""
    return evaluate_subset(
        list(sites),
        conn,
        key,
        P0_mode="realistic",
        return_densities=want_densities,
    )


def one_reef(P1_ii: float, Pe_i: float) -> float:
    """Return one reef's equilibrium adult density under fixed larval input."""
    v0 = np.array(
        [config.IC["J"], config.IC["A"], config.IC["R"], config.IC["S"]],
        dtype=float,
    )

    sol = solve_ivp(
        lambda t, v: odesys(
            t,
            v,
            np.array([Pe_i]),
            np.array([[P1_ii]]),
            config.MU * np.ones(1),
        ),
        [0, config.TMAX],
        v0,
        method="RK45",
        rtol=1e-6,
    )

    if not sol.success:
        raise RuntimeError(f"Single-reef JARS solve failed: {sol.message}")

    # Clip tiny negative numerical noise before applying A**1.72.
    return max(float(sol.y[1, -1]), 0.0)


def alone_densities(P1: np.ndarray, Pe: np.ndarray) -> np.ndarray:
    """Compute each candidate's equilibrium when restored alone."""
    return np.array(
        [one_reef(P1[i, i], Pe[i]) for i in range(len(Pe))],
        dtype=float,
    )


def network_fallback_densities(
    labels: np.ndarray,
    P1: np.ndarray,
    Pe: np.ndarray,
    picked: list[int],
    densities: dict[int, float],
) -> tuple[np.ndarray, int]:
    """Estimate outsiders with inflow from the current selected network."""
    pos = {int(label): i for i, label in enumerate(labels)}
    picked_set = {int(site) for site in picked}
    picked_idx = np.array([pos[int(site)] for site in picked], dtype=int)

    # Current selected reefs are fixed sources during this one-step lookahead.
    # (evaluate_subset already clips, so these are >= 0.)
    source_A = np.array([densities[int(site)] for site in picked], dtype=float)
    # This gives incoming larvae to every candidate from the selected network.
    # Rows are sources, columns are destinations, so the transpose lines everything up.
    incoming = P1[picked_idx, :].T @ (source_A ** config.ALPHA)

    A_next = np.zeros(len(labels), dtype=float)

    for site, density in densities.items():
        A_next[pos[int(site)]] = max(float(density), 0.0)

    outsider_runs = 0
    for i, label in enumerate(labels):
        if int(label) in picked_set:
            continue

        # Only the outsider is integrated; the 25 selected sources stay fixed.
        A_next[i] = one_reef(P1[i, i], Pe[i] + incoming[i])
        outsider_runs += 1

    return A_next, outsider_runs


def solve_miqp(
    matrix_id: str,
    labels: np.ndarray,
    P1: np.ndarray,
    Pe: np.ndarray,
    A: np.ndarray,
) -> list[int]:
    """Solve the source-specific MIQP for the current density estimates."""
    # One source density scales that source's entire outgoing row.
    # The clip is the last line of defense: A ** 1.72 is NaN for any negative A,
    # and a NaN here would sail into the .dat file and violate `param W >= 0`.
    # Everything feeding A is already non-negative, so this should never fire.
    W = P1 * (np.maximum(A, 0.0) ** config.ALPHA)[:, None]
    n = len(labels)

    lines = ["# written by run_iterated.py", "", "set N :="]
    lines += [f"  {i}" for i in range(n)]
    lines += [";", "", f"param K := {config.K};", "", "param Pe :="]
    lines += [f"  {i} {Pe[i]:.6f}" for i in range(n)]
    lines += [";", "", "param W :="]
    lines += [
        f"  [{i}, {j}] {W[i, j]:.6f}"
        for i in range(n)
        for j in range(n)
        if W[i, j] != 0
    ]
    lines += [";", ""]
    TMP_DAT.write_text("\n".join(lines), encoding="utf-8")

    ampl = AMPL()
    ampl.set_output_handler(QuietAMPL())

    try:
        ampl.eval("option solver gurobi;")

        solver_options = f"{config.GUROBI_OPTIONS} outlev=0"
        ampl.eval(f"option gurobi_options '{solver_options}';")
        ampl.eval("option solver_msg 0;")

        ampl.read(str(config.AMPL_DIR / "oyster_quad.mod"))
        ampl.readData(str(TMP_DAT))
        ampl.eval("solve;")

        solve_result = str(ampl.getValue("solve_result"))
        if solve_result != "solved":
            raise RuntimeError(f"MIQP solve did not finish normally: {solve_result}")

        site_labels = pd.read_csv(config.mapping_csv(matrix_id))["site_id"].tolist()
        picked = sorted(
            site_labels[int(row[0])]
            for row in ampl.getVariable("x").getValues().to_list()
            if float(row[1]) > 0.5
        )
    finally:
        ampl.close()

    if len(picked) != config.K:
        raise RuntimeError(f"Expected {config.K} selected reefs, found {len(picked)}")

    return picked


def iterate(
    matrix_id: str,
    A0: np.ndarray,
    fallback: str,
    alone: np.ndarray,
    labels: np.ndarray,
    P1: np.ndarray,
    Pe: np.ndarray,
    conn: np.ndarray,
    key: np.ndarray,
    trace: bool = False,
) -> dict[str, Any]:
    """Alternate MIQP and JARS updates until a set repeats."""
    # We need this lookup later when the JARS densities come back keyed by real site label.
    pos = {int(label): i for i, label in enumerate(labels)}
    A = np.array(A0, dtype=float)

    # history tracks the selected sets so we can spot a fixed point or a cycle.
    history: list[tuple[int, ...]] = []
    score_by_set: dict[tuple[int, ...], float] = {}
    selected_ode_runs = 0
    outsider_ode_runs = 0

    for step in range(config.ITER["max_passes"]):
        picked = solve_miqp(matrix_id, labels, P1, Pe, A)
        picked_key = tuple(picked)

        if picked_key in history:
            first = history.index(picked_key)

            if picked_key == history[-1]:
                # Same set twice in a row: observed selection fixed point.
                final_key = picked_key
                status = "fixed_point"
                cycle_length = 1
            else:
                # Older set returned: keep the best ODE-scored set in the cycle.
                cycle_sets = history[first:]
                final_key = max(cycle_sets, key=lambda sites: score_by_set[sites])
                status = "cycle"
                cycle_length = len(cycle_sets)

            return {
                "sites": list(final_key),
                "updates": step,
                "F": score_by_set[final_key],
                "status": status,
                "cycle_length": cycle_length,
                "selected_ode_runs": selected_ode_runs,
                "outsider_ode_runs": outsider_ode_runs,
            }

        history.append(picked_key)

        # One selected-network JARS run gives the score and all 25 densities.
        score, densities = F(picked, conn, key, want_densities=True)
        score_by_set[picked_key] = score
        selected_ode_runs += 1

        if trace:
            print(f"      round {step}: F={score:.6f}")

        if fallback == "sticky":
            A_next = A.copy()

        elif fallback == "isolated":
            A_next = alone.copy()

        elif fallback == "network":
            A_next, runs = network_fallback_densities(
                labels, P1, Pe, picked, densities
            )
            outsider_ode_runs += runs

        else:
            raise ValueError(f"Unknown fallback rule: {fallback}")

        # Sticky and isolated only set the outsider values above, so we still plug in the real selected densities here.
        if fallback != "network":
            for site, density in densities.items():
                A_next[pos[int(site)]] = max(float(density), 0.0)

        A = A_next

    # If no set repeats, return the best JARS-scored set encountered.
    best_key = max(history, key=lambda sites: score_by_set[sites])
    return {
        "sites": list(best_key),
        "updates": config.ITER["max_passes"],
        "F": score_by_set[best_key],
        "status": "max_passes",
        "cycle_length": 0,
        "selected_ode_runs": selected_ode_runs,
        "outsider_ode_runs": outsider_ode_runs,
    }


def exp_astar(mats: list[str]) -> None:
    """Measure sensitivity to one shared A_STAR."""
    print("\nFROZEN A_STAR SENSITIVITY")
    print("-" * 72)
    rows = []

    for matrix_id in mats:
        conn, key, labels, P1, Pe = load(matrix_id)
        best = F(config.BEST_HEUR_REAL[f"M{matrix_id}"], conn, key)
        print(f"M{matrix_id} benchmark F={best:.6f}")
        print(f"  {'A_STAR':>8}  {'F':>10}  {'% best':>8}")

        # designs tells us how many genuinely different site sets the sweep produced.
        designs = set()
        matrix_rows = []

        for a_star in config.A_STAR_SWEEP:
            picked = solve_miqp(
                matrix_id,
                labels,
                P1,
                Pe,
                np.full(len(labels), a_star),
            )
            score = F(picked, conn, key)
            pct = 100 * score / best
            designs.add(tuple(picked))

            row = {
                "matrix": f"M{matrix_id}",
                "A_star": a_star,
                "F": round(score, 6),
                "pct_of_best": round(pct, 2),
                "sites": " ".join(map(str, picked)),
            }
            rows.append(row)
            matrix_rows.append(row)
            print(f"  {a_star:8g}  {score:10.6f}  {pct:7.1f}%")

        pcts = [row["pct_of_best"] for row in matrix_rows]
        print(
            f"  result: {len(designs)} designs, "
            f"range {min(pcts):.1f}-{max(pcts):.1f}%\n"
        )

    config.RUNS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(config.RUNS_DIR / "astar_sweep.csv", index=False)


def exp_iterated(
    mats: list[str],
    rules: list[str],
    trace: bool,
    show_sites: bool,
) -> None:
    """Run all requested density-feedback experiments."""
    print("ITERATED SOURCE-DENSITY SURROGATE")
    print("-" * 72)
    blocks = []
    rows = []

    for matrix_id in mats:
        conn, key, labels, P1, Pe = load(matrix_id)

        # Precompute these once. No reason to rerun the same 49 isolated ODEs for every start.
        alone = alone_densities(P1, Pe)
        best = F(config.BEST_HEUR_REAL[f"M{matrix_id}"], conn, key)
        # A reef counts as "dead alone" if its isolated equilibrium is basically zero.
        dead = int((alone <= 1e-6).sum())

        print(
            f"M{matrix_id}: benchmark F={best:.6f}; "
            f"{dead}/{len(labels)} reefs fail alone"
        )
        print(
            f"  {'fallback':<9} {'start':>8} {'upd':>4} "
            f"{'status':<12} {'F':>10} {'% best':>8}"
        )

        for rule in rules:
            for start in config.ITER["starts"]:
                # None means use the isolated-density vector; numbers mean give every site that same starting A.
                name = "alone" if start is None else str(start)
                A0 = alone.copy() if start is None else np.full(len(labels), float(start))

                result = iterate(
                    matrix_id,
                    A0,
                    rule,
                    alone,
                    labels,
                    P1,
                    Pe,
                    conn,
                    key,
                    trace=trace,
                )

                pct = 100 * result["F"] / best
                print(
                    f"  {rule:<9} {name:>8} {result['updates']:4d} "
                    f"{result['status']:<12} {result['F']:10.6f} {pct:7.1f}%"
                )

                if show_sites:
                    print(f"      sites: {result['sites']}")

                row = {
                    "matrix": f"M{matrix_id}",
                    "fallback": rule,
                    "start": name,
                    "updates": result["updates"],
                    "status": result["status"],
                    "cycle_length": result["cycle_length"],
                    "selected_ode_runs": result["selected_ode_runs"],
                    "outsider_ode_runs": result["outsider_ode_runs"],
                    "F": round(result["F"], 6),
                    "pct_of_best": round(pct, 2),
                    "sites": " ".join(map(str, result["sites"])),
                }
                rows.append(row)

                blocks.append(
                    "\n".join(
                        [
                            "=" * 72,
                            f"M{matrix_id} | fallback={rule} | start={name}",
                            "=" * 72,
                            f"status            : {result['status']}",
                            f"density updates   : {result['updates']}",
                            f"selected ODE runs : {result['selected_ode_runs']}",
                            f"outsider ODE runs : {result['outsider_ode_runs']}",
                            f"F                 : {result['F']:.6f}",
                            f"percent of best   : {pct:.2f}",
                            f"sites             : {result['sites']}",
                            "",
                        ]
                    )
                )

        print()

    config.RUNS_DIR.mkdir(exist_ok=True)
    (config.RUNS_DIR / "iterated.txt").write_text(
        "\n".join(blocks), encoding="utf-8"
    )
    df = pd.DataFrame(rows)
    df.to_csv(config.RUNS_DIR / "iterated.csv", index=False)

    print("SUMMARY BY MATRIX AND FALLBACK")
    print("-" * 72)
    for matrix_name in sorted(df["matrix"].unique()):
        for rule in rules:
            sub = df[(df.matrix == matrix_name) & (df.fallback == rule)]
            if sub.empty:
                continue

            fixed = int((sub.status == "fixed_point").sum())
            print(
                f"{matrix_name} {rule:<9}: "
                f"{sub.pct_of_best.min():.1f}-{sub.pct_of_best.max():.1f}% | "
                f"{sub.sites.nunique()} design(s) | "
                f"updates {sub.updates.min()}-{sub.updates.max()} | "
                f"fixed {fixed}/{len(sub)}"
            )

    print("\nDetailed results: runs/astar_sweep.csv, runs/iterated.csv, runs/iterated.txt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=["astar", "iterated", "all"], default="all")
    parser.add_argument("--matrix", choices=["1", "2", "both"], default="both")
    parser.add_argument(
        "--fallback",
        choices=config.ITER["rules"] + ["all"],
        default="all",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="print each round's JARS score",
    )
    parser.add_argument(
        "--show-sites",
        action="store_true",
        help="print final selected site sets",
    )
    args = parser.parse_args()

    mats = ["1", "2"] if args.matrix == "both" else [args.matrix]
    rules = config.ITER["rules"] if args.fallback == "all" else [args.fallback]

    # prepare_data creates the mapping files. Better to fail here than halfway through a long run.
    for matrix_id in mats:
        if not config.mapping_csv(matrix_id).exists():
            raise SystemExit("run: python -m scripts.prepare_data")

    config.RUNS_DIR.mkdir(exist_ok=True)
    started = time.time()

    if args.exp in ("astar", "all"):
        exp_astar(mats)
    if args.exp in ("iterated", "all"):
        exp_iterated(mats, rules, trace=args.trace, show_sites=args.show_sites)

    print(f"\nCompleted in {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()