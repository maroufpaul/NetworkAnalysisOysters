"""
Reef-size budget sweep.

The question: when you can't afford to max out every reef, where does the
optimizer spend the acres?

The canonical size model can't answer that honestly, because its ceiling is
uniform now, but this script exists to sweep the budget under uniform bounds (first 26 used to get 50
acres, the rest get 40). So "site X got 50 and site Y got 40" tells you nothing
about X and Y. Here every site gets the SAME bounds, and we sweep the budget
instead. Now if acreage piles up somewhere, that's the network talking.

The budget has to actually bind or there's nothing to see: below L*K=125 acres
everything is pinned to the floor, above U*K=1250 everything can max out. So we
sweep in between.

    python -m scripts.size_sweep                        # both matrices
    python -m scripts.size_sweep --matrix 1
    python -m scripts.size_sweep --model comm+size      # with the equity constraint
    python -m scripts.size_sweep --budgets 300 500 750 1000
    python -m scripts.size_sweep --no-plot

Writes runs/size_sweep.csv, runs/size_sweep_summary.txt,
figures/size_sweep_matrix{1,2}.png
"""
import argparse
import csv
from collections import defaultdict

import numpy as np
import pandas as pd
from amplpy import AMPL

import config
from src.model.jars_ode import load_connectivity


def load_communities():
    """{community number: [site labels]} from the xlsx."""
    raw = pd.read_excel(config.COMMUNITIES_XLSX, header=None).values
    out = {}
    for r in range(1, raw.shape[0]):
        if not pd.isna(raw[r, 0]):
            out[int(raw[r, 0])] = sorted(int(v) for v in raw[r, 2:] if not pd.isna(v))
    return out

TMP_DAT = config.AMPL_DIR / "_sweep_size.dat"   # our own bounds; the real
                                                # ampl/oyster_size.dat is left alone


def strengths(matrix_id):
    """How much each reef sends out and takes in, straight from the connectivity
    matrix. Ignores the diagonal -- a reef feeding itself isn't 'sending' anywhere.
    Returns {site: (out_rank, in_rank, self_recruitment)}, rank 1 = biggest."""
    conn, key = load_connectivity(config.MATRICES[config.matrix_key(matrix_id)])
    keep = ~np.isin(key, config.UNWANTED)
    labels, P = key[keep], conn[np.ix_(keep, keep)] * config.P1SCALING

    diag = np.diag(P).copy()
    off = P.copy()
    np.fill_diagonal(off, 0.0)
    out, inn = off.sum(axis=1), off.sum(axis=0)

    # rank 1 = biggest
    out_rank = {labels[i]: r + 1 for r, i in enumerate(np.argsort(-out))}
    in_rank = {labels[i]: r + 1 for r, i in enumerate(np.argsort(-inn))}
    return {int(labels[i]): (out_rank[labels[i]], in_rank[labels[i]], diag[i])
            for i in range(len(labels))}


def write_size_dat(n, L, U, T):
    """Same L and U for every site, budget T. That's the whole trick."""
    lines = [f"# temp file from size_sweep.py -- uniform L={L}, U={U}, T={T}", "",
             "param L :="]
    lines += [f"  {i} {L:.6f}" for i in range(n)]
    lines += [";", "", "param U :="]
    lines += [f"  {i} {U:.6f}" for i in range(n)]
    lines += [";", "", f"param TotReefSize := {T:.2f};", "",
              f"param Sbar := {config.SIZE_SWEEP['Sbar']:g};", ""]
    TMP_DAT.write_text("\n".join(lines), encoding="utf-8")


def solve(matrix_id, model, T):
    """Solve the sizing model at budget T. Returns (objective, {site: acres})."""
    model_file, extra_dats, objname, _ = config.MIQP_MODELS[model]
    labels = pd.read_csv(config.mapping_csv(matrix_id))["site_id"].tolist()
    write_size_dat(len(labels), config.SIZE_SWEEP["L"], config.SIZE_SWEEP["U"], T)

    a = AMPL()
    a.eval("option solver gurobi;")
    # outlev=0 stops the driver echoing "nonconvex=2 mipgap=1e-9" and its banner
    # on every single solve; solver_msg 0 kills the rest of AMPL's chatter.
    a.eval(f"option gurobi_options '{config.GUROBI_OPTIONS} outlev=0';")
    a.eval("option solver_msg 0;")
    a.read(str(config.AMPL_DIR / model_file))
    a.readData(str(config.quad_dat(matrix_id, "constant")))
    if "COMM_DAT" in extra_dats:
        a.readData(str(config.COMM_DAT))
    a.readData(str(TMP_DAT))          # our uniform bounds, NOT config.SIZE_DAT
    a.eval("solve;")

    if a.getValue("solve_result") != "solved":
        raise RuntimeError(f"M{matrix_id} T={T}: AMPL said '{a.getValue('solve_result')}'")

    x = {int(r[0]): float(r[1]) for r in a.getVariable("x").getValues().to_list()}
    s = {int(r[0]): float(r[1]) for r in a.getVariable("s").getValues().to_list()}
    obj = float(a.getObjective(objname).value())
    a.close()

    acres = {labels[i]: round(s.get(i, 0.0), 2)
             for i in range(len(labels)) if x.get(i, 0) > 0.5}
    return obj, acres


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", choices=["1", "2", "both"], default="both")
    ap.add_argument("--model", choices=["size", "comm+size"], default="size")
    ap.add_argument("--budgets", type=float, nargs="+", default=None)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    L, U = config.SIZE_SWEEP["L"], config.SIZE_SWEEP["U"]
    budgets = args.budgets or config.SIZE_SWEEP["budgets"]
    mats = ["1", "2"] if args.matrix == "both" else [args.matrix]
    bb = set(config.BACKBONE)
    # only needed when the equity constraint is actually on
    comms = load_communities() if "comm" in args.model else {}

    for m in mats:
        if not config.quad_dat(m, "constant").exists():
            raise SystemExit("run: python -m scripts.prepare_data")

    print(f"model={args.model}  uniform bounds L={L:g} U={U:g}  budgets={budgets}")
    print(f"budget only bites between {L*config.K:.0f} and {U*config.K:.0f} acres "
          f"(K={config.K})\n")

    rows, out_lines = [], []
    per_matrix = {}                    # matrix -> {site: {T: acres}}

    for m in mats:
        rank = strengths(m)
        per_matrix[m] = {}
        for T in budgets:
            obj, acres = solve(m, args.model, T)
            maxed = sum(1 for v in acres.values() if v >= U - 1e-6)
            floored = sum(1 for v in acres.values() if v <= L + 1e-6)
            print(f"  M{m} T={T:<6g} obj={obj:10.2f}  {maxed} maxed, {floored} floored, "
                  f"{len(acres)-maxed-floored} in between")

            out_lines.append(f"\nMatrix {m}  |  T={T:g}  |  objective={obj:.4f}  "
                             f"|  {len(acres)} sites")
            grp = defaultdict(list)
            for site, a in acres.items():
                grp[a].append(site)
            for a in sorted(grp, reverse=True):
                tagged = ", ".join(f"{s}{'*' if s in bb else ''}" for s in sorted(grp[a]))
                out_lines.append(f"   {a:6.1f} ac : {tagged}")

            # with comm+size you also want to know WHERE the acres landed, since
            # the whole point of the constraint is to spread them around
            if comms:
                out_lines.append("   by community:")
                for c in sorted(comms):
                    here = sorted(set(acres) & set(comms[c]))
                    ac_here = sum(acres[s] for s in here)
                    rmin = config.COMMUNITY_MINS[c]
                    tight = " (at the minimum)" if len(here) == rmin else ""
                    out_lines.append(
                        f"     C{c} {config.COMM_NAMES[c]:<18} "
                        f"{len(here)}/{rmin} sites{tight:<18} "
                        f"{ac_here:6.1f} ac ({100*ac_here/T:4.1f}% of budget)")
                    out_lines.append("        " + ", ".join(
                        f"{s}{'*' if s in bb else ''}={acres[s]:g}ac" for s in here))

            for site, a in acres.items():
                rows.append({"matrix": f"M{m}", "model": args.model, "T": T,
                             "site": site, "acres": a, "backbone": int(site in bb)})
                per_matrix[m].setdefault(site, {})[T] = a

    # ---- who actually got the money? ----
    out_lines += ["", "=" * 64, "WHERE THE ACRES WENT", "=" * 64]
    for m in mats:
        rank = strengths(m)
        smap = per_matrix[m]

        # picked at every budget, and maxed out at every budget
        always_max = sorted(s for s, t in smap.items()
                            if all(t.get(T, 0) >= U - 1e-6 for T in budgets))
        # backbone sites that never get more than the minimum
        always_floor = sorted(s for s, t in smap.items()
                              if s in bb and all(t.get(T, U) <= L + 1e-6 for T in budgets))
        picked_every_T = set.intersection(*[{s for s, t in smap.items() if T in t}
                                            for T in budgets])

        out_lines.append(f"\nMatrix {m}")
        out_lines.append(f"  picked at every budget: {len(picked_every_T)} of {config.K}")
        out_lines.append(f"  MAXED OUT at every budget: {always_max}")
        for s in always_max:
            o, i, self_r = rank[s]
            out_lines.append(f"     site {s:>3}: out-strength rank {o:>2}, in-strength rank "
                             f"{i:>2}, self-recruitment {self_r:>12,.0f}"
                             f"{'  [backbone]' if s in bb else ''}")
        out_lines.append(f"  BACKBONE sites stuck at the {L:g} ac floor every time: {always_floor}")
        for s in always_floor:
            o, i, _ = rank[s]
            out_lines.append(f"     site {s:>3}: out-strength rank {o:>2}, in-strength rank {i:>2}"
                             f"  -- it's a receiver, not a source. Acres buy outgoing")
            out_lines.append(f"              larvae, and this site doesn't send many.")
        print(f"\n  M{m}: always maxed {always_max} | backbone always floored {always_floor}")

    # ---- save ----
    config.RUNS_DIR.mkdir(exist_ok=True)
    with open(config.RUNS_DIR / "size_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["matrix", "model", "T", "site", "acres", "backbone"])
        w.writeheader()
        w.writerows(rows)
    head = [f"REEF-SIZE BUDGET SWEEP (model={args.model}, uniform L={L:g} U={U:g})",
            "* = one of the 7 backbone sites", "=" * 64]
    (config.RUNS_DIR / "size_sweep_summary.txt").write_text(
        "\n".join(head + out_lines) + "\n", encoding="utf-8")
    print(f"\n-> runs/size_sweep.csv, runs/size_sweep_summary.txt")

    # ---- plot ----
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            config.FIG_DIR.mkdir(exist_ok=True)
            for m, smap in per_matrix.items():
                fig, ax = plt.subplots(figsize=(7, 5))
                for site, tmap in sorted(smap.items()):
                    xs = sorted(tmap)
                    ys = [tmap[t] for t in xs]
                    on_bb = site in bb
                    ax.plot(xs, ys, marker="o", lw=2 if on_bb else 1,
                            color="crimson" if on_bb else "0.6",
                            alpha=0.95 if on_bb else 0.5, zorder=3 if on_bb else 1)
                    if on_bb:
                        ax.annotate(str(site), (xs[-1], ys[-1]), fontsize=8, color="crimson",
                                    xytext=(3, 0), textcoords="offset points")
                ax.set_xlabel("total area budget T (acres)")
                ax.set_ylabel("acres given to the site")
                ax.set_title(f"Where the acres go as the budget grows -- Matrix {m}\n"
                             f"(red = backbone; uniform L={L:g}, U={U:g})")
                ax.grid(alpha=0.3)
                fig.tight_layout()
                fig.savefig(config.FIG_DIR / f"size_sweep_matrix{m}.png", dpi=150)
                plt.close(fig)
                print(f"-> figures/size_sweep_matrix{m}.png")
        except ImportError:
            print("(no matplotlib, skipping the plot)")

    TMP_DAT.unlink(missing_ok=True)


if __name__ == "__main__":
    main()