# scripts/size_sweep.py
#
# Reef-size BUDGET SWEEP experiment.
#
# Question: when the area budget binds, does the optimizer concentrate acreage on
# connectivity hubs, or is the allocation just an artifact of the (index-based)
# ceiling tiering in the canonical model?
#
# Method: use UNIFORM bounds (config.SIZE_SWEEP.L / .U for every site) so no site
# has a built-in acreage advantage, then sweep the total budget T across the
# binding band and record, per site, how many acres it receives at each T.
#
# For each (matrix, budget T):
#   1. regenerate oyster_quad.dat for the matrix (via prepare_data),
#   2. write a temporary uniform-bounds size .dat with that T,
#   3. solve +Size (or +Comm+Size) with Gurobi,
#   4. record the objective and per-site area.
#
# Outputs:
#   runs/size_sweep.csv          long format: matrix, model, T, site, selected, area, backbone
#   runs/size_sweep_summary.txt  human-readable grouped allocations per (matrix, T)
#   figures/size_sweep_matrix{1,2}.png   per-site acres vs T (if matplotlib present)
#
# The canonical ampl/oyster_size.dat is NEVER modified (this writes its own temp
# file), so run_everything.py still reproduces the published +Size numbers.
#
# Usage:
#   python -m scripts.size_sweep                       # both matrices, +Size, default budgets
#   python -m scripts.size_sweep --matrix 1
#   python -m scripts.size_sweep --model comm_size
#   python -m scripts.size_sweep --budgets 200 400 600 800 1000
#   python -m scripts.size_sweep --no-plot

import argparse
import csv
import json
from pathlib import Path

import config

TMP_SIZE_DAT = config.AMPL_DIR / "_sweep_size.dat"

MODELS = {
    # short   : (model file,            extra dats,            objective name)
    "size":      ("oyster_size.mod",      [],                    "Larvae"),
    "comm_size": ("oyster_comm_size.mod", ["oyster_comm.dat"],   "TotalLarvae"),
}


def backbone_set():
    """7-site global backbone from references.json (fallback to the known set)."""
    try:
        d = json.loads((config.ROOT / "references.json").read_text())
        return set(d["backbone"]["global"])
    except Exception:
        return {10, 31, 37, 40, 41, 49, 53}


def load_strength_ranks(matrix_id):
    """Parse runs/network_ranking_matrix{mid}_selfloops_on.txt (if present) into
    {site: {'out': rank, 'in': rank, 'self': value}}. Ranks are 1 = strongest.
    Returns {} if the file isn't there (annotation is then skipped)."""
    path = config.RUNS_DIR / f"network_ranking_matrix{matrix_id}_selfloops_on.txt"
    if not path.exists():
        return {}
    recs = {}
    in_table = False
    for ln in path.read_text().splitlines():
        s = ln.strip()
        if s.startswith("site_id") and "out_strength" in s:
            in_table = True; continue
        if in_table:
            if not s or s.startswith("Backbone") or ":" in s:
                break
            parts = s.replace(",", "").split()
            if len(parts) < 4 or not parts[0].isdigit():
                continue
            site = int(parts[0])
            # cols: site_id backbone out_strength in_strength self_recruitment ...
            recs[site] = {"out_val": float(parts[2]), "in_val": float(parts[3]),
                          "self": float(parts[4])}
    if not recs:
        return {}
    # convert values to ranks (1 = largest)
    def ranks(key):
        order = sorted(recs, key=lambda k: -recs[k][key])
        return {site: i + 1 for i, site in enumerate(order)}
    ro, ri = ranks("out_val"), ranks("in_val")
    return {site: {"out": ro[site], "in": ri[site], "self": recs[site]["self"]}
            for site in recs}


def write_uniform_size_dat(n, L, U, T, Sbar, path):
    """Write a size .dat with the SAME L and U for all n sites and budget T."""
    lines = [f"# TEMPORARY uniform-bounds size data (L={L}, U={U}, T={T}) for size_sweep.py",
             "", "param L :="]
    lines += [f"  {i} {L:.6f}" for i in range(n)]
    lines += [";", "", "param U :="]
    lines += [f"  {i} {U:.6f}" for i in range(n)]
    lines += [";", "", f"param TotReefSize := {T:.2f};", "", f"param Sbar := {Sbar:g};", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def prepare_quad(matrix_id):
    """Regenerate oyster_quad.dat + the mapping CSV for this matrix."""
    import subprocess, sys
    r = subprocess.run([sys.executable, "-m", "scripts.prepare_data", "--matrix", matrix_id],
                       cwd=str(config.ROOT), capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        raise RuntimeError("prepare_data failed")


def solve(matrix_id, model_short, T):
    """Solve the sizing model at budget T; return (objective, {site_label: acres})."""
    from amplpy import AMPL
    import pandas as pd

    model_file, extra_dats, objname = MODELS[model_short]
    labels = pd.read_csv(config.RUNS_DIR / f"oyster_index_mapping_matrix{matrix_id}.csv")["site_id"].tolist()
    n = len(labels)

    s = config.SIZE_SWEEP
    write_uniform_size_dat(n, s["L"], s["U"], T, s["Sbar"], TMP_SIZE_DAT)

    a = AMPL()
    a.eval("option solver gurobi;")
    a.eval(f"option gurobi_options '{config.GUROBI_OPTIONS}';")
    a.read(str(config.AMPL_DIR / model_file))
    a.readData(str(config.quad_dat(matrix_id)))
    for d in extra_dats:
        a.readData(str(config.AMPL_DIR / d))
    a.readData(str(TMP_SIZE_DAT))
    a.eval("solve;")

    xvals = {int(r[0]): float(r[1]) for r in a.getVariable("x").getValues().to_list()}
    svals = {int(r[0]): float(r[1]) for r in a.getVariable("s").getValues().to_list()}
    obj = float(a.getObjective(objname).value())
    area = {labels[i]: round(svals.get(i, 0.0), 3) for i in range(n) if xvals.get(i, 0) > 0.5}
    return obj, area


def main():
    ap = argparse.ArgumentParser(description="Reef-size budget sweep with uniform bounds.")
    ap.add_argument("--matrix", choices=["1", "2", "both"], default="both")
    ap.add_argument("--model", choices=list(MODELS), default="size")
    ap.add_argument("--budgets", type=float, nargs="+", default=None,
                    help="override config.SIZE_SWEEP['budgets']")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    budgets = args.budgets if args.budgets is not None else config.SIZE_SWEEP["budgets"]
    mats = ["1", "2"] if args.matrix == "both" else [args.matrix]
    bb = backbone_set()
    s = config.SIZE_SWEEP
    print(f"[sweep] model={args.model}  uniform L={s['L']} U={s['U']}  budgets={budgets}")
    print(f"[sweep] binding band is {s['L']*config.K:.0f} < T < {s['U']*config.K:.0f} "
          f"(K={config.K})")

    rows = []          # long-format records
    summary = []       # text blocks
    per_matrix = {}    # matrix -> {site: {T: area}} for plotting

    for mid in mats:
        prepare_quad(mid)
        per_matrix[mid] = {}
        for T in budgets:
            obj, area = solve(mid, args.model, T)
            slack = T / (config.K * s["U"])
            summary.append(f"\nMatrix {mid}  |  T={T:g}  (slack {slack:.2f})  |  "
                           f"objective={obj:.4f}  |  {len(area)} sites")
            from collections import defaultdict
            grp = defaultdict(list)
            for site, ac in area.items():
                grp[round(ac, 1)].append(site)
            for ac in sorted(grp, reverse=True):
                sites_lbl = ", ".join(f"{st}{'*' if st in bb else ''}" for st in sorted(grp[ac]))
                summary.append(f"   {ac:6.1f} ac : {sites_lbl}")
            for site, ac in area.items():
                rows.append({"matrix": mid, "model": args.model, "T": T,
                             "site": site, "selected": 1, "area": ac,
                             "backbone": int(site in bb)})
                per_matrix[mid].setdefault(site, {})[T] = ac
            print(f"  [M{mid}] T={T:g}  obj={obj:.4f}  "
                  f"maxed={sum(1 for v in area.values() if abs(v-s['U'])<1e-6)}  "
                  f"floored={sum(1 for v in area.values() if abs(v-s['L'])<1e-6)}  "
                  f"interior={sum(1 for v in area.values() if s['L']+1e-6<v<s['U']-1e-6)}")

    # ---- area-core diagnostic: selection stability + area-priority core ----
    U = s["U"]; eps = 1e-6
    diag = ["", "="*64, "AREA-CORE DIAGNOSTIC", "="*64]
    for mid in mats:
        site_map = per_matrix[mid]                     # {site: {T: area}}
        sel_by_T = {T: {st for st, tm in site_map.items() if T in tm} for T in budgets}
        set_constant = all(sel_by_T[T] == sel_by_T[budgets[0]] for T in budgets)
        sel_core = set.intersection(*sel_by_T.values()) if sel_by_T else set()
        # area-priority core: at (or within eps of) the ceiling U at EVERY budget
        area_core = sorted(st for st, tm in site_map.items()
                           if all(tm.get(T, 0) >= U - eps for T in budgets))
        # backbone sites floored (<= L+eps) at every budget
        floored_bb = sorted(st for st, tm in site_map.items()
                            if st in bb and all(tm.get(T, U) <= s["L"] + eps for T in budgets))
        ranks = load_strength_ranks(mid)

        diag.append(f"\nMatrix {mid}")
        diag.append(f"  selected set constant across budgets? {set_constant}  "
                    f"(|core selected| = {len(sel_core)} of 25)")
        diag.append(f"  AREA-PRIORITY CORE (>= {U:g} ac at every T): {area_core}")
        if ranks:
            for st in area_core:
                r = ranks.get(st, {})
                bbtag = "  [backbone]" if st in bb else ""
                diag.append(f"     site {st:>3}: out-strength rank {r.get('out','?'):>2}, "
                            f"in-strength rank {r.get('in','?'):>2}, "
                            f"self-recruit {r.get('self',0):>12,.0f}{bbtag}")
        diag.append(f"  BACKBONE sites floored (5 ac) at every T: {floored_bb}")
        if ranks:
            for st in floored_bb:
                r = ranks.get(st, {})
                diag.append(f"     site {st:>3}: out-strength rank {r.get('out','?'):>2}, "
                            f"in-strength rank {r.get('in','?'):>2}  "
                            f"(bridge/connector, weak source)")
        print(f"\n  [M{mid}] set constant across T: {set_constant} | "
              f"area-core {area_core} | floored backbone {floored_bb}")
    summary += diag

    # ---- write CSV ----
    config.RUNS_DIR.mkdir(exist_ok=True)
    csv_path = config.RUNS_DIR / "size_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["matrix", "model", "T", "site", "selected", "area", "backbone"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n[sweep] wrote {csv_path}  ({len(rows)} rows)")

    # ---- write summary ----
    head = [f"REEF-SIZE BUDGET SWEEP  (model={args.model}, uniform L={s['L']} U={s['U']})",
            "'*' marks a global-backbone site.", "="*64]
    (config.RUNS_DIR / "size_sweep_summary.txt").write_text("\n".join(head + summary) + "\n")
    print(f"[sweep] wrote {config.RUNS_DIR / 'size_sweep_summary.txt'}")

    # ---- optional plot: per-site acres vs T ----
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            config.FIG_DIR.mkdir(exist_ok=True)
            for mid, site_map in per_matrix.items():
                fig, ax = plt.subplots(figsize=(7, 5))
                for site, tmap in sorted(site_map.items()):
                    xs = sorted(tmap)
                    ys = [tmap[t] for t in xs]
                    is_bb = site in bb
                    ax.plot(xs, ys, marker="o", lw=2 if is_bb else 1,
                            color="crimson" if is_bb else "0.6",
                            alpha=0.95 if is_bb else 0.5, zorder=3 if is_bb else 1)
                    if is_bb:
                        ax.annotate(str(site), (xs[-1], ys[-1]), fontsize=8,
                                    color="crimson", xytext=(3, 0), textcoords="offset points")
                ax.set_xlabel("total area budget T (acres)")
                ax.set_ylabel("acres allocated to site")
                ax.set_title(f"Reef-size allocation vs budget — Matrix {mid}\n"
                             f"(red = 7-site backbone; uniform L={s['L']}, U={s['U']})")
                ax.grid(alpha=0.3)
                out = config.FIG_DIR / f"size_sweep_matrix{mid}.png"
                fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
                print(f"[sweep] wrote {out}")
        except ImportError:
            print("[sweep] matplotlib not available; skipping plot")

    # ---- cleanup temp dat ----
    if TMP_SIZE_DAT.exists():
        TMP_SIZE_DAT.unlink()


if __name__ == "__main__":
    main()