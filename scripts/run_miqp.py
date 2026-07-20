"""
MIQP with constant external supply (Pe = 170 everywhere).

Constant Pe means every site gets the same larval subsidy from outside, so any
difference between designs comes purely from the larval network. That's the
whole point of this experiment. (Realistic Pe lives in run_iterated.py, because
it needs the ODE feedback loop to work properly.)

    python -m scripts.run_miqp                    # 4 models x 2 matrices
    python -m scripts.run_miqp --model size
    python -m scripts.run_miqp --matrix 1
    python -m scripts.run_miqp --model comm+size --matrix 2

Models:
    base       pick 25 sites, nothing else
    comm       + at least rmin sites from each of the 5 regions
    size       + how many acres each reef gets (1000 total to spend)
    comm+size  + both

Writes runs/miqp_<model>.txt and .csv
Run prepare_data first.
"""
import argparse
import time
from collections import defaultdict

import pandas as pd
from amplpy import AMPL

import config

P0 = "constant"


def solve(model, matrix_id):
    """Give AMPL the model + data, get back which sites it picked."""
    model_file, extra_dats, objname, has_size = config.MIQP_MODELS[model]

    ampl = AMPL()
    ampl.eval("option solver gurobi;")
    ampl.eval(f"option gurobi_options '{config.GUROBI_OPTIONS}';")
    ampl.eval("option solver_msg 0;")
    ampl.read(str(config.AMPL_DIR / model_file))
    ampl.readData(str(config.quad_dat(matrix_id, P0)))
    for d in extra_dats:                          # comm and/or size data
        ampl.readData(str(getattr(config, d)))
    ampl.eval("solve;")

    if ampl.getValue("solve_result") != "solved":
        raise RuntimeError(f"{model} M{matrix_id}: AMPL said "
                           f"'{ampl.getValue('solve_result')}'")

    # AMPL thinks in indices 0..48; we want real site labels back
    labels = pd.read_csv(config.mapping_csv(matrix_id))["site_id"].tolist()
    picked = [int(r[0]) for r in ampl.getVariable("x").getValues().to_list()
              if float(r[1]) > 0.5]

    sizes = {}
    if has_size:
        acres = {int(r[0]): float(r[1])
                 for r in ampl.getVariable("s").getValues().to_list()}
        sizes = {labels[i]: round(acres.get(i, 0.0), 2) for i in picked}

    obj = float(ampl.getObjective(objname).value())
    ampl.close()
    return {"sites": sorted(labels[i] for i in picked), "obj": obj, "sizes": sizes}


def load_communities():
    """{community number: [site labels]} straight from the xlsx."""
    raw = pd.read_excel(config.COMMUNITIES_XLSX, header=None).values
    out = {}
    for r in range(1, raw.shape[0]):
        if not pd.isna(raw[r, 0]):
            out[int(raw[r, 0])] = sorted(int(v) for v in raw[r, 2:] if not pd.isna(v))
    return out


def report(model, matrix_id, r, secs, comms):
    L = ["=" * 70,
         f"M{matrix_id}  |  {model}  |  Pe = {config.CONST_P0:g} (constant)",
         "=" * 70,
         f"  objective : {r['obj']:.4f}",
         f"  time      : {secs:.2f}s",
         "",
         f"  SITES ({len(r['sites'])} of {len(config.CANDIDATE_SITES)})",
         f"    {r['sites']}",
         ""]

    # only show the community table if the model actually has that constraint
    if "comm" in model:
        L += ["  COMMUNITIES", f"    {'region':<21} {'got':>4} {'min':>4}  sites"]
        for c in sorted(comms):
            got = sorted(set(r["sites"]) & set(comms[c]))
            rmin = config.COMMUNITY_MINS[c]
            tight = "  <- stuck at the minimum" if len(got) == rmin else ""
            L.append(f"    C{c} {config.COMM_NAMES[c]:<18} {len(got):>4} {rmin:>4}  {got}{tight}")
        L.append("")

    if r["sizes"]:
        L += ["  ACRES",
              f"    {sum(r['sizes'].values()):.0f} of {config.SIZE['TotReefSize']:.0f} spent"]
        grp = defaultdict(list)
        for site, a in r["sizes"].items():
            grp[a].append(site)
        for a in sorted(grp, reverse=True):
            L.append(f"    {a:>6.1f} ac x{len(grp[a]):<2} : {sorted(grp[a])}")
        L += [""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(config.MIQP_MODELS) + ["all"], default="all")
    ap.add_argument("--matrix", choices=["1", "2", "both"], default="both")
    args = ap.parse_args()

    models = list(config.MIQP_MODELS) if args.model == "all" else [args.model]
    mats = ["1", "2"] if args.matrix == "both" else [args.matrix]

    for m in mats:
        if not config.quad_dat(m, P0).exists():
            raise SystemExit(f"missing {config.quad_dat(m, P0).name} -- run:\n"
                             f"    python -m scripts.prepare_data")

    comms = load_communities() if any("comm" in m for m in models) else {}

    blocks, rows = [], []
    for m in mats:
        for model in models:
            t = time.time()
            r = solve(model, m)
            secs = time.time() - t
            block = report(model, m, r, secs, comms)
            print(block)
            blocks.append(block)
            rows.append({"matrix": f"M{m}", "model": model,
                         "objective": round(r["obj"], 4),
                         "acres": round(sum(r["sizes"].values()), 1) if r["sizes"] else "",
                         "seconds": round(secs, 2),
                         "sites": " ".join(map(str, r["sites"])),
                         "per_site_acres": " ".join(f"{s}={a:g}" for s, a in r["sizes"].items())})

    config.RUNS_DIR.mkdir(exist_ok=True)
    tag = "all" if args.model == "all" else args.model.replace("+", "_")
    (config.RUNS_DIR / f"miqp_{tag}.txt").write_text("\n".join(blocks), encoding="utf-8")
    pd.DataFrame(rows).to_csv(config.RUNS_DIR / f"miqp_{tag}.csv", index=False)
    print(f"{len(rows)} solves -> runs/miqp_{tag}.txt, runs/miqp_{tag}.csv")


if __name__ == "__main__":
    main()