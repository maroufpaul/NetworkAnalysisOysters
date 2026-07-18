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


# This script only handles the constant external-supply experiment.
# config.quad_dat() uses this string to choose the matching prepared data file.
P0 = "constant"


def solve(model, matrix_id):
    """Give AMPL the model + data, get back which sites it picked."""

    # Each model name points to:
    #   1. the AMPL .mod file,
    #   2. any extra .dat files it needs,
    #   3. the AMPL objective name,
    #   4. whether the model includes the reef-size variable s.
    #
    # For example, the community model needs the community data file,
    # while the size model needs the reef-size bounds and budget.
    model_file, extra_dats, objname, has_size = config.MIQP_MODELS[model]

    # Start a fresh AMPL session for this one model/matrix combination.
    # Using a fresh session keeps data from one solve from carrying into another.
    ampl = AMPL()

    # Tell AMPL to use Gurobi and apply the common solver settings from config.py.
    ampl.eval("option solver gurobi;")
    ampl.eval(f"option gurobi_options '{config.GUROBI_OPTIONS}';")

    # just hide the solver messages
    ampl.eval("option solver_msg 0;")

    # load the mathematical model.
    ampl.read(str(config.AMPL_DIR / model_file))

    # Then load the matrix-specific  data.
    # This file contains the candidate set, Pe values, connectivity weights, etc.
    ampl.readData(str(config.quad_dat(matrix_id, P0)))

    # Community and size variants need one or more additional data files.
    # getattr(config, d) turns a config variable name into its actual file path.
    for d in extra_dats:                          # comm and/or size data
        ampl.readData(str(getattr(config, d)))

    # At this point AMPL has the model and all required data, so solve it.
    ampl.eval("solve;")

    # Do not silently continue if Gurobi or AMPL failed.
    # A failed solve could otherwise produce an empty or misleading site list.
    if ampl.getValue("solve_result") != "solved":
        raise RuntimeError(f"{model} M{matrix_id}: AMPL said "
                           f"'{ampl.getValue('solve_result')}'")

    # AMPL thinks in indices 0..48; we want real site labels back.
    #
    # The mapping CSV was created by prepare_data.py and connects:
    #     AMPL index 0, 1, ..., 48
    # to:
    #     actual reef labels such as 1, 4, 10, 31, etc.
    labels = pd.read_csv(config.mapping_csv(matrix_id))["site_id"].tolist()

    # AMPL returns every x variable as [index, value].
    # Values above 0.5 are treated as selected binary variables.
    picked = [int(r[0]) for r in ampl.getVariable("x").getValues().to_list()
              if float(r[1]) > 0.5]

    # Fixed-size models leave this empty.
    # Size models fill it with {actual site label: allocated acres}.
    sizes = {}

    if has_size:
        # Read the continuous AMPL size variable s for every internal index.
        acres = {int(r[0]): float(r[1])
                 for r in ampl.getVariable("s").getValues().to_list()}

        # Convert AMPL's internal indices back to actual site labels.
        # Only selected sites are included in the final dictionary.
        sizes = {labels[i]: round(acres.get(i, 0.0), 2) for i in picked}

    # The different AMPL model files use different objective names,
    # so objname comes from config.MIQP_MODELS.
    obj = float(ampl.getObjective(objname).value())

    # We have copied out everything needed, so close this AMPL session.
    ampl.close()

    # Return one consistent Python object regardless of which model was solved.
    return {"sites": sorted(labels[i] for i in picked), "obj": obj, "sizes": sizes}


def load_communities():
    """{community number: [site labels]} straight from the xlsx."""

    # The spreadsheet is arranged manually rather than as a standard header table,
    # so reading it as raw cells is simpler than assigning column names.
    raw = pd.read_excel(config.COMMUNITIES_XLSX, header=None).values

    out = {}

    # Row 0 contains headings, so the actual community rows begin at row 1.
    for r in range(1, raw.shape[0]):

        # A nonempty first cell means this row defines a community.
        if not pd.isna(raw[r, 0]):

            # Column 0 contains the community number.
            # Site labels begin in column 2 and continue across the row.
            # Empty spreadsheet cells are skipped.
            out[int(raw[r, 0])] = sorted(int(v) for v in raw[r, 2:] if not pd.isna(v))

    return out


def report(model, matrix_id, r, secs, comms):
    # Build the report as a list of lines.
    # Joining once at the end is easier to maintain than one very long string.
    L = ["=" * 70,
         f"M{matrix_id}  |  {model}  |  Pe = {config.CONST_P0:g} (constant)",
         "=" * 70,
         f"  objective : {r['obj']:.4f}",
         f"  time      : {secs:.2f}s",
         "",
         f"  SITES ({len(r['sites'])} of {len(config.CANDIDATE_SITES)})",
         f"    {r['sites']}",
         ""]

    # Only show the community table if the model actually has that constraint.
    # The base and size-only models do not need community membership here.
    if "comm" in model:
        L += ["  COMMUNITIES", f"    {'region':<21} {'got':>4} {'min':>4}  sites"]

        for c in sorted(comms):
            # Find which selected sites belong to this community.
            got = sorted(set(r["sites"]) & set(comms[c]))

            # Minimum number required by the AMPL community constraint.
            rmin = config.COMMUNITY_MINS[c]

            # This is useful when reading the result:
            # if got == rmin, the constraint may be actively affecting the solution.
            tight = "  <- stuck at the minimum" if len(got) == rmin else ""

            L.append(f"    C{c} {config.COMM_NAMES[c]:<18} {len(got):>4} {rmin:>4}  {got}{tight}")

        L.append("")

    # A nonempty sizes dictionary means this was a size or comm+size model.
    if r["sizes"]:
        L += ["  ACRES",
              f"    {sum(r['sizes'].values()):.0f} of {config.SIZE['TotReefSize']:.0f} spent"]

        # Group sites by allocated acreage.
        # This makes the common 5-acre/40-acre/50-acre pattern easy to see.
        grp = defaultdict(list)

        for site, a in r["sizes"].items():
            grp[a].append(site)

        # Print the largest allocations first.
        for a in sorted(grp, reverse=True):
            L.append(f"    {a:>6.1f} ac x{len(grp[a]):<2} : {sorted(grp[a])}")

        # Explain why different sites can have different maximum sizes.
        L += ["", "    (40 vs 50 ceiling is just a site's position in the list, see",
              "     config.SIZE.U_SPLIT. size_sweep.py redoes this with uniform bounds.)", ""]

    return "\n".join(L)


def main():
    # Command-line options let us run one model/matrix while debugging,
    # or reproduce the complete experiment with the default "all" and "both".
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(config.MIQP_MODELS) + ["all"], default="all")
    ap.add_argument("--matrix", choices=["1", "2", "both"], default="both")
    args = ap.parse_args()

    # Turn the command-line choice into the actual list of models to run.
    models = list(config.MIQP_MODELS) if args.model == "all" else [args.model]

    # The rest of the code can now use the same loop for one or both matrices.
    mats = ["1", "2"] if args.matrix == "both" else [args.matrix]

    # if any of the required .dat files are missing, abort before starting the solves.
    for m in mats:
        if not config.quad_dat(m, P0).exists():
            raise SystemExit(f"missing {config.quad_dat(m, P0).name} -- run:\n"
                             f"    python -m scripts.prepare_data")

    # Community information is only needed for reports involving a comm model.
    # Avoid reading the spreadsheet when running only base or size.
    comms = load_communities() if any("comm" in m for m in models) else {}

    # blocks holds the readable text reports.
    # rows holds the machine-friendly records that become the CSV.
    blocks, rows = [], []

    # Each model/matrix combination is one independent AMPL solve.
    for m in mats:
        for model in models:
            # Time the complete solve, including AMPL setup and result extraction.
            t = time.time()
            r = solve(model, m)
            secs = time.time() - t

            # just fomatting for txt
            block = report(model, m, r, secs, comms)
            print(block)
            blocks.append(block)

            # Also keep one flat row per solve for  csv output.
            rows.append({"matrix": f"M{m}", "model": model,
                         "objective": round(r["obj"], 4),
                         "acres": round(sum(r["sizes"].values()), 1) if r["sizes"] else "",
                         "seconds": round(secs, 2),
                         "sites": " ".join(map(str, r["sites"])),
                         "per_site_acres": " ".join(f"{s}={a:g}" for s, a in r["sizes"].items())})

    # Create runs/ if this is the first experiment on a clean checkout.
    config.RUNS_DIR.mkdir(exist_ok=True)

    # Use a filesystem-friendly name for comm+size.
    tag = "all" if args.model == "all" else args.model.replace("+", "_")

    # TXT is easy to read.
    (config.RUNS_DIR / f"miqp_{tag}.txt").write_text("\n".join(blocks), encoding="utf-8")

    # CSV stores the same results in a form that is easier to compare or plot.
    pd.DataFrame(rows).to_csv(config.RUNS_DIR / f"miqp_{tag}.csv", index=False)


    print(f"{len(rows)} solves -> runs/miqp_{tag}.txt, runs/miqp_{tag}.csv")


# if imported as module, don't run main...
if __name__ == "__main__":
    main()