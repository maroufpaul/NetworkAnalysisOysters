# src/opt/miqp.py
#
# The ONLY place that talks to AMPL. Same code run_miqp.py always had, in one
# function instead of four copies.
#
# run_miqp.py / run_miqp_comm.py / run_miqp_size.py /
# run_miqp_comm_size.py 

import pandas as pd
from amplpy import AMPL

import config


def solve(model_name, matrix_id, dat_override=None, verbose=False):
    """
    Solve one MIQP model from config.MIQP_MODELS.

    model_name   : "base" | "comm" | "size" | "comm+size"
    matrix_id    : 1 | 2 | "1" | "2" | "M1" | "M2"
    dat_override : Path to use instead of config.quad_dat(matrix_id). Used by
                   run_iterated.py, whose weights change every pass.

    Returns dict: sites (labels), obj, sizes {label: acres}, total_area, model,
                  matrix, dat
    """
    model_file, extra, objname, has_size = config.MIQP_MODELS[model_name]
    quad = dat_override or config.quad_dat(matrix_id)

    ampl = AMPL()
    ampl.eval("option solver gurobi;")
    ampl.eval(f"option gurobi_options '{config.GUROBI_OPTIONS}';")
    if not verbose:
        ampl.eval("option solver_msg 0;")

    ampl.read(str(config.AMPL_DIR / model_file))
    ampl.readData(str(quad))
    for attr in extra:
        ampl.readData(str(getattr(config, attr)))

    ampl.eval("solve;")

    status = ampl.getValue("solve_result")
    if status != "solved":
        raise RuntimeError(f"{model_name} matrix {matrix_id}: solve_result = {status}")

    labels = pd.read_csv(config.mapping_csv(matrix_id))["site_id"].tolist()
    picked = [int(r[0]) for r in ampl.getVariable("x").getValues().to_list()
              if float(r[1]) > 0.5]
    sites = sorted(labels[i] for i in picked)
    obj = float(ampl.getObjective(objname).value())

    sizes, total_area = {}, None
    if has_size:
        sv = {int(r[0]): float(r[1]) for r in ampl.getVariable("s").getValues().to_list()}
        sizes = {labels[i]: round(sv.get(i, 0.0), 3) for i in picked}
        total_area = round(sum(sizes.values()), 2)

    ampl.close()
    return {"sites": sites, "obj": obj, "sizes": sizes, "total_area": total_area,
            "model": model_name, "matrix": f"M{config.matrix_num(matrix_id)}",
            "dat": quad.name}


def selfcheck(matrix_id, tol=0.02):
    """Re-solve the constant-Pe baseline and compare against the paper's Table 4.
    Any script reporting numbers meant to sit beside the paper calls this first."""
    expect_obj = {"1": 14785.03, "2": 16446.59}[config.matrix_num(matrix_id)]
    expect_set = [10, 11, 12, 15, 16, 17, 20, 26, 27, 28, 29, 31, 32, 33,
                  36, 37, 40, 41, 44, 49, 51, 52, 53, 54, 59]
    r = solve("base", matrix_id)
    ok = abs(r["obj"] - expect_obj) < tol and r["sites"] == expect_set
    print(f"[selfcheck] M{config.matrix_num(matrix_id)}: obj={r['obj']:.2f} "
          f"(paper {expect_obj:.2f})  sites match={r['sites'] == expect_set}  "
          f"{'OK' if ok else '*** MISMATCH'}")
    return ok