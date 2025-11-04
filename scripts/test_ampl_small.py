# scripts/test_ampl_small.py

from amplpy import AMPL, DataFrame

def main():
    ampl = AMPL()
    # use the solver you actually installed
    ampl.option["solver"] = "highs"  # or "gurobi"

    # 1) define the model right here
    ampl.eval(r"""
    set N;
    param k integer > 0;
    param Pe{N} >= 0 default 0;
    param P1{N, N} >= 0 default 0;

    var x{N} binary;

    maximize score:
        sum {i in N} Pe[i] * x[i]
      + sum {i in N, j in N} P1[i,j] * x[i] * x[j];

    subject to choose_k:
        sum {i in N} x[i] = k;
    """)

    # 2) tiny data: 3 sites
    sites = [10, 11, 12]
    dfN = DataFrame(1, "site")
    dfN.setValues(sites)
    ampl.setData(dfN, "N")

    ampl.param["k"] = 2

    # external
    dfPe = DataFrame(("site", "Pe"), [(10, 5.0), (11, 3.0), (12, 1.0)])
    ampl.setData(dfPe, "Pe")

    # internal (make 10 and 11 like each other a lot)
    dfP1 = DataFrame(("i", "j", "P1"), [
        (10, 11, 4.0),
        (11, 10, 4.0),
        (10, 10, 0.0),
        (11, 11, 0.0),
        (12, 12, 0.0),
    ])
    ampl.setData(dfP1, "P1")

    ampl.solve()

    print("[AMPL OUTPUT]")
    print(ampl.get_output())

    try:
        print("solve_result =", ampl.get_value("solve_result"))
    except Exception:
        pass

    x_vals = ampl.getVariable("x").getValues().toDict()
    picked = [int(k[0] if isinstance(k, tuple) else k)
              for k, v in x_vals.items() if float(v) > 0.5]

    print("picked =", picked)

if __name__ == "__main__":
    main()
