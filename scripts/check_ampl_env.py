# scripts/check_ampl_env.py

from amplpy import AMPL

def main():
    ampl = AMPL()

    print("=== AMPL ENV CHECK ===")

    # 1) what version?
    try:
        print("AMPL version:", ampl.get_option("version"))
    except Exception as e:
        print("could not get version:", e)

    # 2) what solver does AMPL THINK it's using right now?
    try:
        print("current solver:", ampl.get_option("solver"))
    except Exception as e:
        print("could not get solver:", e)

    # 3) list installation dir (sometimes helpful)
    try:
        print("AMPL installed at:", ampl.get_option("ampl_include"))
    except Exception as e:
        print("could not get ampl_include:", e)

    # 4) run the SMALLEST possible model (LP, not quadratic)
    ampl.eval(r"""
    var x >= 0;
    var y >= 0;
    maximize z: 3*x + 2*y;
    subject to c1: x + y <= 10;
    """)
    ampl.solve()

    out = ampl.get_output()
    print("=== AMPL OUTPUT BELOW ===")
    print(out if out.strip() else "(no output)")

    # try reading the solution
    try:
        x = ampl.get_variable("x").value()
        y = ampl.get_variable("y").value()
        print(f"solution x={x}, y={y}")
    except Exception as e:
        print("could not read solution:", e)

if __name__ == "__main__":
    main()
