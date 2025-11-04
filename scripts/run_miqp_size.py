# D:\NetworkAnalysisOysters\scripts\run_miqp_size.py
from pathlib import Path
from amplpy import AMPL
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AMPL_DIR = ROOT / "ampl"
DATA_XLSX = ROOT / "data" / "nk_All_060102final_56sites_Model.xlsx"
RUNS_DIR = ROOT / "runs"

def load_real_labels():
    df = pd.read_excel(DATA_XLSX, header=None)
    labels = df.iloc[0, 1:].astype(int).tolist()
    drop = {66, 67, 68, 69, 70, 71, 72}
    labels = [x for x in labels if x not in drop]
    return labels

def main():
    RUNS_DIR.mkdir(exist_ok=True)

    ampl = AMPL()
    ampl.eval("option solver gurobi;")

    ampl.read(str(AMPL_DIR / "oyster_size.mod"))     # uses set N, prof model
    ampl.readData(str(AMPL_DIR / "oyster_quad.dat")) # N, K, Pe, W
    ampl.readData(str(AMPL_DIR / "oyster_size.dat")) # L, U, TotReefSize, Sbar

    ampl.eval("solve;")

    x_vals = ampl.getVariable("x").getValues()
    s_vals = ampl.getVariable("s").getValues()

    picked_idx = [int(row[0]) for row in x_vals.to_list() if float(row[1]) > 0.5]
    size_map = {int(row[0]): float(row[1]) for row in s_vals.to_list()}

    labels = load_real_labels()

    rows = []
    for i in picked_idx:
        rows.append({"site_id": labels[i], "size": size_map.get(i, 0.0)})

    out_csv = RUNS_DIR / "miqp_size_sites.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    out_txt = RUNS_DIR / "miqp_size_summary.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("MIQP (prof) with sizing\n")
        f.write(f"picked {len(picked_idx)} sites (x=1)\n")
        for i in picked_idx:
            f.write(f"  label {labels[i]:>3}  size={size_map.get(i, 0.0):8.3f}\n")

    print("[size] picked:", [labels[i] for i in picked_idx])
    print("[size] wrote:", out_csv)
    print("[size] wrote:", out_txt)

if __name__ == "__main__":
    main()
