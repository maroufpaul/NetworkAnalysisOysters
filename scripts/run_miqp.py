# D:\NetworkAnalysisOysters\scripts\run_miqp.py
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
    return labels  # 49 labels, index 0..48

def main():
    RUNS_DIR.mkdir(exist_ok=True)

    ampl = AMPL()
    ampl.eval("option solver gurobi;")

    ampl.read(str(AMPL_DIR / "oyster_quad.mod"))  # uses set N
    ampl.readData(str(AMPL_DIR / "oyster_quad.dat"))  # defines N, K, Pe, W

    ampl.eval("solve;")

    x_vals = ampl.getVariable("x").getValues()
    picked_idx = [int(row[0]) for row in x_vals.to_list() if float(row[1]) > 0.5]

    labels = load_real_labels()
    rows = [{"site_id": labels[i]} for i in picked_idx]

    out_csv = RUNS_DIR / "miqp_sites.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    out_txt = RUNS_DIR / "miqp_summary.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("Plain MIQP (no communities, no sizing)\n")
        f.write(f"picked {len(picked_idx)} sites\n")
        for i in picked_idx:
            f.write(f"  label {labels[i]:>3}\n")

    print("[miqp] picked:", [labels[i] for i in picked_idx])
    print("[miqp] wrote:", out_csv)
    print("[miqp] wrote:", out_txt)

if __name__ == "__main__":
    main()
