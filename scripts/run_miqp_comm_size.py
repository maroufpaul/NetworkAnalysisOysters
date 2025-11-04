# D:\NetworkAnalysisOysters\scripts\run_miqp_comm_size.py
"""
Run MIQP optimization with BOTH community constraints AND variable sizing.

NORMALIZATION:
- Raw objective: Actual solver output (scales with reef sizes)
- Normalized objective: Adjusted for fair comparison with binary models
  Formula: obj_normalized = obj_raw / (avg_size / Sbar)²
"""

from pathlib import Path
from amplpy import AMPL
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AMPL_DIR = ROOT / "ampl"
DATA_XLSX = ROOT / "data" / "nk_All_060102final_56sites_Model.xlsx"
RUNS_DIR = ROOT / "runs"


# Hhelper function to get actual site labels

def load_real_labels():
    """
    Load actual site labels from Excel (e.g., 3, 4, 5, ..., 60).
    Returns list of 49 labels after dropping sites 66-72.
    """
    df = pd.read_excel(DATA_XLSX, header=None)
    labels = df.iloc[0, 1:].astype(int).tolist()
    drop = {66, 67, 68, 69, 70, 71, 72}
    labels = [x for x in labels if x not in drop]
    return labels


def main():
    """
    Run combined MIQP: community constraints + variable sizing.
    """
    
    RUNS_DIR.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("RUNNING COMBINED MIQP: COMMUNITY CONSTRAINTS + VARIABLE SIZING")
    print("=" * 80)
    
    #  Initialize AMPL
    ampl = AMPL()
    ampl.eval("option solver gurobi;")
    print("      Solver: Gurobi")
    
    # Load model
    model_file = AMPL_DIR / "oyster_comm_size.mod"
    ampl.read(str(model_file))
    print(f"      Model: {model_file.name}")
    
    # Load data files
    #print("\n[3/7] Loading data files...")
    
    data_quad = AMPL_DIR / "oyster_quad.dat"
    ampl.readData(str(data_quad))
    print(f"      Data 1: {data_quad.name} (N, K, Pe, W)")
    
    data_comm = AMPL_DIR / "oyster_comm.dat"
    ampl.readData(str(data_comm))
    print(f"      Data 2: {data_comm.name} (C1-C5)")
    
    data_size = AMPL_DIR / "oyster_size.dat"
    ampl.readData(str(data_size))
    print(f"      Data 3: {data_size.name} (L, U, TotReefSize, Sbar)")
    

    #  Solve
    # ------------------------------------------------------------------------
    #print("\n[4/7] Solving MIQP...")
    #print("      (This may take 1-2 minutes...)")
    ampl.eval("solve;")
    

    # Extract solution
    #print("\n[5/7] Extracting solution...")
    
    # Get site selection
    x_vals = ampl.getVariable("x").getValues()
    picked_idx = [int(row[0]) for row in x_vals.to_list() if float(row[1]) > 0.5]
    
    # Get reef sizes
    s_vals = ampl.getVariable("s").getValues()
    size_map = {int(row[0]): float(row[1]) for row in s_vals.to_list()}
    
    # Get objective value (raw)
    obj_raw = ampl.getObjective("TotalLarvae").value()
   
    # NORMALIZATION: Adjust for fair comparison with binary models
    Sbar = 20.0  # Scaling factor from data file
    total_area = sum(size_map[i] for i in picked_idx)
    avg_size = total_area / len(picked_idx)
    
    # Scaling factor accounts for bilinear objective
    # If avg reef is 40 acres (not 20), contribution is (40/20)² = 4× higher
    scaling_factor = (avg_size / Sbar) ** 2
    obj_normalized = obj_raw / scaling_factor
    
    print(f"      Sites selected: {len(picked_idx)}")
    print(f"      Total area: {total_area:.1f} acres")
    print(f"      Average reef size: {avg_size:.1f} acres")
    print(f"      ")
    print(f"      Objective (raw): {obj_raw:.2f}")
    print(f"      Objective (normalized): {obj_normalized:.2f}")
    print(f"      Scaling factor: {scaling_factor:.2f}×")
    
 
    # Map to actual site labels
    #print("\n[6/7] Mapping to actual site labels...")
    labels = load_real_labels()
    
    rows = []
    for i in picked_idx:
        site_id = labels[i]
        site_size = size_map.get(i, 0.0)
        rows.append({
            "site_index": i,
            "site_id": site_id,
            "size_acres": site_size
        })
    
    # ------------------------------------------------------------------------
    # 7. Save results
    # ------------------------------------------------------------------------
    print("\n[7/7] Saving results...")
    
    # Save CSV
    out_csv = RUNS_DIR / "miqp_comm_size_sites.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"      CSV: {out_csv}")
    
    # Save TXT summary
    out_txt = RUNS_DIR / "miqp_comm_size_summary.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("MIQP WITH COMMUNITY CONSTRAINTS + VARIABLE SIZING\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Model: oyster_comm_size.mod\n")
        f.write(f"Data:  oyster_quad.dat + oyster_comm.dat + oyster_size.dat\n\n")
        
        f.write(f"Objective (raw):        {obj_raw:.2f}\n")
        f.write(f"Objective (normalized): {obj_normalized:.2f}\n")
        f.write(f"Scaling factor:         {scaling_factor:.2f}×\n\n")
        
        f.write(f"Sites selected:   {len(picked_idx)}\n")
        f.write(f"Total area:       {total_area:.1f} acres\n")
        f.write(f"Average size:     {avg_size:.1f} acres/site\n\n")
        
        f.write("NORMALIZATION EXPLANATION:\n")
        f.write("-" * 80 + "\n")
        f.write(f"The raw objective ({obj_raw:.2f}) is higher than binary models\n")
        f.write(f"because reefs are larger (avg {avg_size:.1f} acres vs implicit 20).\n")
        f.write(f"\n")
        f.write(f"For fair comparison with binary models, we normalize by:\n")
        f.write(f"  scaling_factor = (avg_size / Sbar)² = ({avg_size:.1f} / 20)² = {scaling_factor:.2f}\n")
        f.write(f"  obj_normalized = {obj_raw:.2f} / {scaling_factor:.2f} = {obj_normalized:.2f}\n")
        f.write(f"\n")
        f.write(f"This normalized value is comparable to:\n")
        f.write(f"  - Baseline (binary, no constraints): 13,692\n")
        f.write(f"  - Community (binary, equity): 12,792\n")
        f.write("-" * 80 + "\n\n")
        
        f.write("Selected sites (sorted by size):\n")
        f.write("-" * 80 + "\n")
        
        sorted_rows = sorted(rows, key=lambda r: r['size_acres'], reverse=True)
        for row in sorted_rows:
            site_id = row['site_id']
            size = row['size_acres']
            f.write(f"  Site {site_id:>3}  │  {size:6.1f} acres\n")
        
        f.write("-" * 80 + "\n")
    
    print(f"      TXT: {out_txt}")
    
    # ------------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("✓ COMBINED MIQP COMPLETE!")
    print("=" * 80)
    print(f"\nObjective (raw):        {obj_raw:.2f}")
    print(f"Objective (normalized): {obj_normalized:.2f}")
    print(f"Scaling factor:         {scaling_factor:.2f}×")
    print(f"\nSites: {len(picked_idx)}")
    print(f"Total area: {total_area:.1f} / 1000 acres ({100*total_area/1000:.1f}% of budget)")
    print(f"Average reef size: {avg_size:.1f} acres")
    print(f"\nSites selected: {[labels[i] for i in picked_idx]}")
    print(f"\nResults saved:")
    print(f"  {out_csv}")
    print(f"  {out_txt}")
    
    print(f"\n" + "=" * 80)
    print("COMPARISON WITH OTHER MODELS:")
    print("=" * 80)
    print(f"Baseline (binary, no constraints):  13,692 (raw)")
    print(f"Community (binary, equity):          12,792 (raw)")
    print(f"Sizing (variable, no constraints):   53,244 (raw) ≈ 13,311 (normalized)")
    print(f"Combined (variable, equity):         {obj_raw:.0f} (raw) ≈ {obj_normalized:.0f} (normalized)")
    print(f"\nNormalized comparison shows:")
    print(f"  Combined achieves {100*obj_normalized/12792:.1f}% of community-only performance")
    print(f"  But with {avg_size/20:.1f}× larger budget ({total_area:.0f} vs ~500 acres)")
    
    ampl.close()
    
    return picked_idx, size_map, obj_raw, obj_normalized

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    try:
        selected_sites, sizes, obj_raw, obj_norm = main()
        print(f"\n✓ Success!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()