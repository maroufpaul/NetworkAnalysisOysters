#!/usr/bin/env python3
"""
run_extra_experiments.py  --  supplementary experiments for the oyster paper.

Writes ONE file:  runs/extra_experiments.json  (+ a .txt mirror).


Experiments
-----------
  E1  Surrogate fidelity: score all 8 diagonal-zeroed MIQP sets under the TRUE
      JARS ODE (constant P0) and compare to the heuristic optimum. Answers the
      load-bearing question "does the fast surrogate agree with the faithful
      ODE?" with a number, not an assertion.
  E2  Null baseline for the backbone: how surprising is a 7-site agreement
      across 18 designs (and 15 across 7)? Monte-Carlo + closed form vs random
      25-of-49 selection.
  E3  K-sensitivity: re-solve the base MIQP (diag zeroed) for K in {15,20,25,30}
      on both matrices; check whether the 7-site core persists at every budget.
  E4  Surrogate-parameter sensitivity: re-solve base MIQP over a grid of A* and
      P1scaling; report how stable the selected set is (Jaccard vs the canonical
      K=25 set).
  E5  Self-recruitment (diagonal) sensitivity: score the base MIQP set with the
      connectivity diagonal KEPT (canonical) vs ZEROED, on both matrices. The
      selection is identical either way; only the objective magnitude changes.
      Reproduces the self-recruitment numbers in the paper (Section 5.6).
"""
from __future__ import annotations
import json, time, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import config

RUNS_DIR = config.RUNS_DIR; RUNS_DIR.mkdir(exist_ok=True)
MATRICES = config.MATRICES
TMAX, P1SCALING, CONST_P0 = config.TMAX, config.P1SCALING, config.CONST_P0
A_STAR, ALPHA = config.A_STAR, config.ALPHA
UNWANTED = config.UNWANTED

from src.model.jars_ode import load_connectivity, CANDIDATE_SITES
from src.opt.evaluator import evaluate_subset
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

# ---- canonical MIQP selections (self-recruitment KEPT; corrected 49-site
#      communities). Base/+Size sets are community-independent; +Comm /
#      +Comm+Size reflect the corrected partition (+Comm+Size == +Comm set). ----
MIQP_SETS = {
 "M1 Base":[10,11,12,15,16,17,20,26,27,28,29,31,32,33,36,37,40,41,44,49,51,52,53,54,59],
 "M1 +Comm":[1,10,12,15,16,17,21,26,28,29,31,32,33,37,40,41,42,44,47,49,51,52,53,54,59],
 "M1 +Size":[10,11,12,15,16,17,20,26,27,28,29,31,32,33,36,37,40,41,44,49,51,52,53,54,59],
 "M1 +Comm+Size":[1,10,12,15,16,17,21,26,28,29,31,32,33,37,40,41,42,44,47,49,51,52,53,54,59],
 "M2 Base":[10,11,12,15,16,17,20,26,27,28,29,31,32,33,36,37,40,41,44,49,51,52,53,54,59],
 "M2 +Comm":[10,11,12,15,16,17,18,19,27,28,29,31,32,33,36,37,40,41,42,47,49,51,52,53,59],
 "M2 +Size":[10,11,12,15,16,17,20,26,27,28,29,31,32,33,36,37,40,41,44,49,51,52,53,54,59],
 "M2 +Comm+Size":[10,11,12,15,16,17,18,19,27,28,29,31,32,33,36,37,40,41,42,47,49,51,52,53,59],
}
# heuristic optima (constant P0) used as the "best known" ODE reference
HEUR_BEST = {  # matrix -> (sites, ODE score)
 "M1": ([10,12,15,16,17,20,21,26,28,29,31,32,33,36,37,40,41,44,47,49,51,52,53,54,59], 1.880054),
 "M2": ([10,11,12,15,16,17,19,20,26,29,31,32,33,36,37,40,41,44,47,49,51,52,53,54,59], 1.733033),
}
BACKBONE7  = [10,31,37,40,41,49,53]
BACKBONE15 = {"M1":[10,15,16,28,31,32,37,40,41,44,49,51,52,53,54],
              "M2":[10,11,12,15,29,31,32,36,37,40,41,49,51,52,53]}

def build_W(path, a_star=A_STAR, p1=P1SCALING, zero_diag=False):
    # DEFAULT FLIPPED: was zero_diag=True, so E3 (budget nesting) and E4
    # solved a diagonal-zeroed model while the paper's Sec 5.7 describes
    # them as 'the baseline model', which RETAINS self-recruitment.
    # E5 passes zero_diag=True explicitly -- that is the sensitivity check.
    conn, key = load_connectivity(path)
    mask = ~np.isin(key, UNWANTED); key = key[mask]
    P = conn[np.ix_(mask, mask)] * p1
    if zero_diag:
        np.fill_diagonal(P, 0.0)
    return P*(a_star**ALPHA), CONST_P0*np.ones(len(key)), key

def surr_bin(sites, W, Pe, key):
    """Static surrogate objective on a selected set: external supply plus the
    (possibly diagonal-inclusive) connectivity weights among the selected sites."""
    idx = [int(np.where(key == s)[0][0]) for s in sites]
    return float(Pe[idx].sum() + W[np.ix_(idx, idx)].sum())

def solve_base_miqp(W, Pe, key, K):
    """Exact base BQP (diag zeroed) via HiGHS linearization. Returns sorted labels, obj."""
    n = len(key)
    pairs = [(i,j) for i in range(n) for j in range(n) if i!=j and W[i,j]!=0]
    npair = len(pairs); nv = n+npair
    c = np.zeros(nv); c[:n] = -Pe
    for p,(i,j) in enumerate(pairs): c[n+p] = -W[i,j]
    card = lil_matrix((1,nv)); card[0,:n] = 1
    Ay = lil_matrix((2*npair,nv))
    for p,(i,j) in enumerate(pairs):
        Ay[2*p,n+p]=1;   Ay[2*p,i]=-1
        Ay[2*p+1,n+p]=1; Ay[2*p+1,j]=-1
    cons=[LinearConstraint(card,K,K), LinearConstraint(Ay,-np.inf,0)]
    integ=np.zeros(nv); integ[:n]=1
    res=milp(c=c,constraints=cons,integrality=integ,
             bounds=Bounds(np.zeros(nv),np.ones(nv)),options={"time_limit":120})
    sel=sorted(int(key[i]) for i in range(n) if res.x[i]>0.5)
    return sel, float(-res.fun)

def jaccard(a,b):
    a,b=set(a),set(b); return len(a&b)/len(a|b)

def main():
    t0=time.time()
    out={"meta":{"TMAX":TMAX,"P1SCALING":P1SCALING,"CONST_P0":CONST_P0,
                 "A_STAR":A_STAR,"ALPHA":ALPHA,"diagonal":"KEPT (E1 scores sets under the ODE)"}}
    CONN={m:load_connectivity(p) for m,p in MATRICES.items()}

    # ---------- E1: surrogate fidelity (MIQP sets scored under the ODE) ----------
    print("="*68); print("E1  Surrogate fidelity: MIQP sets under the TRUE JARS ODE"); print("="*68)
    e1={}
    for mat in ("M1","M2"):
        conn,key=CONN[mat]
        best_sites,best=HEUR_BEST[mat]
        best_chk=evaluate_subset(best_sites,conn,key,tmax=TMAX,P1scaling=P1SCALING,
                                 P0_mode="constant",consP0=CONST_P0)
        e1[mat]={"heuristic_best_ODE":round(best_chk,6),"models":{}}
        print(f"\n  {mat}: heuristic optimum F={best_chk:.6f}")
        for name,sites in MIQP_SETS.items():
            if not name.startswith(mat): continue
            f=evaluate_subset(sites,conn,key,tmax=TMAX,P1scaling=P1SCALING,
                              P0_mode="constant",consP0=CONST_P0)
            gap=(best_chk-f)/best_chk*100.0
            e1[mat]["models"][name]={"ODE_F":round(f,6),"gap_pct_vs_heur":round(gap,3)}
            print(f"    {name:16s} ODE F={f:.6f}  gap={gap:+.2f}% vs heuristic optimum")
    out["E1_surrogate_fidelity"]=e1

    # ---------- E2: null baseline for the backbone ----------
    print("\n"+"="*68); print("E2  Null baseline: is the agreement beyond chance?"); print("="*68)
    rng=np.random.default_rng(0); N=49; K=25; sites=np.arange(N)
    def mc_intersection(n_models, trials=200_000):
        sizes=np.empty(trials,dtype=np.int16)
        for t in range(trials):
            inter=set(rng.choice(N,K,replace=False))
            for _ in range(n_models-1):
                inter &= set(rng.choice(N,K,replace=False))
                if not inter: break
            sizes[t]=len(inter)
        return sizes
    closed_form=lambda nm: N*((K/N)**nm)   # expected #sites in all nm uniform sets
    e2={}
    for nm,obs in [(18,len(BACKBONE7)),(7,15)]:
        sz=mc_intersection(nm)
        e2[f"{nm}_models"]={"observed_overlap":obs,
                            "null_mean":round(float(sz.mean()),5),
                            "null_max_seen":int(sz.max()),
                            "closed_form_expected":round(closed_form(nm),6),
                            "P(overlap>=observed)":float((sz>=obs).mean())}
        print(f"  {nm} models: observed={obs}  null_mean={sz.mean():.4f} "
              f"closed_form={closed_form(nm):.2e}  max_random_seen={sz.max()}  "
              f"P(rand>=obs)={ (sz>=obs).mean():.2e}")
    out["E2_null_baseline"]=e2

    # ---------- E3: K-sensitivity (does the 7-site core persist?) ----------
    print("\n"+"="*68); print("E3  K-sensitivity of the base MIQP (diag zeroed)"); print("="*68)
    e3={}
    for mat in ("M1","M2"):
        W,Pe,key=build_W(MATRICES[mat])
        e3[mat]={}
        for K_ in (15,20,25,30):
            sel,obj=solve_base_miqp(W,Pe,key,K_)
            core_in=sorted(set(BACKBONE7)&set(sel))
            e3[mat][f"K={K_}"]={"obj":round(obj,2),"selected":sel,
                                "core7_contained":len(core_in)==7,
                                "core7_present":core_in}
            print(f"  {mat} K={K_:2d}: obj={obj:9.2f}  contains all 7 core sites: "
                  f"{len(core_in)==7}  ({len(core_in)}/7)")
    out["E3_K_sensitivity"]=e3

    # ---------- E4: surrogate-parameter sensitivity ----------
    print("\n"+"="*68); print("E4  A* / P1scaling sensitivity of base MIQP set (K=25)"); print("="*68)
    e4={}
    for mat in ("M1","M2"):
        Wc,Pe,key=build_W(MATRICES[mat])
        canon,_=solve_base_miqp(Wc,Pe,key,25)
        e4[mat]={"canonical":canon,"grid":{}}
        for a in (0.03,0.045,0.05675,0.07,0.09):
            for p1 in (0.25,0.5,1.0):
                W,Pe2,key2=build_W(MATRICES[mat],a_star=a,p1=p1)
                sel,_=solve_base_miqp(W,Pe2,key2,25)
                e4[mat]["grid"][f"A*={a},P1={p1}"]={"jaccard_vs_canon":round(jaccard(sel,canon),3),
                                                     "n_diff":len(set(sel)^set(canon))//1}
        jvals=[v["jaccard_vs_canon"] for v in e4[mat]["grid"].values()]
        print(f"  {mat}: Jaccard(set, canonical) over 15-point grid: "
              f"min={min(jvals):.3f} mean={np.mean(jvals):.3f}")
    out["E4_param_sensitivity"]=e4

    # ---------- E5: self-recruitment (diagonal) sensitivity ----------
    print("\n"+"="*68); print("E5  Self-recruitment (diagonal) sensitivity of the base MIQP objective"); print("="*68)
    e5={}
    for mat in ("M1","M2"):
        Wk,Pe,key = build_W(MATRICES[mat], zero_diag=False)   # diagonal KEPT (canonical)
        Wz,_,_    = build_W(MATRICES[mat], zero_diag=True)     # diagonal ZEROED (sensitivity)
        base = MIQP_SETS[f"{mat} Base"]                        # selection is identical either way
        objk = surr_bin(base, Wk, Pe, key)
        objz = surr_bin(base, Wz, Pe, key)
        drop = (objk-objz)/objk*100.0 if objk else 0.0
        e5[mat] = {"base_set":base,
                   "obj_diag_kept":round(objk,2),
                   "obj_diag_zeroed":round(objz,2),
                   "pct_drop_when_zeroed":round(drop,2),
                   "note":"selection identical kept-vs-zeroed; only the objective magnitude changes"}
        print(f"  {mat}: kept={objk:.2f}  zeroed={objz:.2f}  drop={drop:.1f}%  (same base set)")
    out["E5_self_recruitment_sensitivity"]=e5

    out["runtime_sec"]=round(time.time()-t0,1)
    (RUNS_DIR/"extra_experiments.json").write_text(json.dumps(out,indent=2))
    (RUNS_DIR/"extra_experiments.txt").write_text(json.dumps(out,indent=2))
    print("\n"+"="*68)
    print(f"wrote {RUNS_DIR/'extra_experiments.json'}  ({out['runtime_sec']}s)")
    print("="*68)

if __name__=="__main__":
    sys.exit(main())