# MIQP with sizing 

set N;

param K integer > 0;
param Pe{N} >= 0;
param W{N,N} >= 0 default 0;

param L{N} >= 0 default 0;   # lower bound
param U{N} >= 0;             # upper bound
param TotReefSize > 0;           # total reef available
param Sbar > 0;                  # scaling for size

var x{N} binary;             # site open/close
var s{N} >= 0;               # reef size

maximize Larvae:
    sum {i in N} Pe[i] * (s[i]/Sbar)
  + sum {l in N, i in N} W[l,i] * (s[l]/Sbar) * (s[i]/Sbar);

subject to PickK:
    sum {i in N} x[i] = K;

subject to LowerBound {i in N}:
    L[i] * x[i] <= s[i];

subject to UpperBound {i in N}:
    s[i] <= U[i] * x[i];

subject to TotalSize:
    sum {i in N} s[i] <= TotReefSize; 
    

#subject to ReefUpper:
 #   sum {i in N} U[i] * x[i] <= TotReefSize;