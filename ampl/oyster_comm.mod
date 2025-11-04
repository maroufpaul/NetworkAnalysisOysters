# MIQP + community minimums (no sizing)
# oyster_comm.mod
set N;
set C1; set C2; set C3; set C4; set C5;

param K integer > 0;
param Pe{N} >= 0;
param W{N,N} >= 0 default 0;

var x{N} binary;

maximize Larvae:
    sum {i in N} Pe[i]*x[i]
  + sum {i in N, j in N} W[i,j]*x[i]*x[j];

subject to PickK:
    sum {i in N} x[i] = K;

subject to Community1: sum {i in C1} x[i] >= 2;
subject to Community2: sum {i in C2} x[i] >= 5;
subject to Community3: sum {i in C3} x[i] >= 3;
subject to Community4: sum {i in C4} x[i] >= 2;
subject to Community5: sum {i in C5} x[i] >= 3;
