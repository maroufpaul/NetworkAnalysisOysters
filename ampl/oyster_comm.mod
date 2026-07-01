# MIQP + community minimums (no sizing)
# oyster_comm.mod
set N;
set C1; set C2; set C3; set C4; set C5;

param K integer > 0;
param Pe{N} >= 0;
param W{N,N} >= 0 default 0;

# community minimums (supplied by oyster_comm.dat; defaults match config)
param rmin1 default 2; param rmin2 default 5; param rmin3 default 3;
param rmin4 default 2; param rmin5 default 3;

var x{N} binary;

maximize Larvae:
    sum {i in N} Pe[i]*x[i]
  + sum {i in N, j in N} W[i,j]*x[i]*x[j];

subject to PickK:
    sum {i in N} x[i] = K;

subject to Community1: sum {i in C1} x[i] >= rmin1;
subject to Community2: sum {i in C2} x[i] >= rmin2;
subject to Community3: sum {i in C3} x[i] >= rmin3;
subject to Community4: sum {i in C4} x[i] >= rmin4;
subject to Community5: sum {i in C5} x[i] >= rmin5;
