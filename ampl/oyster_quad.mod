# oyster_quad.mod

# sets / params
set N;                      # sites
param K integer > 0;        # how many sites to pick
param Pe{N} >= 0 default 0; # external larvae vector
param W{N, N} >= 0 default 0;  # internal weights (already scaled!) e.g. (A^1.72 * P1)

# vars
var x{N} binary;            # pick site i ?

# objective:
# maximize external + internal
# external  : sum_i Pe[i] * x[i]
# internal  : sum_i sum_j P1[i,j] * x[i] * x[j]
maximize score:
    sum {i in N} Pe[i] * x[i]
  + sum {i in N, j in N} W[i,j] * x[i] * x[j];

# choose exactly k sites
subject to choose_k:
    sum {i in N} x[i] = K;
