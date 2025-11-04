#
# oyster_comm_size.mod
# Combined: Community Constraints + Variable Reef Sizing
#
# This model combines:
# - Community equity constraints (minimum sites per region)
# - Variable reef sizing (each site can have different area)
# - Area budget constraint (total acres limited)
#



set N;    # All candidate sites (0-48 in your case, 49 sites total)

# Community definitions (geographic regions, e.g., rivers)
set C1;   # Community 1 (7 sites)
set C2;   # Community 2 (19 sites) 
set C3;   # Community 3 (10 sites)
set C4;   # Community 4 (5 sites)
set C5;   # Community 5 (8 sites)


# Selection parameters
param K integer > 0;              # Number of sites to select (K=25)

# Larvae parameters
param Pe{N} >= 0;                 # External larvae P0(k) at each site
param W{N,N} >= 0 default 0;      # Internal connectivity weights (pre-computed)

# Sizing parameters
param L{N} >= 0 default 5;        # Minimum reef size (acres) per site
param U{N} >= 0 default 50;       # Maximum reef size (acres) per site
param TotReefSize > 0;            # Total area budget (acres), e.g., 1000
param Sbar > 0 default 20;        # Scaling factor for size normalization


# DECISION VARIABLES


var x{N} binary;                  # x[i] = 1 if site i is selected, 0 otherwise
var s{N} >= 0;                    # s[i] = reef size at site i (acres)
                                  # If x[i]=0, then s[i]=0
                                  # If x[i]=1, then L[i] <= s[i] <= U[i]


# OBJECTIVE FUNCTION

# Maximize total larvae (external + internal connectivity)
# Contribution scales with reef size
maximize TotalLarvae:
    sum {i in N} Pe[i] * (s[i]/Sbar)                        # External larvae
  + sum {l in N, i in N} W[l,i] * (s[l]/Sbar) * (s[i]/Sbar); # Internal larvae


# - External: Larger reefs receive more larvae (proportional to size)
# - Internal: Larvae flow from l to i scales with BOTH reef sizes
#   * Larger source (s[l]) produces more larvae
#   * Larger destination (s[i]) receives more larvae
# - Sbar=20 normalizes contributions (20-acre reef contributes 1.0)


# CONSTRAINTS

# Constraint 1: Budget - select exactly K sites
subject to PickK:
    sum {i in N} x[i] = K;
# Must select exactly 25 sites

# ----------------------------------------------------------------------------
# Constraint 2-6: Community coverage (minimum sites per community)
# These ensure each stakeholder region gets oysters to harvest
# ----------------------------------------------------------------------------

subject to Community1_Min:
    sum {i in C1} x[i] >= 2;
# Community 1 must have at least 2 sites

subject to Community2_Min:
    sum {i in C2} x[i] >= 5;
# Community 2 (largest) must have at least 5 sites

subject to Community3_Min:
    sum {i in C3} x[i] >= 3;
# Community 3 must have at least 3 sites

subject to Community4_Min:
    sum {i in C4} x[i] >= 2;
# Community 4 must have at least 2 sites

subject to Community5_Min:
    sum {i in C5} x[i] >= 3;
# Community 5 must have at least 3 sites

# Total minimums: 2+5+3+2+3 = 15 sites required by communities
# Remaining: K - 15 = 25 - 15 = 10 "flex" sites optimizer can place anywhere

# ----------------------------------------------------------------------------
# Constraint 7-8: Reef sizing bounds (only applies if site is selected)
# ----------------------------------------------------------------------------

subject to MinSize {i in N}:
    L[i] * x[i] <= s[i];
# If x[i]=0 (not selected): s[i] >= 0 (automatically satisfied)
# If x[i]=1 (selected): s[i] >= L[i] (must build at least L[i] acres)
# This prevents building reefs too small to be viable

subject to MaxSize {i in N}:
    s[i] <= U[i] * x[i];
# If x[i]=0 (not selected): s[i] <= 0 (forces s[i]=0, no reef)
# If x[i]=1 (selected): s[i] <= U[i] (can't exceed site capacity)
# Site capacity limited by water depth, substrate area, geography

# ----------------------------------------------------------------------------
# Constraint 9: Total area budget
# ----------------------------------------------------------------------------

subject to AreaBudget:
    sum {i in N} s[i] <= TotReefSize;
# Total reef area across all selected sites can't exceed budget
# Example: With 1000 acres and K=25, average = 40 acres/site
# But optimizer can allocate unevenly (e.g., 50 for hubs, 5 for weak sites)

# ============================================================================
# HOW THIS MODEL WORKS
# ============================================================================
#
# The optimizer must:
# 1. Select exactly 25 sites (PickK)
# 2. Ensure C1≥2, C2≥5, C3≥3, C4≥2, C5≥3 (Community minimums)
# 3. For each selected site: 5 ≤ size ≤ 50 (or 40) acres (MinSize, MaxSize)
# 4. Total area ≤ 1000 acres (AreaBudget)
# 5. Among valid combinations, maximize TotalLarvae
#
# Expected behavior:
# - Sites required for community equity → build small (5-10 acres) if low value
# - High-connectivity hub sites → build large (40-50 acres) to max out value
# - Remaining flex sites → optimize size based on connectivity contribution
#
# Trade-offs:
# - Community constraints force inclusion of some suboptimal sites
# - Sizing flexibility allows "damage control" by building them small
# - Result: Better than fixed-size community model (can size down weak sites)
#          Worse than unconstrained sizing model (forced to include weak sites)