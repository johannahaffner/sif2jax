# ECKERLE4LS - Nonlinear Least-Squares problem (NIST dataset)
#
# This problem involves a nonlinear least squares fit to a Gaussian peak function,
# arising from a circular interference transmittance study.
#
# Source: Problem 7 from
# NIST nonlinear least squares test set
# http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml
#
# SIF input: Ph. Toint, April 1997.
#
# Classification: SUR2-MN-3-0
#
# Number of observations: 35
# Number of parameters: 3
#
# Model: y = (b1/b2) * exp[-0.5*((x-b3)/b2)^2]
#
# Parameters:
#   b1: amplitude parameter
#   b2: width parameter (standard deviation)
#   b3: location parameter (mean)

# Number of data points
param m := 35;

# Data points
param x{1..m};
param y{1..m};

# Data values
data;
param x :=
1  400.0
2  405.0
3  410.0
4  415.0
5  420.0
6  425.0
7  430.0
8  435.0
9  436.5
10 438.0
11 439.5
12 441.0
13 442.5
14 444.0
15 445.5
16 447.0
17 448.5
18 450.0
19 451.5
20 453.0
21 454.5
22 456.0
23 457.5
24 459.0
25 460.5
26 462.0
27 463.5
28 465.0
29 470.0
30 475.0
31 480.0
32 485.0
33 490.0
34 495.0
35 500.0;

param y :=
1  1.575e-1
2  1.699e-1
3  2.350e-1
4  3.102e-1
5  4.917e-1
6  8.710e-1
7  1.718e0
8  3.682e0
9  4.944e0
10 6.637e0
11 8.796e0
12 1.168e1
13 1.484e1
14 1.739e1
15 1.710e1
16 1.342e1
17 7.600e0
18 2.245e0
19 1.120e0
20 8.178e-1
21 6.615e-1
22 5.880e-1
23 4.401e-1
24 3.431e-1
25 2.400e-1
26 1.909e-1
27 1.750e-1
28 1.381e-1
29 9.576e-2
30 7.274e-2
31 6.907e-2
32 6.012e-2
33 5.629e-2
34 4.992e-2
35 4.670e-2;

# Variables
var b1;
var b2;
var b3;

# Starting values - first starting point
let b1 := 1.0;
let b2 := 10.0;
let b3 := 500.0;

# Alternative starting point (commented out)
# let b1 := 1.5;
# let b2 := 5.0;
# let b3 := 450.0;

# Objective function: minimize sum of squared residuals
minimize obj: sum {i in 1..m} ((b1/b2) * exp(-0.5*((x[i]-b3)/b2)^2) - y[i])^2;

# Solution from NIST (certified values):
# b1 = 1.5543827178
# b2 = 4.0888321754
# b3 = 451.54121844
# Residual sum of squares = 1.4635887487E-03