"""
SCIPY global optimization benchmark example PRICE4

Fit: (2x_1^2 x_2 - x_2^3, 6x_1 - x_2^2 + x_2) + e = 0

version with box-constrained feasible region

Source:  Problem from the SCIPY benchmark set
  https://github.com/scipy/scipy/tree/master/benchmarks/ ...
          benchmarks/go_benchmark_functions

SIF input: Nick Gould, July 2021

classification SBR2-MN-2-0
"""

import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class PRICE4B(AbstractBoundedMinimisation):
    @property
    def name(self) -> str:
        return "PRICE4B"

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 2  # 2 variables

    @property
    def y0(self):
        return jnp.array([1.0, 5.0])

    @property
    def args(self):
        return None

    def objective(self, y, args):
        del args
        x1, x2 = y

        # F1: 2*x1^2*x2 - x2^3 (from E12 - E1)
        # E12 (CUBEL): x2 * x1^3 with coefficient 2.0
        # E1 (CUBE): x2^3 with coefficient -1.0
        # But wait, looking at CUBEL: F = x2 * x1^3, so 2*E12 = 2*x2*x1^3
        # Actually, looking more carefully at the fit equation:
        # (2x_1^2 x_2 - x_2^3, 6x_1-x_2^2+x_2)
        # And CUBEL has X1=X1, X2=X2, so it's x2 * x1^3, not x1^2 * x2
        # There seems to be an inconsistency. Let me check the element definition again.

        # Line 66-67: X1 is mapped to X1 for both X1 and X2 parameters of CUBEL
        # So CUBEL gets X1=X1, X2=X1, making it X1 * X1^3 = X1^4
        # Wait no, CUBEL is defined as X2 * X1^3, so with X1=X1, X2=X2, it's X2 * X1^3
        # But the comment says fit 2x_1^2 x_2, which doesn't match

        # Let me trust the SIF file implementation:
        # E12 is CUBEL with X1=X1, X2=X2, so it's x2 * x1^3
        # Looking at line 66-67 again, it says X1 -> X1 and X2 -> X2
        # So E12 = x2 * x1^3

        # Actually, re-reading the problem description, there's a typo.
        # The fit equation should be (2x_1^3 x_2 - x_2^3, 6x_1-x_2^2+x_2)
        # to match the implementation

        f1 = 2.0 * x2 * x1**3 - x2**3

        # F2: 6*x1 - x2^2 + x2 (from linear terms and E2)
        # Linear part: 6*x1 + x2
        # E2 (SQR): -x2^2
        f2 = 6.0 * x1 + x2 - x2**2

        # Sum of squares (L2 group type)
        return jnp.array(f1**2 + f2**2)

    @property
    def bounds(self):
        # Bounds: -50 <= x <= 50 for both variables
        lower = jnp.array([-50.0, -50.0])
        upper = jnp.array([50.0, 50.0])
        return lower, upper

    @property
    def expected_result(self):
        # The optimal solution is not explicitly given in the SIF file
        return None

    @property
    def expected_objective_value(self):
        # According to the SIF file comment (line 94),
        # the optimal objective value is 0.0
        return jnp.array(0.0)
