# TODO: Human review needed
# Attempts made: Fixed objective function interpretation (G group exclusion), tests pass
# Suspected issues: Jacobian tests hang due to computational complexity (20K vars)
# Resources needed: Performance optimization for large-scale Jacobian computation
# Status: Functionally correct - passes objective, gradient, constraints. Jacobian fails

"""
JANNSON3 problem

Source: example 3 in
C. Jannson
"Convex-concave extensions"
BIT 40(2) 2000:291-313

SIF input: Nick Gould, September 2000

classification OQR2-AN-V-3
"""

import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class JANNSON3(AbstractConstrainedMinimisation):
    @property
    def name(self) -> str:
        return "JANNSON3"

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 20000  # 2*N where N=10000 by default
    n_equality_constraints: int = 1  # L constraint
    n_inequality_constraints: int = 2  # Q and P constraints

    @property
    def y0(self):
        # Starting point is not specified in the SIF file, use default 0
        return jnp.zeros(self.n)

    @property
    def args(self):
        return None

    def objective(self, y, args):
        del args
        # Based on pycutest testing, the G group doesn't seem to contribute to objective
        # Only G0 and G(i) groups contribute
        x1 = y[0]

        # G0 group: has x1 with coeff 1.0, constant 1.0, uses L2 group type (gvar^2)
        # gvar = x1 - 1.0
        g0_term = (x1 - 1.0) ** 2

        # G(i) groups: each has x(i) with coeff 1.0, constant 1.0, uses L2 group type
        # gvar = x(i) - 1.0 for each i
        gi_terms = jnp.sum((y - 1.0) ** 2)

        return g0_term + gi_terms

    def constraint(self, y):
        n_half = self.n // 2

        # L constraint (equality): sum of x_i = 1
        eq_constraint = jnp.sum(y) - 1.0

        # Q constraint (inequality): sum of x_i^2 >= 0.75
        q_constraint = jnp.sum(y**2) - 0.75

        # P constraint (inequality): sum of x_i * x_{n+i} >= 1/(5*n)
        x_first = y[:n_half]
        x_second = y[n_half : 2 * n_half]
        p_constraint = jnp.sum(x_first * x_second) - 1.0 / (5.0 * n_half)

        return jnp.array([eq_constraint]), jnp.array([q_constraint, p_constraint])

    @property
    def bounds(self):
        # -1 <= x_i <= 1 for all i
        lower = jnp.full(self.n, -1.0)
        upper = jnp.full(self.n, 1.0)
        return lower, upper

    @property
    def expected_result(self):
        return None

    @property
    def expected_objective_value(self):
        # From SIF file: Solution is 1.99985D+04 for n = 10000
        return jnp.array(19998.5)
