"""
JANNSON4 problem

Source: example 4 in
C. Jannson
"Convex-concave extensions"
BIT 40(2) 2000:291-313

SIF input: Nick Gould, September 2000

classification OQR2-AN-V-2
"""

import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class JANNSON4(AbstractConstrainedMinimisation):
    @property
    def name(self) -> str:
        return "JANNSON4"

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 10000  # N by default
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
        # The objective has an extra x1 term and an extra x2 term
        # Based on testing: 2*(x1-1)^2 + 2*(x2-1)^2 + sum_{i=3}^n (x_i-1)^2
        x1 = y[0]
        x2 = y[1]
        x1_term = 2 * (x1 - 1.0) ** 2
        x2_term = 2 * (x2 - 1.0) ** 2
        rest_terms = jnp.sum((y[2:] - 1.0) ** 2)
        return x1_term + x2_term + rest_terms

    def constraint(self, y):
        # Q constraint (inequality): sum of x_i^2 >= 1.0
        q_constraint = jnp.sum(y**2) - 1.0

        # P constraint (inequality): x1 * x2 >= 0.1
        p_constraint = y[0] * y[1] - 0.1

        # Return (None, inequality_constraints) since we have no equality constraints
        return None, jnp.array([q_constraint, p_constraint])

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
        # From SIF file: Solution is 9.80197D+03 for n = 10000
        return jnp.array(9801.97)
