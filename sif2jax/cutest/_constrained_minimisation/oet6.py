"""
Problem OET6: Nonlinear Chebyshev approximation problem (discretized).

A nonlinear programming formulation of a discretization of a nonlinear
Chebychev problem.

The problem is:
    min  max || phi(x,w) ||, for all w in the interval I.
     x    w

I is discretized, and the problem solved over the discrete points.

Nonlinear programming formulation:
    min   u     s.t.  u - phi >= 0, u + phi >= 0
    x,u

Specific problem: I = [-0.5,0.5]
phi(x,w) = 1/(1+w) - x1*exp(w*x3) - x2*exp(w*x4)

Source: K. Oettershagen
"Ein superlinear knonvergenter algorithmus zur losung
 semi-infiniter optimierungsproblem",
Ph.D thesis, Bonn University, 1982

SIF input: Nick Gould, February, 1994.

classification LOR2-AN-5-V
"""

import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class OET6(AbstractConstrainedMinimisation):
    """OET6: Nonlinear Chebyshev approximation problem.

    Variables: u, x1, x2, x3, x4 (5 total)
    Discretization: M+1 points over interval [-0.5, 0.5] (default M=500)
    Constraints: 2*(M+1) inequality constraints
    """

    # Problem parameters
    M: int = 500
    _lower: float = -0.5
    _upper: float = 0.5

    @property
    def _h(self):
        return (self._upper - self._lower) / self.M

    @property
    def _w_points(self):
        return jnp.linspace(self._lower, self._upper, self.M + 1)

    @property
    def name(self) -> str:
        return "OET6"

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self) -> int:
        return 5  # u, x1, x2, x3, x4

    @property
    def m(self) -> int:
        return 2 * (self.M + 1)  # 2 inequality constraints per discretization point

    @property
    def y0(self):
        # Initial values should match SIF file (all zeros by default)
        return jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])  # u, x1, x2, x3, x4

    @property
    def args(self):
        return self._w_points

    def objective(self, y, args):
        """Minimize u."""
        u = y[0]
        return u

    def _phi(self, w, x1, x2, x3, x4):
        """Evaluate phi(x,w) = 1/(1+w) - x1*exp(w*x3) - x2*exp(w*x4)."""
        return 1.0 / (1.0 + w) - x1 * jnp.exp(w * x3) - x2 * jnp.exp(w * x4)

    def constraint(self, y):
        """Constraints from SIF file:
        LO(I): U - X1*EXP(X3*W) - X2*EXP(X4*W) >= -1/(1+W)
              →  U - X1*EXP(X3*W) - X2*EXP(X4*W) + 1/(1+W) >= 0
        UP(I): U + X1*EXP(X3*W) + X2*EXP(X4*W) >= 1/(1+W)
              →  U + X1*EXP(X3*W) + X2*EXP(X4*W) - 1/(1+W) >= 0
        """
        w_points = self.args
        u, x1, x2, x3, x4 = y

        # Compute the exponential elements
        exp1 = x1 * jnp.exp(x3 * w_points)  # X1*EXP(X3*W)
        exp2 = x2 * jnp.exp(x4 * w_points)  # X2*EXP(X4*W)

        # LO(I): U - X1*EXP(X3*W) - X2*EXP(X4*W) + 1/(1+W) >= 0
        lower_constraints = u - exp1 - exp2 + 1.0 / (1.0 + w_points)

        # UP(I): U + X1*EXP(X3*W) + X2*EXP(X4*W) - 1/(1+W) >= 0
        upper_constraints = u + exp1 + exp2 - 1.0 / (1.0 + w_points)

        # Interleave constraints: LO(0), UP(0), LO(1), UP(1), ...
        # This matches how pycutest orders them
        inequality_constraints = jnp.stack(
            [lower_constraints, upper_constraints], axis=1
        ).reshape(-1)

        return None, inequality_constraints

    @property
    def bounds(self):
        """All variables are free (unbounded)."""
        return None

    @property
    def expected_result(self):
        """Optimal solution not provided in SIF file."""
        return None

    @property
    def expected_objective_value(self):
        """Optimal objective value not provided in SIF file."""
        return None
