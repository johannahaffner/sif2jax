"""MEYER3NE - Meyer's problem in 3 variables (nonlinear equation version).

Problem:
--------
A problem arising in the analysis of the resistance of a
thermistor, as formulated by Meyer. This is a nonlinear equation version
of problem MEYER3.

This function is a nonlinear least squares with 16 groups. Each
group has a nonlinear element.

Source:
-------
Problem 10 in
J.J. More', B.S. Garbow and K.E. Hillstrom,
"Testing Unconstrained Optimization Software",
ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

See also Buckley #29 (p. 73).

SIF input: Ph. Toint, Dec 1989.
Modification as a set of nonlinear equations: Nick Gould, Oct 2015.

classification NOR2-RN-3-16
"""

import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class MEYER3NE(AbstractNonlinearEquations):
    """MEYER3NE - Meyer's problem in 3 variables (nonlinear equation version)."""

    n: int = 3
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def starting_point(self) -> Array:
        """Return the starting point."""
        return jnp.array([0.02, 4000.0, 250.0], dtype=jnp.float64)

    def num_residuals(self) -> int:
        """Return the number of residuals."""
        return 16

    def residual(self, y: Array, args) -> Array:
        """Compute the residuals of Meyer's problem."""
        x1, x2, x3 = y

        # Constants from the SIF file
        y_data = jnp.array(
            [
                34780.0,
                28610.0,
                23650.0,
                19630.0,
                16370.0,
                13720.0,
                11540.0,
                9744.0,
                8261.0,
                7030.0,
                6005.0,
                5147.0,
                4427.0,
                3820.0,
                3307.0,
                2872.0,
            ],
            dtype=jnp.float64,
        )

        # Compute residuals
        residuals = []
        for i in range(16):
            # T values: 45 + 5*i for i=1..16 -> 50, 55, ..., 125
            t = 45.0 + 5.0 * (i + 1)

            # Gaussian function: x1 * exp(x2 / (t + x3))
            tpv3 = t + x3
            expa = jnp.exp(x2 / tpv3)
            fval = x1 * expa

            residuals.append(fval - y_data[i])

        return jnp.array(residuals)

    @property
    def y0(self) -> Array:
        """Initial guess for the optimization problem."""
        return self.starting_point()

    @property
    def args(self):
        """Additional arguments for the residual function."""
        return None

    def expected_result(self) -> Array:
        """Expected result of the optimization problem."""
        # For nonlinear equations, the expected result is where residuals are zero
        # The actual solution is not provided in the SIF file
        return jnp.zeros(3, dtype=jnp.float64)

    def expected_objective_value(self) -> Array:
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)

    def constraint(self, y):
        """Returns the residuals as equality constraints."""
        return self.residual(y, self.args), None

    @property
    def bounds(self) -> tuple[Array, Array] | None:
        """No bounds for this problem."""
        return None
