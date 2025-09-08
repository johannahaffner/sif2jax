import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class DENSCHNDNE(AbstractNonlinearEquations):
    """
    Source: an example problem (p. 83) in
    J.E. Dennis and R.B. Schnabel,
    "Numerical Methods for Unconstrained Optimization and Nonlinear
    Equations",
    Prentice-Hall, Englewood Cliffs, 1983.

    SIF input: Ph. Toint, Nov 1990.
    Nonlinear-equations version of DENSCHND.SIF, Nick Gould, Jan 2020.

    classification NOR2-AN-3-3
    """

    n: int = 3
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def starting_point(self) -> Array:
        return jnp.full(3, 10.0, dtype=jnp.float64)

    def num_residuals(self) -> int:
        return jnp.array(3)

    def residual(self, y: Array, args) -> Array:
        """Compute the residuals of the Dennis-Schnabel problem D"""
        x1, x2, x3 = y[0], y[1], y[2]

        # Group A: x1^2 + x2^3 - x3^4
        res_a = x1**2 + x2**3 - x3**4

        # Group B: 2 * x1*x2*x3
        res_b = 2.0 * x1 * x2 * x3

        # Group C: 2*x1*x2 - 3*x2*x3 + x1*x3
        res_c = 2.0 * x1 * x2 - 3.0 * x2 * x3 + x1 * x3

        return jnp.array([res_a, res_b, res_c])

    @property
    def y0(self) -> Array:
        """Initial guess for the optimization problem."""
        return self.starting_point()

    @property
    def args(self):
        """Additional arguments for the residual function."""
        return None

    @property
    def expected_result(self) -> Array:
        """Expected result of the optimization problem."""
        # Solution is not provided in the SIF file
        return self.starting_point()

    @property
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
