import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class DENSCHNCNE(AbstractNonlinearEquations):
    """
    Source: an example problem (p. 98) in
    J.E. Dennis and R.B. Schnabel,
    "Numerical Methods for Unconstrained Optimization and Nonlinear
    Equations",
    Prentice-Hall, Englewood Cliffs, 1983.

    SIF input: Ph. Toint, Nov 1990.
    Nonlinear-equations version of DENSCHNC.SIF, Nick Gould, Jan 2020.

    classification NOR2-AN-2-2
    """

    n: int = 2
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def starting_point(self) -> Array:
        return jnp.array([2.0, 3.0], dtype=jnp.float64)

    def num_residuals(self) -> int:
        return 2

    def residual(self, y: Array, args) -> Array:
        """Compute the residuals of the Dennis-Schnabel problem C"""
        x1, x2 = y[0], y[1]

        # Group A: x1^2 + x2^2 - 2
        res_a = x1**2 + x2**2 - 2.0

        # Group B: exp(x1 - 1) + x2^3 - 2
        res_b = jnp.exp(x1 - 1.0) + x2**3 - 2.0

        return jnp.array([res_a, res_b])

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
        # Solution is not provided in the SIF file
        return self.starting_point()

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
