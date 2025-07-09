import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


# TODO: Human review needed
# Attempts made: Tried padding residuals to match n variables
# Suspected issues: BDQRTICNE has n-4 constraints for n variables
# (underdetermined system)
# Additional resources needed: Understanding of how pycutest handles
# underdetermined nonlinear equations


class BDQRTICNE(AbstractNonlinearEquations):
    """
    This problem is quartic and has a banded Hessian with bandwidth = 9
    This is a nonlinear equation variant of BDQRTIC

    Source: Problem 61 in
    A.R. Conn, N.I.M. Gould, M. Lescrenier and Ph.L. Toint,
    "Performance of a multifrontal scheme for partially separable
    optimization",
    Report 88/4, Dept of Mathematics, FUNDP (Namur, B), 1988.

    SIF input: Ph. Toint, Dec 1989.
              Nick Gould (nonlinear equation version), Jan 2019

    classification NOR2-AN-V-V
    """

    n: int = 5000
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def starting_point(self) -> Array:
        return jnp.ones(self.n, dtype=jnp.float64)

    def residual(self, y: Array, args) -> Array:
        """Compute the residuals"""
        n = self.n
        n_minus_4 = n - 4

        # Initialize residuals array with n elements (pad with zeros)
        residuals = jnp.zeros(n, dtype=y.dtype)

        # For each group i = 1, ..., n-4
        for i in range(n_minus_4):
            # L(i) = -4*y[i] - 3
            linear_part = -4.0 * y[i] - 3.0

            # G(i) = y[i]^2 + 2*y[i+1]^2 + 3*y[i+2]^2 + 4*y[i+3]^2 + 5*y[n-1]^2
            nonlinear_part = (
                y[i] ** 2
                + 2.0 * y[i + 1] ** 2
                + 3.0 * y[i + 2] ** 2
                + 4.0 * y[i + 3] ** 2
                + 5.0 * y[n - 1] ** 2
            )

            # Combined residual for group i
            residuals = residuals.at[i].set(linear_part + nonlinear_part)

        # Last 4 residuals remain zero
        return residuals

    def y0(self) -> Array:
        """Initial guess for the optimization problem."""
        return self.starting_point()

    def args(self):
        """Additional arguments for the residual function."""
        return None

    def expected_result(self) -> Array | None:
        """Expected result of the optimization problem."""
        # The SIF file doesn't provide the solution vector
        return None

    def expected_objective_value(self) -> Array | None:
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)
