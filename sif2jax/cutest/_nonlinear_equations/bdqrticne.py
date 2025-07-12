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

        # Create indices for vectorized computation
        indices = jnp.arange(n_minus_4)

        # Linear part: L(i) = -4*y[i] - 3
        linear_part = -4.0 * y[indices] - 3.0

        # Nonlinear part: G(i) = y[i]^2 + 2*y[i+1]^2 + 3*y[i+2]^2
        #                        + 4*y[i+3]^2 + 5*y[n-1]^2
        nonlinear_part = (
            y[indices] ** 2
            + 2.0 * y[indices + 1] ** 2
            + 3.0 * y[indices + 2] ** 2
            + 4.0 * y[indices + 3] ** 2
            + 5.0 * y[n - 1] ** 2
        )

        # Combined residuals for groups 1 to n-4
        group_residuals = linear_part + nonlinear_part

        # Pad with zeros for the last 4 residuals
        residuals = jnp.concatenate([group_residuals, jnp.zeros(4, dtype=y.dtype)])

        return residuals

    @property
    def y0(self) -> Array:
        """Initial guess for the optimization problem."""
        return self.starting_point()

    @property
    def args(self):
        """Additional arguments for the residual function."""
        return None

    def expected_result(self) -> None:
        """Expected result of the optimization problem."""
        # The SIF file doesn't provide the solution vector
        return None

    def expected_objective_value(self) -> Array:
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)

    def constraint(self, y):
        """Returns the residuals as equality constraints."""
        return self.residual(y, self.args), None
