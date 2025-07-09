# TODO: Human review needed
# Constraint dimension mismatch: pycutest returns 998 equality constraints
# but our implementation expects 499. This needs investigation to determine
# the correct constraint formulation

from jax import numpy as jnp
from jaxtyping import Array, Float

from ..._problem import AbstractNonlinearEquations


class ARWHDNE(AbstractNonlinearEquations):
    """A quartic problem whose Hessian is an arrow-head (downwards) with
    diagonal central part and border-width of 1.
    Nonlinear equation version of problem ARWHEAD.

    Source: Problem 55 in
    A.R. Conn, N.I.M. Gould, M. Lescrenier and Ph.L. Toint,
    "Performance of a multifrontal scheme for partially separable
    optimization",
    Report 88/4, Dept of Mathematics, FUNDP (Namur, B), 1988.

    SIF input: Ph. Toint, Dec 1989.

    classification NOR2-AN-V-V
    """

    n: int = 500  # Default to n=500
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def residual(self, y, args) -> Float[Array, "499"]:
        """Residual function for the nonlinear equations."""
        x = y
        n = self.n
        ngs = n - 1  # Number of group sets

        # Initialize residuals array
        residuals = []

        # For each group set i = 1, ..., n-1
        for i in range(ngs):
            # Linear part: -4*x(i) - 3
            linear_part = -4.0 * x[i] - 3.0

            # Nonlinear part: x(i)^2 + x(n)^2
            nonlinear_part = x[i] * x[i] + x[n - 1] * x[n - 1]

            # Combined residual for group i
            residuals.append(linear_part + nonlinear_part)

        return jnp.array(residuals)

    def y0(self) -> Float[Array, "500"]:
        """Initial guess for the optimization problem."""
        return jnp.ones(self.n)

    def args(self):
        """Additional arguments for the residual function."""
        return None

    def expected_result(self) -> Float[Array, "500"] | None:
        """Expected result of the optimization problem."""
        # The SIF file doesn't provide a solution
        return None

    def expected_objective_value(self) -> Float[Array, ""] | None:
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)
