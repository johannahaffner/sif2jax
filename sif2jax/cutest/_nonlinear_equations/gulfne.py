import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class GULFNE(AbstractNonlinearEquations):
    """
    The Gulf RD test problem in 3 variables.
    This function  is a nonlinear least squares with 99 groups.  Each
    group has a nonlinear element of exponential type.

    The number of groups may be varied, but must be larger or equal to 3

    Source: problem 11 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#27
    SIF input: Ph. Toint, Dec 1989.
    Modification as a set of nonlinear equations: Nick Gould, Oct 2015.

    classification NOR2-MN-3-99
    """

    m: int = 99
    n: int = 3
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def starting_point(self) -> Array:
        return jnp.array([5.0, 2.5, 0.15], dtype=jnp.float64)

    def num_residuals(self) -> int:
        return self.m

    def residual(self, y: Array, args) -> Array:
        """Compute the residuals of the Gulf problem"""
        x1, x2, x3 = y[0], y[1], y[2]

        # Initialize residuals
        residuals = jnp.zeros(self.m, dtype=jnp.float64)

        for i in range(self.m):
            # t = (i+1) * 0.01 for 1-based indexing
            t = (i + 1) * 0.01

            # Gulfian function element calculations
            ymv2 = 25.0 + (-50.0 * jnp.log(t)) ** (2.0 / 3.0) - x2
            a = jnp.abs(ymv2) ** x3 / x1
            expma = jnp.exp(-a)

            # G(i) = E(i) - t
            residuals = residuals.at[i].set(expma - t)

        return residuals

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
