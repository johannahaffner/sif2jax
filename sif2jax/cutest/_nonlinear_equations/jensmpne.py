import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class JENSMPNE(AbstractNonlinearEquations):
    """
    The Jennrich and Sampson problem. This is a nonlinear equation version
    of problem JENSMP.

    Source:  Problem 6 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    SIF input: Ph. Toint, Dec 1989.
    Modification as a set of nonlinear equations: Nick Gould, Oct 2015.

    classification NOR2-AN-2-10
    """

    n: int = 2
    m: int = 10
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def num_residuals(self) -> int:
        """Number of residuals."""
        return self.m

    def starting_point(self) -> Array:
        """Return the starting point for the problem."""
        return jnp.array([0.3, 0.4])

    def residual(self, y: Array, args) -> Array:
        """Compute the residual vector.

        The problem has residuals of the form:
        G(i): exp(i*x1) + exp(i*x2) - (2 + 2*i) = 0, for i = 1..10
        """
        x1, x2 = y[0], y[1]
        residuals = jnp.zeros(self.m, dtype=jnp.float64)

        for i in range(1, self.m + 1):
            # G(i) = exp(i*x1) + exp(i*x2) - (2 + 2*i)
            a_i = jnp.exp(i * x1)
            b_i = jnp.exp(i * x2)
            target = 2.0 + 2.0 * i
            residuals = residuals.at[i - 1].set(a_i + b_i - target)

        return residuals

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
        # Not explicitly given in the SIF file
        return jnp.array([0.0, 0.0])

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
        """Free bounds for all variables."""
        return None
