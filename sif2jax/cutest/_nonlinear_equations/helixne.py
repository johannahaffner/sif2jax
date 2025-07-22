import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class HELIXNE(AbstractNonlinearEquations):
    """
    The "Helix" problem in 3 variables. This is a nonlinear equation version
    of problem HELIX

    Source: problem 7 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#12 (p. 58)
    SIF input: Ph. Toint, Dec 1989.
    Modification as a set of nonlinear equations: Nick Gould, Oct 2015.

    classification  NOR2-AN-3-3
    """

    n: int = 3
    m: int = 3
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def num_residuals(self) -> int:
        """Number of residuals."""
        return self.m

    def starting_point(self) -> Array:
        """Return the starting point for the problem."""
        return jnp.array([-1.0, 0.0, 0.0])

    def residual(self, y: Array, args) -> Array:
        """Compute the residual vector.

        The problem has residuals of the form:
        A: 0.1 * (x3 - 10.0 * theta(x1, x2)) = 0
        B: 0.1 * (sqrt(x1^2 + x2^2) - 1.0) = 0
        C: x3 = 0

        where theta(x1, x2) = arctan2(x2, x1) / (2*pi)
        """
        x1, x2, x3 = y[0], y[1], y[2]
        residuals = jnp.zeros(self.m, dtype=jnp.float64)

        # Compute theta = arctan2(x2, x1) / (2*pi)
        # Using TWOPII = 0.15915494 = 1/(2*pi)
        twopii = 0.15915494
        theta = jnp.arctan2(x2, x1) * twopii

        # A: 0.1 * (x3 - 10.0 * theta) = 0
        residuals = residuals.at[0].set(0.1 * (x3 - 10.0 * theta))

        # B: 0.1 * (sqrt(x1^2 + x2^2) - 1.0) = 0
        r = jnp.sqrt(x1 * x1 + x2 * x2)
        residuals = residuals.at[1].set(0.1 * (r - 1.0))

        # C: x3 = 0
        residuals = residuals.at[2].set(x3)

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
        # Solution is x1 = 1.0, x2 = 0.0, x3 = 0.0
        return jnp.array([1.0, 0.0, 0.0])

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
        lower = jnp.full(self.n, -jnp.inf, dtype=jnp.float64)
        upper = jnp.full(self.n, jnp.inf, dtype=jnp.float64)
        return lower, upper
