import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class LIARWHDNE(AbstractNonlinearEquations):
    """
    This is a nonlinear equation variant of LIARWHD

    Source:
    G. Li,
    "The secant/finite difference algorithm for solving sparse
    nonlinear systems of equations",
    SIAM Journal on Optimization, (to appear), 1990.

    SIF input: Ph. Toint, Aug 1990.

    classification NOR2-AN-V-V

    This is a simplified version of problem NONDIA.
    """

    n: int = 5000  # Number of variables
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def num_residuals(self) -> int:
        """Number of residuals."""
        return 2 * self.n  # Two groups A(i) and B(i) for each i

    def starting_point(self) -> Array:
        """Return the starting point for the problem."""
        return jnp.full(self.n, 4.0, dtype=jnp.float64)

    def residual(self, y: Array, args) -> Array:
        """Compute the residual vector.

        The problem has residuals of the form:
        A(i): 2 * (x(i)^2 - x(1)) = 0  (pycutest uses 2.0 instead of 0.5)
        B(i): x(i) - 1.0 = 0
        """
        x = y

        # Vectorized implementation
        # A(i) residuals: appears pycutest uses factor 2 instead of 0.5
        a_residuals = 2.0 * (x * x - x[0])

        # B(i) residuals: x(i) - 1.0
        b_residuals = x - 1.0

        # Interleave A and B residuals
        residuals = jnp.zeros(2 * self.n, dtype=x.dtype)
        residuals = residuals.at[0::2].set(a_residuals)  # Even indices: A residuals
        residuals = residuals.at[1::2].set(b_residuals)  # Odd indices: B residuals

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
        # Solution should be all ones
        return jnp.ones(self.n, dtype=jnp.float64)

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
