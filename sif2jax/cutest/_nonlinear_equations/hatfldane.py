import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class HATFLDANE(AbstractNonlinearEquations):
    """
    A test problem from the OPTIMA user manual.

    Source:
    "The OPTIMA user manual (issue No.8, p. 12)",
    Numerical Optimization Centre, Hatfield Polytechnic (UK), 1989.

    SIF input: Ph. Toint, May 1990.
    Bound-constrained nonlinear equations version: Nick Gould, June 2019.

    classification NOR2-AN-4-4
    """

    m: int = 4
    n: int = 4
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def starting_point(self) -> Array:
        # From SIF: 'DEFAULT' 0.1
        return jnp.full(self.n, 0.1, dtype=jnp.float64)

    def num_residuals(self) -> int:
        return self.m

    def residual(self, y: Array, args) -> Array:
        """Compute the residuals of the Hatfield A problem"""
        x = y
        n = self.n

        # Initialize residuals
        residuals = jnp.zeros(self.m, dtype=jnp.float64)

        # G(1): x(1) - 1.0 = 0
        residuals = residuals.at[0].set(x[0] - 1.0)

        # G(i) for i = 2 to n: x(i-1) - sqrt(x(i)) = 0
        for i in range(1, n):
            # Element E(i): sqrt(x(i))
            root = jnp.sqrt(x[i])
            residuals = residuals.at[i].set(x[i - 1] - root)

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
        # Solution would be x1=1, x2=1, x3=1, x4=1
        return jnp.ones(self.n, dtype=jnp.float64)

    def expected_objective_value(self) -> Array:
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)

    def constraint(self, y):
        """Returns the residuals as equality constraints."""
        return self.residual(y, self.args), None

    @property
    def bounds(self) -> tuple[Array, Array] | None:
        """Lower bounds from the SIF file."""
        # XL HATFLDA 'DEFAULT' 0.0000001
        lower = jnp.full(self.n, 0.0000001, dtype=jnp.float64)
        upper = jnp.full(self.n, jnp.inf, dtype=jnp.float64)
        return lower, upper
