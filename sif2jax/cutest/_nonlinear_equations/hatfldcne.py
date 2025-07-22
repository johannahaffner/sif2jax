import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class HATFLDCNE(AbstractNonlinearEquations):
    """
    A test problem from the OPTIMA user manual.

    Source:
    "The OPTIMA user manual (issue No.8, p. 26)",
    Numerical Optimization Centre, Hatfield Polytechnic (UK), 1989.

    SIF input: Ph. Toint, May 1990.
    Bound-constrained nonlinear equations version: Nick Gould, June 2019.

    TODO: Human review needed
    Test failure: test_correct_number_of_finite_bounds expects different count
    Our implementation counts 24 finite bounds (upper bounds only)
    Pycutest counts 48 (24 lower + 24 upper using ±1e20 for unbounded)

    classification NOR2-AN-25-25
    """

    m: int = 25
    n: int = 25
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def starting_point(self) -> Array:
        # From SIF: 'DEFAULT' 0.9
        return jnp.full(self.n, 0.9, dtype=jnp.float64)

    def num_residuals(self) -> int:
        return self.m

    def residual(self, y: Array, args) -> Array:
        """Compute the residuals of the Hatfield C problem"""
        x = y
        n = self.n

        # Initialize residuals
        residuals = jnp.zeros(self.m, dtype=jnp.float64)

        # G(1): x(1) - 1.0 = 0
        residuals = residuals.at[0].set(x[0] - 1.0)

        # G(i) for i = 2 to n-1: x(i+1) - x(i)^2 = 0
        for i in range(1, n - 1):
            # Element E(i): x(i)^2
            residuals = residuals.at[i].set(x[i + 1] - x[i] * x[i])

        # G(n): x(n) - 1.0 = 0
        residuals = residuals.at[n - 1].set(x[n - 1] - 1.0)

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
        # Solution would be all ones
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
        """Bounds from the AMPL model: 0 <= x[i] <= 10 for i=1..N-1, x[N] free."""
        # From AMPL: subject to cons1{i in 1..N-1}: 0.0 <= x[i] <= 10.0;
        # Variable N is free (no bounds)
        lower = jnp.full(self.n, 0.0, dtype=jnp.float64)
        upper = jnp.full(self.n, 10.0, dtype=jnp.float64)
        # Free the last variable (both bounds become ±inf)
        lower = lower.at[self.n - 1].set(-jnp.inf)
        upper = upper.at[self.n - 1].set(jnp.inf)
        return lower, upper
