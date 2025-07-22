import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class EXTROSNBNE(AbstractNonlinearEquations):
    """
    The extended Rosenbrock function (nonseparable version).
    This is a nonlinear equation variant of EXTROSNB

    Source: problem 10 in
    Ph.L. Toint,
    "Test problems for partially separable optimization and results
    for the routine PSPMIN",
    Report 83/4, Department of Mathematics, FUNDP (Namur, B), 1983.

    See also Buckley#116.  Note that MGH#21 is the separable version.
    SIF input: Ph. Toint, Dec 1989.
              Nick Gould (nonlinear equation version), Jan 2019

    classification NOR2-AN-V-V
    """

    m: int = 1000  # Number of equations (same as n)
    n: int = 1000  # Default value from SIF
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def starting_point(self) -> Array:
        # From SIF: 'DEFAULT' -1.0
        return jnp.full(self.n, -1.0, dtype=jnp.float64)

    def num_residuals(self) -> int:
        return self.m

    def residual(self, y: Array, args) -> Array:
        """Compute the residuals of the extended Rosenbrock problem"""
        x = y
        n = self.n

        # Initialize residuals
        residuals = jnp.zeros(self.m, dtype=jnp.float64)

        # First equation: SQ1 = x1 - 1.0
        residuals = residuals.at[0].set(x[0] - 1.0)

        # Remaining equations: SQ(i) = 0.1 * (x(i) - x(i-1)^2)
        # Element type ETYPE: -V1^2
        for i in range(1, n):
            # SQ(i) has x(i) with coefficient 1.0 and scale 0.1
            # ELA(i) uses x(i-1) as V1, contributing -V1^2
            # So: SQ(i) = 0.1 * (x(i) - x(i-1)^2) = 0
            residuals = residuals.at[i].set(0.1 * (x[i] - x[i - 1] ** 2))

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
        # Solution would be all ones for this problem
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
        """No bounds for this problem."""
        return None
