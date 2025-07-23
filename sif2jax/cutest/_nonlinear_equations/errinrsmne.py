import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class ERRINRSMNE(AbstractNonlinearEquations):
    """
    A variable dimension version of an incorrect version of the
    chained Rosenbrock function (ERRINROS) by Luksan et al.
    This is a nonlinear equation variant of ERRINRSM.

    Source: problem 28 in
    L. Luksan, C. Matonoha and J. Vlcek
    Modified CUTE problems for sparse unconstrained optimization
    Technical Report 1081
    Institute of Computer Science
    Academy of Science of the Czech Republic

    SIF input: Ph. Toint, Sept 1990.
               this version Nick Gould, June, 2013
               Nick Gould (nonlinear equation version), Jan 2019

    classification NOR2-AN-V-V
    """

    n: int = 50
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def num_residuals(self) -> int:
        """Number of residuals."""
        return 2 * (self.n - 1)  # Two constraints per i from 2 to n

    def starting_point(self) -> Array:
        """Return the starting point for the problem."""
        return jnp.full(self.n, -1.0, dtype=jnp.float64)

    def residual(self, y: Array, args) -> Array:
        """Compute the residual vector.

        The problem has residuals of the form:
        SQ(i): x(i-1) + alpha(i) * ELA(i) = 0
        B(i): x(i) - 1.0 = 0

        where alpha(i) = 16.0 * (1.5 + sin(i))^2
        and ELA(i) = -x(i)^2
        """
        x = y
        residuals = jnp.zeros(2 * (self.n - 1), dtype=jnp.float64)

        for i in range(2, self.n + 1):
            # Compute alpha(i) = 16.0 * (1.5 + sin(i))^2
            sini = jnp.sin(float(i))
            alpha = 1.5 + sini
            ai = 16.0 * alpha * alpha

            # SQ(i): x(i-1) + ai * (-x(i)^2) = 0
            residuals = residuals.at[2 * (i - 2)].set(
                x[i - 2] - ai * x[i - 1] * x[i - 1]
            )

            # B(i): x(i) - 1.0 = 0
            residuals = residuals.at[2 * (i - 2) + 1].set(x[i - 1] - 1.0)

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
        # Not provided in the SIF file
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
        """Free bounds for all variables."""
        return None
