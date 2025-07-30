import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class HIMMELBFNE(AbstractNonlinearEquations):
    """
    A 4 variables data fitting problems by Himmelblau.

    Source: problem 32 in
    D.H. Himmelblau,
    "Applied nonlinear programming",
    McGraw-Hill, New-York, 1972.

    See Buckley#76 (p. 66)

    SIF input: Ph. Toint, Dec 1989.
    Nonlinear-equations version of HIMMELBF.SIF, Nick Gould, Jan 2020.

    classification NOR2-AN-4-7
    """

    n: int = 4
    m: int = 7
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def num_residuals(self) -> int:
        """Number of residuals."""
        return self.m

    def starting_point(self) -> Array:
        """Return the starting point for the problem."""
        return jnp.array([2.7, 90.0, 1500.0, 10.0])

    def residual(self, y: Array, args) -> Array:
        """Compute the residual vector.

        The problem has residuals of the form:
        G(i): 0.0001 * (HF(x, a_i, b_i) - 1.0) = 0

        where HF(x, a, b) = U/V with:
        U = x1^2 + a*x2^2 + a^2*x3^2
        V = b*(1 + a*x4^2)
        """
        x1, x2, x3, x4 = y[0], y[1], y[2], y[3]

        # Problem data
        a_vals = jnp.array(
            [0.0, 0.000428, 0.001000, 0.001610, 0.002090, 0.003480, 0.005250]
        )
        b_vals = jnp.array([7.391, 11.18, 16.44, 16.20, 22.20, 24.02, 31.32])

        residuals = jnp.zeros(self.m, dtype=jnp.float64)

        for i in range(self.m):
            a = a_vals[i]
            b = b_vals[i]

            # Compute U and V
            u = x1 * x1 + a * x2 * x2 + a * a * x3 * x3
            v = b * (1.0 + a * x4 * x4)

            # HF = U/V
            hf = u / v

            # G(i) = 0.0001 * (HF - 1.0)
            # Note: pycutest inverts the scale factor for NLE problems
            residuals = residuals.at[i].set(10000.0 * (hf - 1.0))

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
        return jnp.array([0.0, 0.0, 0.0, 0.0])

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
