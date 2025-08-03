import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class EXPFITNE(AbstractNonlinearEquations):
    """
    A simple exponential fit in 2 variables

    Source:
    A.R. Conn, N. Gould and Ph.L. Toint,
    "LANCELOT, a Fortran package for large-scale nonlinear optimization",
    Springer Verlag, FUNDP, 1992.

    SIF input: Ph. Toint, Jan 1991.
    Nonlinear-equations version of EXPFIT.SIF, Nick Gould, Jan 2020.

    classification NOR2-AN-2-10
    """

    m: int = 10
    n: int = 2
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Parameters
    h: float = 0.25

    def starting_point(self) -> Array:
        # Default starting point since not specified in SIF
        return jnp.zeros(self.n, dtype=jnp.float64)

    def num_residuals(self) -> int:
        return self.m

    def residual(self, y: Array, args) -> Array:
        """Compute the residuals of the exponential fitting problem"""
        alpha, beta = y[0], y[1]

        # Initialize residuals
        residuals = jnp.zeros(self.m, dtype=jnp.float64)

        for i in range(self.m):
            # i+1 because SIF uses 1-based indexing
            real_i = float(i + 1)
            ih = self.h * real_i

            # Element EXPIH: alpha * exp(beta * ih)
            expwih = jnp.exp(beta * ih)
            f_i = alpha * expwih

            # Residual R(i) = f_i - real_i * h
            residuals = residuals.at[i].set(f_i - real_i * self.h)

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
        # Solution is not provided in the SIF file
        return jnp.zeros(self.n, dtype=jnp.float64)

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
        """No bounds for this problem."""
        return None
