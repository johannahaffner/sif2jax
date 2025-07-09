import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class CYCLIC3(AbstractNonlinearEquations):
    """
    The cyclic cubic system whose root at zero has exponential multiplicity
    as a function of dimension.

    Source:  problem 8.2 in
    Wenrui Hao, Andrew J. Sommese and Zhonggang Zeng,
    "An algorithm and software for computing multiplicity structures
     at zeros of nonlinear systems", Technical Report,
    Department of Applied & Computational Mathematics & Statistics,
    University of Notre Dame, Indiana, USA (2012)

    SIF input: Nick Gould, Jan 2012.

    classification NOR2-AN-V-0
    """

    n: int = 100000  # Dimension parameter
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def num_vars(self) -> int:
        """Number of variables is N+2"""
        return self.n + 2

    def residual(self, y: Array, args) -> Array:
        """Compute the residuals of the cyclic cubic system"""
        n = self.n
        n_plus_2 = self.num_vars
        residuals = jnp.zeros(n_plus_2, dtype=y.dtype)

        # For i = 1 to N: E(i) = x(i)^3 - x(i+1)*x(i+2)
        for i in range(n):
            cubic_term = y[i] ** 3
            product_term = y[i + 1] * y[i + 2]
            residuals = residuals.at[i].set(cubic_term - product_term)

        # E(N+1) = x(N+1) - x(1)
        residuals = residuals.at[n].set(y[n] - y[0])

        # E(N+2) = x(N+2) - x(2)
        residuals = residuals.at[n + 1].set(y[n + 1] - y[1])

        return residuals

    def y0(self) -> Array:
        """Initial guess for the optimization problem."""
        # Starting point: all variables = 1000.0
        return jnp.full(self.num_vars, 1000.0, dtype=jnp.float64)

    def args(self):
        """Additional arguments for the residual function."""
        return None

    def expected_result(self) -> Array | None:
        """Expected result of the optimization problem."""
        # Solution should have all variables = 0
        return jnp.zeros(self.num_vars, dtype=jnp.float64)

    def expected_objective_value(self) -> Array | None:
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)
