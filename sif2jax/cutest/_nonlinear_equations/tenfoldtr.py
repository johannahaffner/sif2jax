import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class TENFOLDTR(AbstractNonlinearEquations):
    """
    The ten-fold triangular system whose root at zero has multiplicity 10

    Problem source:
    Problem 8.3 in Wenrui Hao, Andrew J. Sommese and Zhonggang Zeng,
    "An algorithm and software for computing multiplicity structures
    at zeros of nonlinear systems", Technical Report,
    Department of Applied & Computational Mathematics & Statistics,
    University of Notre Dame, Indiana, USA (2012)

    SIF input: Nick Gould, Jan 2012.

    classification NOR2-AN-V-V
    """

    n: int = 1000
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def name(self) -> str:
        return "10FOLDTR"

    def starting_point(self) -> Array:
        return jnp.full(self.n, 10.0, dtype=jnp.float64)

    def num_residuals(self) -> int:
        return self.n

    def residual(self, y: Array, args) -> Array:
        """Compute the residuals of the ten-fold triangular system"""
        res = jnp.zeros(self.n, dtype=y.dtype)

        # E(i) = sum(x[j] for j in range(i)) for i in range(n-2)
        for i in range(self.n - 2):
            res = res.at[i].set(jnp.sum(y[: i + 1]))

        # E(n-2) = (sum(x[j] for j in range(n-1)))^2
        res = res.at[self.n - 2].set(jnp.sum(y[: self.n - 1]) ** 2)

        # E(n-1) = (sum(x[j] for j in range(n)))^5
        res = res.at[self.n - 1].set(jnp.sum(y) ** 5)

        return res

    def y0(self) -> Array:
        """Initial guess for the optimization problem."""
        return self.starting_point()

    def args(self):
        """Additional arguments for the residual function."""
        return None

    def expected_result(self) -> Array | None:
        """Expected result of the optimization problem."""
        # The SIF file mentions root at zero has multiplicity 10
        return jnp.zeros(self.n)

    def expected_objective_value(self) -> Array | None:
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)
