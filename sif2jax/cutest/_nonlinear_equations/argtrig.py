import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class ARGTRIG(AbstractNonlinearEquations):
    """
    Variable dimension trigonometric problem
    This problem is a sum of n least-squares groups, each of
    which has n+1 nonlinear elements.  Its Hessian matrix is dense.

    Source:  Problem 26 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    SIF input: Ph. Toint, Dec 1989.

    classification NOR2-AN-V-V
    """

    n: int = 200
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def starting_point(self) -> Array:
        return jnp.full(self.n, 1.0 / self.n, dtype=jnp.float64)

    def num_residuals(self) -> int:
        return self.n

    def residual(self, y: Array, args) -> Array:
        """Compute the residuals of the trigonometric problem"""
        n = self.n

        # Compute residuals
        residuals = []
        for i in range(n):
            # For each group i, compute:
            # REALI * (cos(y[i]) + sin(y[i])) + sum(cos(y[j]) for j in range(n))
            # minus (n + i + 1)
            sum_cos = jnp.sum(jnp.cos(y))
            res_i = (
                float(i + 1) * (jnp.cos(y[i]) + jnp.sin(y[i]))
                + sum_cos
                - float(n + i + 1)
            )
            residuals.append(res_i)

        return jnp.array(residuals)

    def y0(self) -> Array:
        """Initial guess for the optimization problem."""
        return self.starting_point()

    def args(self):
        """Additional arguments for the residual function."""
        return None

    def expected_result(self) -> Array | None:
        """Expected result of the optimization problem."""
        # The SIF file mentions solution value 0.0, but not the exact solution vector
        return None

    def expected_objective_value(self) -> Array | None:
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)
