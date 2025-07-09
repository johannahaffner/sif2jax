import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class BROWNALE(AbstractNonlinearEquations):
    """
    Brown almost linear least squares problem.
    This problem is a sum of n least-squares groups, the last one of
    which has a nonlinear element.
    It Hessian matrix is dense.
    This is a nonlinear equation version of problem BROWNAL.

    Source: Problem 27 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#79
    SIF input: Ph. Toint, Dec 1989.

    classification NOR2-AN-V-0
    """

    n: int = 200
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def starting_point(self) -> Array:
        return jnp.full(self.n, 0.5, dtype=jnp.float64)

    def num_residuals(self) -> int:
        return self.n

    def residual(self, y: Array, args) -> Array:
        """Compute the residuals of the Brown almost linear problem"""
        n = self.n
        n_plus_1 = float(n + 1)

        # Initialize residuals array
        residuals = []

        # For each group i = 1, ..., n-1
        for i in range(n - 1):
            # G(i) = sum(y[j] for j != i) + 2*y[i] - (n+1)
            # This is equivalent to sum(y) + y[i] - (n+1)
            res_i = jnp.sum(y) + y[i] - n_plus_1
            residuals.append(res_i)

        # For the last group G(n), we have a product of all variables minus 1
        # G(n) = prod(y[j] for j in range(n)) - 1
        res_n = jnp.prod(y) - 1.0
        residuals.append(res_n)

        return jnp.array(residuals)

    def y0(self) -> Array:
        """Initial guess for the optimization problem."""
        return self.starting_point()

    def args(self):
        """Additional arguments for the residual function."""
        return None

    def expected_result(self) -> Array | None:
        """Expected result of the optimization problem."""
        # The solution has all components equal to 1 except the last
        # which solves the product equation
        # For the Brown almost linear problem, the solution is approximately all ones
        return jnp.ones(self.n, dtype=jnp.float64)

    def expected_objective_value(self) -> Array | None:
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)
