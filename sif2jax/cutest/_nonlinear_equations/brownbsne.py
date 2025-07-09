import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class BROWNBSNE(AbstractNonlinearEquations):
    """
    Brown badly scaled problem in 2 variables.
    This problem is a sum of n-1 sets of 3 groups, one of then involving
    a nonlinear element and all being of the least square type.
    It Hessian matrix is tridiagonal.
    This is a nonlinear equation version of BROWNBS

    Source: Problem 4 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#25
    SIF input: Ph. Toint, Dec 1989.

    classification NOR2-AN-2-3
    """

    n: int = 2
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def starting_point(self) -> Array:
        return jnp.ones(self.n, dtype=jnp.float64)

    def num_residuals(self) -> int:
        return 3  # 3 groups for n=2

    def residual(self, y: Array, args) -> Array:
        """Compute the residuals of the Brown badly scaled problem"""
        # For n=2, we have n-1=1 sets of 3 groups
        # Group A(1): y[0] - 1000000.0
        # Group B(1): y[1] - 0.000002
        # Group C(1): y[0] * y[1] - 2.0

        res1 = y[0] - 1000000.0
        res2 = y[1] - 0.000002
        res3 = y[0] * y[1] - 2.0

        return jnp.array([res1, res2, res3])

    def y0(self) -> Array:
        """Initial guess for the optimization problem."""
        return self.starting_point()

    def args(self):
        """Additional arguments for the residual function."""
        return None

    def expected_result(self) -> Array | None:
        """Expected result of the optimization problem."""
        # The solution is x[0] = 1e6, x[1] = 2e-6
        return jnp.array([1e6, 2e-6], dtype=jnp.float64)

    def expected_objective_value(self) -> Array | None:
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)
