import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


# TODO: Human review needed - Jacobian mismatch with pycutest
# The last row of the Jacobian (derivatives of product term) differs
# between automatic differentiation and pycutest's approximation
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

        # The problem definition from More et al.:
        # For i = 1 to n-1: f_i(x) = x_i + sum(x_j) - (n+1)
        # For i = n: f_n(x) = prod(x_j) - 1

        # Note: The formulation actually uses x_i + sum of ALL x_j (including x_i)
        # So f_i = x_i + sum(x) - (n+1) for i = 1 to n-1
        sum_y = jnp.sum(y)

        # First n-1 residuals
        residuals_n_minus_1 = y[:-1] + sum_y - n_plus_1

        # For the last residual, we have a product of all variables minus 1
        res_n = jnp.prod(y) - 1.0

        # Concatenate all residuals
        residuals = jnp.concatenate([residuals_n_minus_1, jnp.array([res_n])])

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
        # The solution has all components equal to 1 except the last
        # which solves the product equation
        # For the Brown almost linear problem, the solution is approximately all ones
        return jnp.ones(self.n, dtype=jnp.float64)

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
