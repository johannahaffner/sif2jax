import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class BROWNAL(AbstractUnconstrainedMinimisation):
    """Brown almost linear least squares problem.

    This problem is a sum of n least-squares groups, the last one of
    which has a nonlinear element. Its Hessian matrix is dense.

    Source: Problem 27 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#79
    SIF input: Ph. Toint, Dec 1989.

    Classification: SUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 200  # Other suggested dimensions are 10, 100, and 1000

    def objective(self, y, args):
        del args
        n = self.n

        # First n-1 groups: each group G(i) has 2*x[i] + sum_{j≠i}(x[j]) - (n+1)
        # This is equivalent to x[i] + sum_all(x[j]) - (n+1)
        sum_all = jnp.sum(y)

        # Vectorized computation for first n-1 residuals
        first_residuals = y[:-1] + sum_all - (n + 1)

        # Last group: product of components - 1
        product_residual = jnp.prod(y) - 1.0

        # Combine all residuals
        all_residuals = jnp.concatenate(
            [first_residuals, jnp.array([product_residual])]
        )

        return jnp.sum(all_residuals**2)

    def y0(self):
        # Initial values from SIF file (all 0.5)
        return jnp.full(self.n, 0.5)

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution for Brown almost linear problem is all variables = 1
        # This satisfies both constraints:
        # - Linear constraints: x[i] + sum(x) - (n+1) = 1 + n - (n+1) = 0
        # - Product constraint: prod(x) - 1 = 1 - 1 = 0
        # giving an optimal objective value of 0.0
        return jnp.ones(self.n)

    def expected_objective_value(self):
        # The optimal objective value for Brown almost linear problem is 0.0
        # when all variables equal 1. This satisfies both:
        # - Linear constraints: x[i] + sum(x) - (n+1) = 1 + n - (n+1) = 0
        # - Product constraint: prod(x) - 1 = 1 - 1 = 0
        return jnp.array(0.0)
