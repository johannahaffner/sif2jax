import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractUnconstrainedMinimisation


class BROYDN3DLS(AbstractUnconstrainedMinimisation):
    """Broyden tridiagonal system of nonlinear equations in the least square sense.

    Source: problem 30 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Toint#17 and Buckley#78.
    SIF input: Ph. Toint, Dec 1989.
    Least-squares version: Nick Gould, Oct 2015.

    Classification: SUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 5000  # Dimension of the problem
    kappa1: float = 2.0  # Parameter
    kappa2: float = 1.0  # Parameter

    def objective(self, y, args):
        del args
        n = self.n
        k1 = self.kappa1
        k2 = self.kappa2

        # Compute residuals without conditional logic
        residuals = jnp.zeros(n)

        # First residual: (3-2*x1)*x1 - 2*x2 + k2
        residuals = residuals.at[0].set((3.0 - k1 * y[0]) * y[0] - 2.0 * y[1] + k2)

        # Last residual: (3-2*xn)*xn - xn-1 + k2
        residuals = residuals.at[n - 1].set(
            (3.0 - k1 * y[n - 1]) * y[n - 1] - y[n - 2] + k2
        )

        # Middle residuals: (3-2*xi)*xi - xi-1 - 2*xi+1 + k2 for i=1 to n-2
        if n > 2:
            middle_indices = jnp.arange(1, n - 1)
            middle_residuals = (
                (3.0 - k1 * y[middle_indices]) * y[middle_indices]
                - y[middle_indices - 1]
                - 2.0 * y[middle_indices + 1]
                + k2
            )
            residuals = residuals.at[middle_indices].set(middle_residuals)

        # Return the sum of squared residuals
        return jnp.sum(residuals**2)

    def y0(self):
        # Initial values from SIF file (all -1.0)
        return inexact_asarray(jnp.full(self.n, -1.0))

    def args(self):
        return None

    def expected_result(self):
        # Set values of all components to the same value r
        # where r is approximately -k2/(n*k1)
        k2 = self.kappa2
        n = self.n
        k1 = self.kappa1
        r = -k2 / (n * k1)
        return jnp.full(self.n, r)

    def expected_objective_value(self):
        # According to the SIF file comment (line 110),
        # the optimal objective value is 0.0
        return jnp.array(0.0)
