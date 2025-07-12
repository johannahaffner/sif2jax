import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class POWELLBS(AbstractNonlinearEquations):
    """POWELLBS problem - Powell badly scaled problem.

    This problem is a sum of n-1 sets of 2 groups, both involving
    nonlinear elements and being of the least square type.
    Its Hessian matrix is tridiagonal.

    Source: Problem 3 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Toint#34, Buckley#22 (p. 82).

    SIF input: Ph. Toint, Dec 1989.

    Classification: NOR2-AN-2-2
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return 2  # Default size

    def num_residuals(self):
        """Number of residuals."""
        # 2*(n-1) residuals
        return 2 * (self.n - 1)

    def residual(self, y, args):
        """Compute the residuals."""
        del args
        x = y

        # Vectorized computation for residuals
        # For i = 1 to n-1:
        # A(i) = 10000 * x(i) * x(i+1) - 1 = 0
        # B(i) = exp(-x(i)) + exp(-x(i+1)) - 1.0001 = 0

        # Extract consecutive pairs
        x_i = x[:-1]  # x[0] to x[n-2]
        x_ip1 = x[1:]  # x[1] to x[n-1]

        # Compute A residuals
        a = 10000.0 * x_i * x_ip1 - 1.0

        # Compute B residuals
        b = jnp.exp(-x_i) + jnp.exp(-x_ip1) - 1.0001

        # Interleave A and B residuals
        residuals = jnp.stack([a, b], axis=1).ravel()

        return residuals

    @property
    def y0(self):
        """Initial guess."""
        x0 = jnp.zeros(self.n)
        x0 = x0.at[0].set(0.0)
        x0 = x0.at[1].set(1.0)
        return x0

    @property
    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return None

    def expected_objective_value(self):
        """Expected optimal objective value."""
        return jnp.array(0.0)

    def constraint(self, y):
        """Returns the residuals as equality constraints."""
        return self.residual(y, self.args), None

    @property
    def bounds(self) -> tuple[jnp.ndarray, jnp.ndarray] | None:
        """No bounds for this problem."""
        return None
