import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class LUKSAN14(AbstractNonlinearEquations):
    """Problem 14 (chained and modified HS53) in the paper.

    L. Luksan
    Hybrid methods in large sparse nonlinear least squares
    J. Optimization Theory & Applications 89(3) 575-595 (1996)

    SIF input: Nick Gould, June 2017.

    Classification: NOR2-AN-V-V
    """

    _s: int = 32
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return 3 * self._s + 2

    @property
    def m(self):
        """Number of residuals/equations."""
        return 7 * self._s

    @property
    def y0(self):
        """Initial guess."""
        return jnp.full(self.n, -1.0)

    @property
    def args(self):
        """No additional arguments."""
        return None

    def residual(self, x, args):
        """Compute the residual vector."""
        del args  # Not used

        s = self._s

        # Extract variable indices for vectorized computation
        # Variable indices: i, i+1, i+2, i+3, i+4 for each block j
        # where i = 3*j
        j_indices = jnp.arange(s)
        i_indices = 3 * j_indices  # i = 3*j

        # Variable indices for each block
        xi_indices = i_indices  # x[i]
        xi1_indices = i_indices + 1  # x[i+1]
        xi2_indices = i_indices + 2  # x[i+2]
        xi3_indices = i_indices + 3  # x[i+3]
        xi4_indices = i_indices + 4  # x[i+4]

        # Extract variables for each block
        xi = x[xi_indices]  # x[i] for all blocks
        xi1 = x[xi1_indices]  # x[i+1] for all blocks
        xi2 = x[xi2_indices]  # x[i+2] for all blocks
        xi3 = x[xi3_indices]  # x[i+3] for all blocks
        xi4 = x[xi4_indices]  # x[i+4] for all blocks

        # Compute elements
        e_k = xi**2  # E(K) = X(I)^2
        e_k6 = xi1**2  # E(K+6) = X(I+1)^2

        # Compute all residuals of each type
        # E(K): -10*X(I+1) + 10*E(K)
        res0 = -10.0 * xi1 + 10.0 * e_k

        # E(K+1): X(I+1) + X(I+2) - 2.0
        res1 = xi1 + xi2 - 2.0

        # E(K+2): X(I+3) - 1.0
        res2 = xi3 - 1.0

        # E(K+3): X(I+4) - 1.0
        res3 = xi4 - 1.0

        # E(K+4): X(I) + 3*X(I+1)
        res4 = xi + 3.0 * xi1

        # E(K+5): X(I+2) + X(I+3) - 2*X(I+4)
        res5 = xi2 + xi3 - 2.0 * xi4

        # E(K+6): -10*X(I+4) + 10*E(K+6)
        res6 = -10.0 * xi4 + 10.0 * e_k6

        # Stack residuals in correct order for each block
        residuals = jnp.stack(
            [res0, res1, res2, res3, res4, res5, res6], axis=1
        )  # shape: (s, 7)

        # Flatten to get final residual vector
        return residuals.flatten()

    def constraint(self, y):
        """Return the constraint values as required by the abstract base class."""
        # For nonlinear equations, residuals are equality constraints
        return self.residual(y, self.args), None

    @property
    def bounds(self):
        """Bounds on variables."""
        # All variables are free
        return None

    @property
    def expected_result(self):
        """Expected optimal solution."""
        # Not provided in SIF file
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value is 0.0 for nonlinear equations."""
        return jnp.array(0.0)
