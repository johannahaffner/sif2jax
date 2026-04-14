import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ..._problem import AbstractUnconstrainedMinimisation


class MSQRTBLS(AbstractUnconstrainedMinimisation):
    """The dense matrix square root problem by Nocedal and Liu (Case 1) - LS variant.

    This is a least-squares variant of problem MSQRTB.

    Source: problem 204 (p. 93) in
    A.R. Buckley,
    "Test functions for unconstrained minimization",
    TR 1989CS-3, Mathematics, statistics and computing centre,
    Dalhousie University, Halifax (CDN), 1989.

    SIF input: Ph. Toint, Dec 1989.

    classification SUR2-AN-V-V
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Problem dimension (must be at least 3)
    p: int = eqx.field(default=32, init=False)

    @property
    def n(self):
        """Number of variables."""
        return self.p * self.p

    def _compute_matrices(self):
        """Compute the B matrix and A = B * B using numpy (no JAX tracing)."""
        p = self.p

        k_vals = np.arange(1, p * p + 1, dtype=np.float64)
        B = np.sin(k_vals * k_vals).reshape((p, p))

        if p >= 3:
            B[2, 0] = 0.0

        A = B @ B

        return B, A

    def objective(self, y, args):
        """Compute the objective function.

        The objective is the sum of squared residuals: ||X*X - A||²_F
        where ||·||_F is the Frobenius norm.
        """
        del args
        p = self.p

        # Reshape y into a p x p matrix
        X = y.reshape((p, p))

        # Get the target matrix A
        _, A = self._compute_matrices()

        # Compute X * X - A
        residual_matrix = X @ X - A

        # Return the squared Frobenius norm
        return jnp.sum(residual_matrix**2)

    @property
    def y0(self):
        """Starting point."""
        p = self.p
        B, _ = self._compute_matrices()

        # Starting point is B + perturbation
        # Vectorized computation of perturbation
        k_vals = jnp.arange(1, p * p + 1, dtype=jnp.float64)
        k2_vals = k_vals * k_vals
        sk2_vals = jnp.sin(k2_vals)
        perturbations = -0.8 * sk2_vals
        perturbations_matrix = perturbations.reshape((p, p))

        X0 = B + perturbations_matrix

        return X0.flatten()

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        """Expected solution is the B matrix."""
        B, _ = self._compute_matrices()
        return B.flatten()

    @property
    def expected_objective_value(self):
        """Expected optimal objective value is 0.0."""
        return jnp.array(0.0)
