import equinox as eqx
import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class MSQRTB(AbstractNonlinearEquations):
    """The dense matrix square root problem by Nocedal and Liu (Case 1).

    Source: problem 204 (p. 93) in
    A.R. Buckley,
    "Test functions for unconstrained minimization",
    TR 1989CS-3, Mathematics, statistics and computing centre,
    Dalhousie University, Halifax (CDN), 1989.

    SIF input: Ph. Toint, Dec 1989.

    classification NQR2-AN-V-V
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Problem dimension (must be at least 3)
    p: int = eqx.field(default=32, init=False)

    @property
    def n(self):
        """Number of variables."""
        return self.p * self.p

    @property
    def m(self):
        """Number of equations."""
        return self.p * self.p

    def _compute_matrices(self):
        """Compute the B matrix and A = B * B."""
        p = self.p

        # Build B matrix using vectorized operations
        # k values go from 1 to p*p
        k_vals = jnp.arange(1, p * p + 1, dtype=jnp.float64)
        k2_vals = k_vals * k_vals
        B_flat = jnp.sin(k2_vals)
        B = B_flat.reshape((p, p))

        # Set B(3,1) = 0.0 (using 0-based indexing: B[2,0])
        if p >= 3:
            B = B.at[2, 0].set(0.0)

        # Compute A = B * B
        A = B @ B

        return B, A

    def residual(self, y, args):
        """Compute the residuals.

        The problem is to find X such that X * X = A.
        The residuals are the elements of (X * X - A).
        """
        del args
        p = self.p

        # Reshape y into a p x p matrix
        X = y.reshape((p, p))

        # Get the target matrix A
        _, A = self._compute_matrices()

        # Compute X * X - A
        residual_matrix = X @ X - A

        # Flatten to vector form
        return residual_matrix.flatten()

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
        """Expected optimal objective value."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)

    def constraint(self, y):
        """Returns the residuals as equality constraints."""
        return self.residual(y, self.args), None

    @property
    def bounds(self) -> tuple[jnp.ndarray, jnp.ndarray] | None:
        """No bounds for this problem."""
        return None
