import jax
import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# TODO: This implementation requires human review and verification against
# another CUTEst interface
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
        return jnp.full(self.n, -1.0)

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


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class BROYDN7D(AbstractUnconstrainedMinimisation):
    """A seven diagonal variant of the Broyden tridiagonal system.

    Features a band far away from the diagonal.

    Source: Ph.L. Toint,
    "Some numerical results using a sparse matrix updating formula in
    unconstrained optimization",
    Mathematics of Computation, vol. 32(114), pp. 839-852, 1978.

    See also Buckley#84
    SIF input: Ph. Toint, Dec 1989.

    Classification: OUR2-AN-V-0
    """

    n: int = 5000  # Dimension of the problem (should be even)

    def objective(self, y, args):
        del args
        n = self.n
        half_n = n // 2

        # Compute g terms (tridiagonal structure)
        g_terms = jnp.zeros(n)

        # g₁ = -2x₂ + 1 + (3-2x₁)x₁
        g_terms = g_terms.at[0].set(-2.0 * y[1] + 1.0 + (3.0 - 2.0 * y[0]) * y[0])

        # gₙ = -xₙ₋₁ + 1 + (3-2xₙ)xₙ
        g_terms = g_terms.at[n - 1].set(
            -y[n - 2] + 1.0 + (3.0 - 2.0 * y[n - 1]) * y[n - 1]
        )

        # gᵢ = 1 - xᵢ₋₁ - 2xᵢ₊₁ + (3-2xᵢ)xᵢ for i = 2,...,N-1
        # (1-indexed becomes 1,...,N-2 in 0-indexed)
        if n > 2:
            middle_indices = jnp.arange(1, n - 1)
            g_terms = g_terms.at[middle_indices].set(
                1.0
                - y[middle_indices - 1]
                - 2.0 * y[middle_indices + 1]
                + (3.0 - 2.0 * y[middle_indices]) * y[middle_indices]
            )

        # Compute s terms (distant band)
        # sᵢ = xᵢ + x_{i+N/2} for i = 1,...,N/2
        # (1-indexed becomes 0,...,N/2-1 in 0-indexed)
        s_terms = y[:half_n] + y[half_n:]

        # Objective: sum of |gᵢ|^(7/3) + |sᵢ|^(7/3)
        g_objective = jnp.sum(jnp.abs(g_terms) ** (7.0 / 3.0))
        s_objective = jnp.sum(jnp.abs(s_terms) ** (7.0 / 3.0))

        return g_objective + s_objective

    def y0(self):
        # Initial values from SIF file (all 1.0)
        return jnp.ones(self.n)

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution is not explicitly provided in the SIF file
        return None

    def expected_objective_value(self):
        # According to the SIF file comment (line 111),
        # the optimal objective value is 1.2701
        return jnp.array(1.2701)


# TODO: Human review needed
# Attempts made: Dynamic slicing in vmap causes JAX errors
# Suspected issues: Complex banded matrix structure requires careful vectorization
# without dynamic indexing
# Additional resources needed: Alternative vectorization approach for banded matrix
# operations
class BROYDNBDLS(AbstractUnconstrainedMinimisation):
    """Broyden banded system of nonlinear equations in least square sense.

    Source: problem 31 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#73 and Toint#18
    SIF input: Ph. Toint, Dec 1989.
    Least-squares version: Nick Gould, Oct 2015

    Classification: SUR2-AN-V-0
    """

    n: int = 5000  # Dimension of the problem
    kappa1: float = 2.0  # Parameter
    kappa2: float = 5.0  # Parameter
    kappa3: float = 1.0  # Parameter
    lb: int = 5  # Lower bandwidth
    ub: int = 1  # Upper bandwidth

    def objective(self, y, args):
        del args
        n = self.n
        k1 = self.kappa1
        k2 = self.kappa2
        k3 = self.kappa3
        lb = self.lb
        ub = self.ub

        # Define compute_residual function for different regions
        def compute_upper_left_residual(i):
            # For indices 0 to lb-1
            term = (
                k1 * y[i]
                - k3 * jnp.sum(y[:i])
                - k3 * jnp.sum(y[i + 1 : jnp.minimum(i + ub + 1, n)])
            )
            return term - k2 * y[i] ** 3

        def compute_middle_residual(i):
            # For indices lb to n-ub-1
            term = (
                k1 * y[i]
                - k3 * jnp.sum(y[i - lb : i])
                - k3 * jnp.sum(y[i + 1 : i + ub + 1])
            )
            return term - k2 * y[i] ** 2

        def compute_lower_right_residual(i):
            # For indices n-ub to n-1
            term = k1 * y[i] - k3 * jnp.sum(y[i - lb : i]) - k3 * jnp.sum(y[i + 1 : n])
            return term - k2 * y[i] ** 3

        # Apply vmap to each region
        upper_indices = jnp.arange(lb)
        middle_indices = jnp.arange(lb, n - ub)
        lower_indices = jnp.arange(n - ub, n)

        upper_residuals = jax.vmap(compute_upper_left_residual)(upper_indices)
        middle_residuals = jax.vmap(compute_middle_residual)(middle_indices)
        lower_residuals = jax.vmap(compute_lower_right_residual)(lower_indices)

        # Combine all residuals
        all_residuals = jnp.concatenate(
            [upper_residuals, middle_residuals, lower_residuals]
        )

        # Return the sum of squared residuals
        return jnp.sum(all_residuals**2)

    def y0(self):
        # Initial values from SIF file (all 1.0)
        return jnp.ones(self.n)

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution is not explicitly provided in the SIF file
        return None

    def expected_objective_value(self):
        # According to the SIF file comment (line 212),
        # the optimal objective value is 0.0
        return jnp.array(0.0)


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class BRYBND(AbstractUnconstrainedMinimisation):
    """Broyden banded system of nonlinear equations.

    Source: problem 31 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#73 (p. 41) and Toint#18
    SIF input: Ph. Toint, Dec 1989.

    Classification: SUR2-AN-V-0
    """

    n: int = 5000  # Dimension of the problem
    kappa1: float = 2.0  # Parameter
    kappa2: float = 5.0  # Parameter
    kappa3: float = 1.0  # Parameter
    lb: int = 5  # Lower bandwidth
    ub: int = 1  # Upper bandwidth

    def objective(self, y, args):
        del args
        n = self.n
        k1 = self.kappa1
        k2 = self.kappa2
        k3 = self.kappa3
        lb = self.lb
        ub = self.ub

        # Vectorized computation without dynamic slicing
        all_residuals = []

        # Upper left region: indices 0 to lb-1
        for i in range(lb):
            sum1 = jnp.sum(y[:i]) if i > 0 else 0.0
            end_idx = min(i + ub + 1, n)
            sum2 = jnp.sum(y[i + 1 : end_idx]) if i + 1 < end_idx else 0.0
            term = k1 * y[i] - k3 * sum1 - k3 * sum2
            residual = term - k2 * y[i] ** 3
            all_residuals.append(residual)

        # Middle region: indices lb to n-ub-1
        for i in range(lb, n - ub):
            sum1 = jnp.sum(y[i - lb : i])
            sum2 = jnp.sum(y[i + 1 : i + ub + 1])
            term = k1 * y[i] - k3 * sum1 - k3 * sum2
            residual = term - k2 * y[i] ** 2
            all_residuals.append(residual)

        # Lower right region: indices n-ub to n-1
        for i in range(n - ub, n):
            sum1 = jnp.sum(y[i - lb : i])
            sum2 = jnp.sum(y[i + 1 : n]) if i + 1 < n else 0.0
            term = k1 * y[i] - k3 * sum1 - k3 * sum2
            residual = term - k2 * y[i] ** 3
            all_residuals.append(residual)

        residuals = jnp.array(all_residuals)

        # Return the sum of squared residuals
        return jnp.sum(residuals**2)

    def y0(self):
        # Initial values from SIF file (all 1.0)
        return jnp.ones(self.n)

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution is not explicitly provided in the SIF file
        return None

    def expected_objective_value(self):
        # According to the SIF file comment (line 213),
        # the optimal objective value is 0.0
        return jnp.array(0.0)
