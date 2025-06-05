import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# TODO: needs human review
class ARGLINA(AbstractUnconstrainedMinimisation):
    """ARGLINA function.

    Variable dimension full rank linear problem.

    Source: Problem 32 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#80 (with different N and M)
    SIF input: Ph. Toint, Dec 1989.

    Classification: SUR2-AN-V-0
    """

    n: int = 200  # SIF file suggests 10, 50, 100, or 200
    m: int = 400  # SIF file suggests m >= n and values like 20, 100, 200, or 400

    def objective(self, y, args):
        del args
        n = self.n
        m = self.m

        # Based on the AMPL model in arglina.mod
        # First n residuals: for i in 1..N
        # (sum{j in 1..i-1} -2*x[j]/M) + x[i]*(1-2/M) +
        # (sum {j in i+1..N} -2*x[j]/M) - 1

        # Sum of all x values scaled by -2/m
        total_sum = jnp.sum(y) * (-2.0 / m)

        # First n residuals
        # Each residual i = total_sum + y[i] * (1 - (-2/m)) - 1
        # = total_sum + y[i] * (1 + 2/m) - 1
        # = total_sum + y[i] - 1
        first_n_residuals = total_sum + y - 1.0

        # Remaining m-n residuals: all equal to total_sum - 1
        remaining_residuals = jnp.full(m - n, total_sum - 1.0)

        # Combine all residuals
        residuals = jnp.concatenate([first_n_residuals, remaining_residuals])

        # Sum of squares of residuals
        return jnp.sum(residuals**2)

    def y0(self):
        # Initial value of 1.0 as specified in the SIF file
        return jnp.ones(self.n)

    def args(self):
        return None

    def expected_result(self):
        return None

    def expected_objective_value(self):
        return None


# TODO: needs human review
class ARGLINB(AbstractUnconstrainedMinimisation):
    """ARGLINB function.

    Variable dimension rank one linear problem.

    Source: Problem 33 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#93 (with different N and M)
    SIF input: Ph. Toint, Dec 1989.

    Classification: SUR2-AN-V-0
    """

    n: int = 200  # SIF file suggests 10, 50, 100, or 200
    m: int = 400  # SIF file suggests m >= n and values like 20, 100, 200, or 400

    def objective(self, y, args):
        del args
        n = self.n
        m = self.m

        # Based on AMPL model: each residual g_i = sum_j (i*j) * x_j - 1.0
        # This can be computed as a matrix-vector product
        # Create matrix A where A[i,j] = i*j (using 1-based indices)
        i_indices = jnp.arange(1, m + 1)[:, None]  # Shape (m, 1)
        j_indices = jnp.arange(1, n + 1)[None, :]  # Shape (1, n)
        A = i_indices * j_indices  # Shape (m, n)

        # Compute residuals as A @ y - 1
        residuals = A @ y - 1.0

        # Sum of squares of residuals
        return jnp.sum(residuals**2)

    def y0(self):
        # Initial value of 1.0 as specified in the SIF file
        return jnp.ones(self.n)

    def args(self):
        return None

    def expected_result(self):
        return None

    def expected_objective_value(self):
        # The SIF file comments mention:
        # *LO SOLTN(10)          4.6341D+00
        # *LO SOLTN(50)          24.6268657
        # *LO SOLTN(100)         49.6259352
        # But no value for n=200 is provided
        return None


# TODO: needs human review
class ARGLINC(AbstractUnconstrainedMinimisation):
    """ARGLINC function.

    Variable dimension rank one linear problem, with zero rows and columns.

    Source: Problem 34 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#101 (with different N and M)
    SIF input: Ph. Toint, Dec 1989.

    Classification: SUR2-AN-V-0
    """

    n: int = 200  # SIF file suggests 10, 50, 100, or 200
    m: int = 400  # SIF file suggests m >= n and values like 20, 100, 200, or 400

    def objective(self, y, args):
        del args
        n = self.n
        m = self.m

        # Based on AMPL model: 2 + sum {i in 2..M-1}
        # (sum {j in 2..N-1} x[j]*j*(i-1) - 1.0)^2
        # Note: AMPL uses 1-based indexing

        # Middle residuals: for i from 2 to M-1 (AMPL), which is 1 to M-2 (0-based)
        # g_i = sum_j x[j]*j*(i-1) - 1.0 for j from 2 to N-1 (AMPL),
        # which is 1 to N-2 (0-based)

        # i-1 where i ranges from 2 to M-1 means (i-1) ranges from 1 to M-2
        i_values = jnp.arange(1, m - 1)[:, None]  # Shape (m-2, 1), values 1 to m-2
        # j ranges from 2 to N-1 in AMPL (1-based), so j values are 2 to N-1
        j_values = jnp.arange(2, n)[None, :]  # Shape (1, n-2), values 2 to n-1
        A = i_values * j_values  # Shape (m-2, n-2)

        # Extract x[j] for j from 2 to N-1 (AMPL), which is indices 1 to n-2 (0-based)
        y_middle = y[1 : n - 1]

        # Compute middle residuals
        middle_residuals = A @ y_middle - 1.0

        # Sum of squares plus the constant term
        return 2.0 + jnp.sum(middle_residuals**2)

    def y0(self):
        # Initial value of 1.0 as specified in the SIF file
        return jnp.ones(self.n)

    def args(self):
        return None

    def expected_result(self):
        return None

    def expected_objective_value(self):
        # The SIF file comments mention:
        # *LO SOLTN(10)           6.13513513
        # *LO SOLTN(50)           26.1269035
        # *LO SOLTN(100)          26.1269
        # But no value for n=200 is provided
        return None
