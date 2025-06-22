import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


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

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

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
