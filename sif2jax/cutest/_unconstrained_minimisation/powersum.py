"""SCIPY global optimization benchmark example POWERSUM.

Fit: y = sum_j=1^n x_j^i + e

Source: Problem from the SCIPY benchmark set
https://github.com/scipy/scipy/tree/master/benchmarks/benchmarks/go_benchmark_functions

SIF input: Nick Gould, Jan 2020

Classification: SUR2-MN-V-0
"""

import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class POWERSUM(AbstractUnconstrainedMinimisation):
    """POWERSUM problem with n variables fitting power sums."""

    _n: int = 4
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return self._n

    @property
    def args(self):
        """Additional arguments including data points."""
        n = self.n
        m = n

        # Data values (for standard n=4 case)
        if n == 4:
            x_data = jnp.array([1.0, 2.0, 3.0, 2.0])
        else:
            # For other n, generate reasonable test data
            x_data = jnp.ones(4)
            x_data = x_data.at[0].set(1.0)
            x_data = x_data.at[1].set(2.0)
            x_data = x_data.at[2].set(3.0)
            x_data = x_data.at[3].set(2.0)

        # Compute target values y(i) = sum_j x_j^i
        y = jnp.zeros(m)
        for i in range(m):
            power = i + 1
            sum_val = 0.0
            for j in range(min(4, n)):
                sum_val += x_data[j] ** power
            y = y.at[i].set(sum_val)

        return (y,)

    @property
    def y0(self):
        """Initial guess."""
        return jnp.full(self.n, 2.0)

    def objective(self, y, args):
        """Compute the least squares objective function."""
        y_data = args[0]
        m = len(y_data)

        # Compute residuals
        residuals = jnp.zeros(m)
        for i in range(m):
            power = i + 1
            # Compute sum_j y_j^power
            sum_val = jnp.sum(y**power)
            residuals = residuals.at[i].set(sum_val - y_data[i])

        # Return sum of squared residuals
        return jnp.sum(residuals**2)

    @property
    def expected_result(self):
        """Expected optimal solution."""
        n = self.n
        if n == 4:
            return jnp.array([1.0, 2.0, 3.0, 2.0])
        else:
            # For larger n, first 4 values as in n=4 case, rest are 0
            result = jnp.zeros(n)
            result = result.at[0].set(1.0)
            result = result.at[1].set(2.0)
            result = result.at[2].set(3.0)
            result = result.at[3].set(2.0)
            return result

    @property
    def expected_objective_value(self):
        """Expected objective value at solution."""
        return jnp.array(0.0)
