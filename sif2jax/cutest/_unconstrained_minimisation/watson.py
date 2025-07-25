from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class WATSON(AbstractUnconstrainedMinimisation):
    """Watson problem in 12 variables.

    This function is a nonlinear least squares with 31 groups. Each
    group has 1 nonlinear and 1 linear elements.

    Source: problem 20 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#128 (p. 100).

    SIF input: Ph. Toint, Dec 1989.
    (bug fix July 2007)

    classification SUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 12  # Number of variables
    m: int = 31  # Number of groups

    def objective(self, y: Any, args: Any) -> Any:
        """Compute the objective function."""
        # For groups 1 to 29
        obj = 0.0

        for i in range(1, 30):  # Groups 1 to 29
            # t_i = i/29
            ti = i / 29.0

            # Linear part: sum of coefficients times variables
            linear_sum = 0.0
            for j in range(2, self.n + 1):  # j from 2 to n
                rj_minus_1 = j - 1
                rj_minus_2 = j - 2
                # coefficient is (j-1) * ti^(j-2)
                coeff = rj_minus_1 * (ti**rj_minus_2)
                linear_sum = linear_sum + coeff * y[j - 1]

            # Element MWSQ: weighted sum of all variables with ti^(j-1) weights
            element_sum = 0.0
            for j in range(1, self.n + 1):  # j from 1 to n
                weight = ti ** (j - 1)
                element_sum = element_sum + weight * y[j - 1]

            # MWSQ element function is -u^2 where u is the weighted sum
            element_val = -element_sum * element_sum

            # Group contribution with L2 type: (linear + element - 1)^2
            group_val = linear_sum + element_val - 1.0
            obj = obj + group_val * group_val

        # Group 30: x1 (constant is 0)
        group_30 = y[0] - 0.0
        obj = obj + group_30 * group_30

        # Group 31: x2 + MSQ(x1) - 1
        # MSQ element function is -x1^2
        element_31 = -y[0] * y[0]
        group_31 = y[1] + element_31 - 1.0
        obj = obj + group_31 * group_31

        return obj

    @property
    def y0(self):
        return jnp.zeros(self.n)

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        # From the SIF file comment: solution for n=12 is 2.27559922D-9
        # This suggests the optimal value is very close to zero
        return None  # Not provided explicitly

    @property
    def expected_objective_value(self):
        return jnp.array(2.27559922e-9)
