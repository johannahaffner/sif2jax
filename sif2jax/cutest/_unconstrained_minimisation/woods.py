"""The extended Woods problem."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class WOODS(AbstractUnconstrainedMinimisation):
    """The extended Woods problem.

    This problem is a sum of n/4 sets of 6 terms, each of which is
    assigned its own group. For a given set i, the groups are
    A(i), B(i), C(i), D(i), E(i) and F(i). Groups A(i) and C(i) contain 1
    nonlinear element each, denoted Y(i) and Z(i).

    The problem dimension is defined from the number of these sets.
    The number of problem variables is then 4 times larger.

    This version uses a slightly unorthodox expression of Woods
    function as a sum of squares (see Buckley)

    Source: problem 14 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Toint#27, Buckley#17 (p. 101), Conn, Gould, Toint#7

    SIF input: Ph. Toint, Dec 1989.

    classification SUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    ns: int = 1000  # Number of sets (= n/4), default from SIF
    n: int = 4000  # Number of variables

    def objective(self, y: Any, args: Any) -> Any:
        """Compute the objective function.

        This implements the Woods function as a sum of squares.
        For each set of 4 variables, we have:
        f = 100*(x2 - x1^2)^2 + (1 - x1)^2 + 90*(x4 - x3^2)^2 + (1 - x3)^2
            + 10.1*((x2 - 1)^2 + (x4 - 1)^2) + 19.8*(x2 - 1)*(x4 - 1)
        """
        # Reshape into sets of 4 variables
        y_reshaped = y.reshape(-1, 4)

        # Extract each variable type
        x1 = y_reshaped[:, 0]
        x2 = y_reshaped[:, 1]
        x3 = y_reshaped[:, 2]
        x4 = y_reshaped[:, 3]

        # Compute all terms vectorized
        term1 = 100.0 * (x2 - x1**2) ** 2
        term2 = (1.0 - x1) ** 2
        term3 = 90.0 * (x4 - x3**2) ** 2
        term4 = (1.0 - x3) ** 2
        term5 = 10.1 * ((x2 - 1.0) ** 2 + (x4 - 1.0) ** 2)
        term6 = 19.8 * (x2 - 1.0) * (x4 - 1.0)

        # Sum all terms for all sets
        return jnp.sum(term1 + term2 + term3 + term4 + term5 + term6)

    @property
    def y0(self):
        # Starting point
        x0 = jnp.zeros(self.n)
        # Set odd indices (0-based) to -3.0 and even to -1.0
        x0 = x0.at[0::2].set(-3.0)
        x0 = x0.at[1::2].set(-1.0)
        return x0

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        # Optimal solution has all variables equal to 1.0
        return jnp.ones(self.n)

    @property
    def expected_objective_value(self):
        return jnp.array(0.0)
