import jax
import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractUnconstrainedMinimisation


# TODO: human review required
class HATFLDGLS(AbstractUnconstrainedMinimisation):
    """A test problem from the OPTIMA user manual.

    Least-squares version of HATFLDG.

    Source:
    "The OPTIMA user manual (issue No.8, p. 49)",
    Numerical Optimization Centre, Hatfield Polytechnic (UK), 1989.

    SIF input: Ph. Toint, May 1990.
    Least-squares version of HATFLDG.SIF, Nick Gould, Jan 2020.

    Classification: SUR2-AY-25-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        del args

        # Create residuals for each group
        # Each group is defined by G(i) = x(i) + x(13) - 1.0 for i=1...n
        residuals = y + y[12] - 1.0

        # Group element G(1) also involves A(1) with a -1.0 coefficient
        # A(1) is a 2PR element with x1 and x2, computing x1 * x2
        residuals = residuals.at[0].add(-1.0 * y[0] * y[1])

        # Groups G(2) to G(n) involve A(i) elements
        # For i=2 to n-1, A(i) is a 2PRI element with x(i), x(i-1), and x(i+1)
        # 2PRI computes (x(i) + 1.0) * (x(i-1) + 1.0 - x(i+1))

        # Use vmap instead of for-loop
        def compute_2pri_element(i):
            return (y[i] + 1.0) * (y[i - 1] + 1.0 - y[i + 1])

        indices = inexact_asarray(jnp.arange(1, len(y)) * 1)
        inner_residuals = jax.vmap(compute_2pri_element)(indices)
        residuals = residuals.at[1:-1].add(inner_residuals)

        # For i=n, A(n) is a 2PR element with x(n-1) and x(n), computing x(n-1) * x(n)
        residuals = residuals.at[-1].add(y[-2] * y[-1])

        # Objective function is sum of squared residuals
        return jnp.sum(residuals**2)

    def y0(self):
        # Initial point from SIF file (line 57)
        return inexact_asarray(jnp.ones(25))  # Hard-coded as 25 per SIF file

    def args(self):
        return None

    def expected_result(self):
        # Not provided in the SIF file
        return None

    def expected_objective_value(self):
        # Not provided in the SIF file
        return None
