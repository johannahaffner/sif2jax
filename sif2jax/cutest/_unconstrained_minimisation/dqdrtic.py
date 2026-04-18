import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractUnconstrainedMinimisation


class DQDRTIC(AbstractUnconstrainedMinimisation):
    """A simple diagonal quadratic optimization test problem.

    This problem is a sum of N-2 squared terms, where each term involves three variables
    with specific coefficients.

    Source: problem 22 in
    Ph. L. Toint,
    "Test problems for partially separable optimization and results
    for the routine PSPMIN",
    Report 83/4, Department of Mathematics, FUNDP (Namur, B), 1983.

    SIF input: Ph. Toint, Dec 1989.

    Classification: QUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 5000  # Default dimension (other options: 10, 50, 100, 500, 1000)

    def objective(self, y, args):
        del args

        # From AMPL model: sum {i in 1..N-2} (100*x[i+1]^2+100*x[i+2]^2+x[i]^2)
        # Converting to 0-based indexing: i from 0 to n-3

        # Compute terms for indices 0 to n-3 using slices
        n = self.n
        return jnp.sum(
            y[: n - 2] ** 2 + 100.0 * y[1 : n - 1] ** 2 + 100.0 * y[2:n] ** 2
        )

    @property
    def y0(self):
        # Starting point from the SIF file: all variables = 3.0
        return inexact_asarray(jnp.full(self.n, 3.0))

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        # The optimal solution is all variables = 0
        return jnp.zeros(self.n)

    @property
    def expected_objective_value(self):
        # The minimum objective value is 0.0
        return jnp.array(0.0)
