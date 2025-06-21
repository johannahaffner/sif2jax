import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# TODO: human review required
class HIMMELBH(AbstractUnconstrainedMinimisation):
    """Himmelblau's HIMMELBH function.

    A 2-variable problem.

    Source: problem 33 in
    D.H. Himmelblau,
    "Applied nonlinear programming",
    McGraw-Hill, New-York, 1972.

    SIF input: Ph. Toint, Dec 1989.
    Classification: OUR2-AN-2-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        del args
        x1, x2 = y

        term1 = x1**3 - 3 * x1
        term2 = x2**2 - 2 * x2
        constant = 2

        return term1 + term2 + constant

    def y0(self):
        return jnp.array([0.0, 2.0])

    def args(self):
        return None

    def expected_result(self):
        # The solution is at x1 = 1, x2 = 1
        return jnp.array([1.0, 1.0])

    def expected_objective_value(self):
        # Evaluating at (1, 1): 1³ - 3*1 + 1² - 2*1 + 2 = 1 - 3 + 1 - 2 + 2 = -1
        return jnp.array(-1.0)
