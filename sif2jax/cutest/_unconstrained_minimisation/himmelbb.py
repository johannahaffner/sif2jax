import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class HIMMELBB(AbstractUnconstrainedMinimisation):
    """
    HIMMELBB problem.

    A 2 variables problem by Himmelblau.

    Source: problem 27 in
    D.H. Himmelblau,
    "Applied nonlinear programming",
    McGraw-Hill, New-York, 1972.

    See Buckley#77 (p. 62)

    SIF input: Ph. Toint, Dec 1989.

    classification OUR2-AN-2-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        del args
        x1, x2 = y

        # Element H function calculation
        r1 = x1 * x2
        r2 = 1.0 - x1
        r3 = 1.0 - x2 - x1 * r2**5
        h_element = r1 * r2 * r3

        # Group L2 type
        # L2 group function: gvar^2
        return h_element * h_element

    def y0(self):
        return jnp.array([-1.2, 1.0])

    def args(self):
        return None

    def expected_result(self):
        # Solution not provided in SIF file
        return None

    def expected_objective_value(self):
        # Solution value is 0.0
        return jnp.array(0.0)
