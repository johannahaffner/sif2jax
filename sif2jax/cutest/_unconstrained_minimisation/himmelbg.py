import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# TODO: Needs verification against another CUTEst interface
class HIMMELBG(AbstractUnconstrainedMinimisation):
    """The HIMMELBG function.

    A 2-variable problem by Himmelblau.

    Source: problem 33 in
    D.H. Himmelblau,
    "Applied nonlinear programming",
    McGraw-Hill, New-York, 1972.
    See Buckley#87 (p. 67)

    SIF input: Ph. Toint, Dec 1989.
    Classification: OUR2-AN-2-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        del args
        x1, x2 = y
        exponential = jnp.exp(-x1 - x2)
        quadratic = 2 * x1**2 + 3 * x2**2
        return exponential * quadratic

    @property
    def y0(self):
        return jnp.array([0.5, 0.5])

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        return None

    @property
    def expected_objective_value(self):
        return jnp.array(0.0)
