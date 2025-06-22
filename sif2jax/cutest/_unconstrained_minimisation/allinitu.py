import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# TODO: needs human review
class ALLINITU(AbstractUnconstrainedMinimisation):
    """The ALLINITU function.

    A problem with "all in it". Intended to verify that changes to LANCELOT are safe.

    Source: N. Gould, private communication.
    SIF input: Nick Gould, June 1990.

    Classification: OUR2-AY-4-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        del args
        x1, x2, x3, x4 = y

        # Based on the AMPL model in allinitu.mod
        return (
            x3
            - 1
            + x1**2
            + x2**2
            + (x3 + x4) ** 2
            + jnp.sin(x3) ** 2
            + x1**2 * x2**2
            + x4
            - 3
            + jnp.sin(x3) ** 2
            + (x4 - 1) ** 2
            + (x2**2) ** 2
            + (x3**2 + (x4 + x1) ** 2) ** 2
            + (x1 - 4 + jnp.sin(x4) ** 2 + x2**2 * x3**2) ** 2
            + jnp.sin(x4) ** 4
        )

    def y0(self):
        # Initial point is not explicitly given in the SIF file
        # Using zeros as a reasonable starting point
        return jnp.zeros(4)

    def args(self):
        return None

    def expected_result(self):
        # No expected result is given in the SIF file
        return None

    def expected_objective_value(self):
        # No expected objective value is given in the SIF file
        return None
