import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class LOGHAIRY(AbstractUnconstrainedMinimisation):
    """
    LOGHAIRY problem.

    A more difficult variant of the HAIRY problem in two variables.
    It is defined by a logarithmic transformation of the HAIRY surface,
    which is defined by this function has a large number of relatively
    sharp hills between which a valley leads to the minimizer.
    This problem contains a large number of saddle points.

    The problem is nonconvex.

    Source:
    Ph. Toint, private communication,

    SIF input: Ph. Toint, April 1997.

    classification OUR2-AN-2-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        del args
        x1, x2 = y

        # Parameters
        hlength = 30.0
        cslope = 100.0

        # Element parameters
        dens = 7.0
        smooth = 0.01

        # FUR element
        dv1 = dens * x1
        dv2 = dens * x2
        s1sq = jnp.sin(dv1) ** 2
        c2sq = jnp.cos(dv2) ** 2
        fur_value = s1sq * c2sq

        # DCUP element (2D diagonal cup)
        v = x1 - x2
        vsq = v * v
        arg = smooth + vsq
        dcup_value = jnp.sqrt(arg)

        # 1CUP element (1D cup)
        v_1d = x1
        vsq_1d = v_1d * v_1d
        arg_1d = smooth + vsq_1d
        cup1_value = jnp.sqrt(arg_1d)

        # Group contributions with weights
        group_sum = hlength * fur_value + cslope * dcup_value + cslope * cup1_value

        # LOG group type transformation
        s = 100.0
        return jnp.log((s + group_sum) / s)

    def y0(self):
        return jnp.array([-500.0, -700.0])

    def args(self):
        return None

    def expected_result(self):
        # Solution not provided in SIF file
        return None

    def expected_objective_value(self):
        # Solution value given in SIF file
        return jnp.array(0.1823216)
