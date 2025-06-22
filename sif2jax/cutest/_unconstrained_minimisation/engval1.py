import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# TODO: Needs verification against another CUTEst interface
class ENGVAL1(AbstractUnconstrainedMinimisation):
    """The ENGVAL1 function.

    This problem is a sum of 2n-2 groups, n-1 of which contain 2 nonlinear elements.

    Source: problem 31 in
    Ph.L. Toint,
    "Test problems for partially separable optimization and results
    for the routine PSPMIN",
    Report 83/4, Department of Mathematics, FUNDP (Namur, B), 1983.

    See also Buckley#172 (p. 52)
    SIF input: Ph. Toint and N. Gould, Dec 1989.

    Classification: OUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 5000  # Other dimensions suggested: 2, 50, 100, 1000

    def objective(self, y, args):
        del args
        # From AMPL model: sum {i in 1..N-1} (x[i]^2+x[i+1]^2)^2 +
        # sum {i in 1..N-1} (-4*x[i]+3.0)
        # Converting to 0-based indexing: i from 0 to N-2

        y2 = y**2
        nonlinear = jnp.sum((y2[:-1] + y2[1:]) ** 2)
        linear = jnp.sum(-4 * y[:-1] + 3.0)  # Fixed: +3.0 not -3

        return nonlinear + linear

    def y0(self):
        return jnp.full(self.n, 2.0)

    def args(self):
        return None

    def expected_result(self):
        return None

    def expected_objective_value(self):
        return jnp.array(0.0)
