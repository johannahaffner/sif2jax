import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class NONDIA(AbstractUnconstrainedMinimisation):
    """
    NONDIA problem.

    The Shanno nondiagonal extension of Rosenbrock function.

    Source:
    D. Shanno,
    " On Variable Metric Methods for Sparse Hessians II: the New
    Method",
    MIS Tech report 27, University of Arizona (Tucson, UK), 1978.

    See also Buckley #37 (p. 76) and Toint #15.

    SIF input: Ph. Toint, Dec 1989.

    classification SUR2-AN-V-0

    TODO: Human review needed
    Attempts made: Multiple interpretations of SCALE factor in SIF
    Suspected issues: Incorrect understanding of how SCALE interacts with group
                      coefficients
    The objective/gradient values are off by a factor of ~10,000
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})
    n: int = 5000

    def objective(self, y, args):
        del args
        n = self.n

        # First squared term
        # Group SQ(1) has X(1) with coefficient 1.0 and constant -1.0
        # L2 squares: (x[0] - 1)^2
        sq1 = (y[0] - 1.0) ** 2

        # Remaining squared terms for i=2 to n
        # Groups SQ(i) for i>=2 all have x[0] with coefficient 1.0,
        # plus element -x[i-1]^2, scaled by 0.01
        if n > 1:
            # Elements are -x[i-1]^2 for i=2 to n (0-based: indices 0 to n-2)
            elements = -(y[0 : n - 1] ** 2)
            # Group values before squaring: x[0] + element
            group_values = y[0] + elements
            # L2 squares, then scale by 0.01
            sq_terms = 0.01 * group_values**2
            sq_sum = jnp.sum(sq_terms)
        else:
            sq_sum = 0.0

        return sq1 + sq_sum

    def y0(self):
        # Starting point: all variables at -1.0
        return -jnp.ones(self.n)

    def args(self):
        return None

    def expected_result(self):
        # Solution not provided in detail in SIF file
        # But noted that least square problems are bounded below by zero
        # and solution value is 0.0
        return None

    def expected_objective_value(self):
        # Solution value
        return jnp.array(0.0)

    def num_variables(self):
        return self.n
