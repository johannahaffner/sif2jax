import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class POWELLSG(AbstractUnconstrainedMinimisation):
    """
    POWELLSG problem.

    The extended Powell singular problem.
    This problem is a sum of n/4 sets of four terms, each of which is
    assigned its own group.

    Source:  Problem 13 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Toint#19, Buckley#34 (p.85)

    SIF input: Ph. Toint, Dec 1989.

    classification OUR2-AN-V-0

    TODO: Human review needed
    Attempts made: Standard interpretation of SCALE factor
    Suspected issues: Possible misinterpretation of SCALE semantics or group definitions
    The objective value is off by a factor of ~4.15
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})
    n: int = 5000

    def __check_init__(self):
        if self.n % 4 != 0:
            raise ValueError("n must be a multiple of 4 for POWELLSG")

    def objective(self, y, args):
        del args

        # Reshape x into groups of 4
        x_reshaped = y.reshape(-1, 4)
        x1 = x_reshaped[:, 0]
        x2 = x_reshaped[:, 1]
        x3 = x_reshaped[:, 2]
        x4 = x_reshaped[:, 3]

        # G(I): (x[i] + 10*x[i+1])^2
        g1 = (x1 + 10.0 * x2) ** 2

        # G(I+1): 0.2 * (x[i+2] - x[i+3])^2
        g2 = 0.2 * (x3 - x4) ** 2

        # G(I+2): (x[i+1] - 2*x[i+2])^4
        g3 = (x2 - 2.0 * x3) ** 4

        # G(I+3): 0.1 * (x[i] - x[i+3])^4
        g4 = 0.1 * (x1 - x4) ** 4

        return jnp.sum(g1 + g2 + g3 + g4)

    def y0(self):
        # Starting point: (3, -1, 0, 1) repeated
        n = self.n
        y0 = jnp.zeros(n)
        for i in range(0, n, 4):
            y0 = y0.at[i].set(3.0)
            y0 = y0.at[i + 1].set(-1.0)
            y0 = y0.at[i + 2].set(0.0)
            y0 = y0.at[i + 3].set(1.0)
        return y0

    def args(self):
        return None

    def expected_result(self):
        # Solution not provided in detail in SIF file
        return None

    def expected_objective_value(self):
        # Solution value is 0.0
        return jnp.array(0.0)

    def num_variables(self):
        return self.n
