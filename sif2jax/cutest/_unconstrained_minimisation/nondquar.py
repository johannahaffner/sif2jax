import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class NONDQUAR(AbstractUnconstrainedMinimisation):
    """
    NONDQUAR problem.

    A nondiagonal quartic test problem.

    This problem has an arrow-head type Hessian with a tridiagonal
    central part and a border of width 1.
    The Hessian is singular at the solution.

    Source: problem 57 in
    A.R. Conn, N.I.M. Gould, M. Lescrenier and Ph.L. Toint,
    "Performance of a multi-frontal scheme for partially separable
    optimization"
    Report 88/4, Dept of Mathematics, FUNDP (Namur, B), 1988.

    SIF input: Ph. Toint, Dec 1989.

    classification OUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})
    n: int = 5000

    def objective(self, y, args):
        del args
        n = self.n

        # First n-2 groups: (x[i] + x[i+1] + x[n-1])^4 for i=0 to n-3
        if n > 2:
            # Vectorized computation
            x_i = y[0 : n - 2]
            x_ip1 = y[1 : n - 1]
            x_n = y[n - 1]
            group_values = x_i + x_ip1 + x_n
            l4_terms = group_values**4
            l4_sum = jnp.sum(l4_terms)
        else:
            l4_sum = 0.0

        # Last two groups use L2 (squared) type
        # L(N-1): (x[0] - x[1])^2
        ln_1 = (y[0] - y[1]) ** 2

        # L(N): (x[n-2] - x[n-1])^2
        ln = (y[n - 2] - y[n - 1]) ** 2

        return l4_sum + ln_1 + ln

    def y0(self):
        # Starting from (1, -1, 1, -1, ... )
        n = self.n
        y0 = jnp.ones(n)
        y0 = y0.at[1::2].set(-1.0)
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
