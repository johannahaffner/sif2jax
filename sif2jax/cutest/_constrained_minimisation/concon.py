import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class CONCON(AbstractConstrainedMinimisation):
    """A small gas network problem.

    SIF input: Sybille Schachler, Oxford, August 1992.
              minor correction by Ph. Shott, Jan 1995.

    Classification: LOI2-MN-15-11
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Problem dimensions
    N = 7  # Number of pressure variables
    M = 4  # Number of flow variables

    @property
    def n(self):
        """Number of variables."""
        return self.N + 2 * self.M  # P(1:7), then Q(i),F(i) pairs

    @property
    def m(self):
        """Number of constraints."""
        return 4 + 7  # 4 PAN constraints + 7 MBAL constraints

    def objective(self, y, args):
        """Compute the objective function.

        The objective is -sum(P(i)) for i=1 to 7
        """
        del args
        # Extract P variables (first 7)
        p = y[: self.N]

        # Objective is -sum(P)
        return -jnp.sum(p)

    def constraint(self, y):
        """Implement the abstract constraint method."""
        eq, ineq = self.equality_constraints(y, self.args())
        return eq, ineq

    def equality_constraints(self, y, args):
        """Compute the equality constraints."""
        del args

        # Extract variables - pycutest uses interleaved Q,F ordering
        p = y[: self.N]  # P(1:7)
        q = jnp.zeros(self.M)
        f = jnp.zeros(self.M)

        # Q and F are interleaved: Q1, F1, Q2, F2, Q3, F3, Q4, F4
        for i in range(self.M):
            q = q.at[i].set(y[self.N + 2 * i])
            f = f.at[i].set(y[self.N + 2 * i + 1])

        # PAN constraints
        k = -0.597053452

        # Helper functions for elements
        def sqr(x):
            return x * jnp.abs(x)

        def forq(y):
            return y * jnp.abs(y) ** 0.8539

        pan = []
        # PAN1: PSQ1 - PSQ2 + K*QTO1 = 0
        pan.append(sqr(p[0]) - sqr(p[1]) + k * forq(q[0]))
        # PAN2: PSQ3 - PSQ4 + K*QTO2 = 0
        pan.append(sqr(p[2]) - sqr(p[3]) + k * forq(q[1]))
        # PAN3: PSQ4 - PSQ5 + K*QTO3 = 0
        pan.append(sqr(p[3]) - sqr(p[4]) + k * forq(q[2]))
        # PAN4: PSQ6 - PSQ7 + K*QTO4 = 0
        pan.append(sqr(p[5]) - sqr(p[6]) + k * forq(q[3]))

        # MBAL constraints
        mbal = []
        # MBAL1: Q1 - F3 = 0
        mbal.append(q[0] - f[2])
        # MBAL2: -Q1 + F1 = 0
        mbal.append(-q[0] + f[0])
        # MBAL3: Q2 - F1 = 0
        mbal.append(q[1] - f[0])
        # MBAL4: -Q2 + Q3 = -1000
        mbal.append(-q[1] + q[2] + 1000.0)
        # MBAL5: -Q3 - F2 = 0
        mbal.append(-q[2] - f[1])
        # MBAL6: Q4 + F2 = 0
        mbal.append(q[3] + f[1])
        # MBAL7: -Q4 - F4 = 0
        mbal.append(-q[3] - f[3])

        return jnp.array(pan + mbal), None

    def y0(self):
        """Initial guess."""
        y0 = jnp.zeros(self.n)

        # P variables
        y0 = y0.at[: self.N].set(965.0)

        # Q and F variables are interleaved: Q1, F1, Q2, F2, Q3, F3, Q4, F4
        y0 = y0.at[self.N].set(100.0)  # Q1
        y0 = y0.at[self.N + 1].set(1000.0)  # F1
        y0 = y0.at[self.N + 2].set(100.0)  # Q2
        y0 = y0.at[self.N + 3].set(1000.0)  # F2
        y0 = y0.at[self.N + 4].set(-100.0)  # Q3
        y0 = y0.at[self.N + 5].set(1000.0)  # F3
        y0 = y0.at[self.N + 6].set(-100.0)  # Q4
        y0 = y0.at[self.N + 7].set(1000.0)  # F4

        return y0

    def bounds(self):
        """Variable bounds."""
        lower = jnp.full(self.n, -jnp.inf)
        upper = jnp.full(self.n, jnp.inf)

        # Upper bounds on P variables
        upper = upper.at[2].set(904.73)  # P3
        upper = upper.at[4].set(904.73)  # P5
        upper = upper.at[0].set(914.73)  # P1
        upper = upper.at[6].set(914.73)  # P7

        # Upper bound on F4 (last variable in interleaved ordering)
        upper = upper.at[self.N + 2 * self.M - 1].set(400.0)  # F4

        return lower, upper

    def args(self):
        """No additional arguments."""
        return None

    def expected_result(self):
        """Expected optimal solution."""
        # Not provided in SIF file
        return None

    def expected_objective_value(self):
        """Expected optimal objective value."""
        # Not provided in SIF file
        return None
