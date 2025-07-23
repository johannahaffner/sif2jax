"""
A nonlinear minmax problem in eleven variables.

Note: the original statement of the problem contains an inconsistent
index i. This has been replaced by 1, assuming a very common typo.
But the optimal solution of the resulting problem differs from that
quoted in the source.

Source:
E. Polak, D.H. Mayne and J.E. Higgins,
"Superlinearly convergent algorithm for min-max problems"
JOTA 69, pp. 407-439, 1991.

SIF input: Ph. Toint, Nov 1993.

classification LOR2-AN-12-10
"""

import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class POLAK3(AbstractConstrainedMinimisation):
    @property
    def name(self) -> str:
        return "POLAK3"

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 12  # 11 X variables + 1 U variable
    m: int = 10  # 10 inequality constraints

    @property
    def y0(self):
        # All variables start at 1.0
        return jnp.ones(self.n)

    @property
    def args(self):
        return None

    def objective(self, y, args):
        del args
        # Objective is simply U (the 12th variable)
        u = y[11]
        return jnp.array(u)

    def constraint(self, y):
        # Variables: X(1) to X(11) are y[0] to y[10], U is y[11]
        x = y[:11]
        u = y[11]

        # F(I) constraints: U - sum_j (1/j) * E(I,J) >= 0
        # where E(I,J) = exp((X(J) - sin((I-1) + 2*J))^2)
        # But pycutest expects -F(I) <= 0, so we return the negative

        inequality_constraints = []

        for i in range(1, 11):  # I from 1 to 10
            f_i = u
            for j in range(1, 12):  # J from 1 to 11
                # Element E(I,J) with A = I-1, B = J-1
                # From SIF lines 68-69: ZP E(I,J) A RI-1, ZP E(I,J) B RJ-1
                # But in lines 67-68, RJ-1 = J (not J-1)
                a = float(i - 1)
                b = float(j)  # B = J, not J-1
                v = x[j - 1] - jnp.sin(a + b + b)  # XX - sin(A + B + B)
                e_ij = jnp.exp(v * v)
                f_i = f_i - (1.0 / j) * e_ij
            # Pycutest expects -F(I) <= 0 instead of F(I) >= 0
            inequality_constraints.append(-f_i)

        return None, jnp.array(inequality_constraints)

    @property
    def bounds(self):
        # All variables are free (unbounded)
        return None

    @property
    def expected_result(self):
        # The optimal solution is not explicitly given in the SIF file
        return None

    @property
    def expected_objective_value(self):
        # According to the SIF file comment (line 90),
        # the optimal objective value is 5.93300252
        return jnp.array(5.93300252)
