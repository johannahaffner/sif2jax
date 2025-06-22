import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class ERRINRSM(AbstractUnconstrainedMinimisation):
    """A variable dimension version of an incorrect version of the
    chained Rosenbrock function (ERRINROS) by Luksan et al.

    Source: problem 28 in
    L. Luksan, C. Matonoha and J. Vlcek
    Modified CUTE problems for sparse unconstrained optimization
    Technical Report 1081
    Institute of Computer Science
    Academy of Science of the Czech Republic

    SIF input: Ph. Toint, Sept 1990.
              this version Nick Gould, June, 2013

    Classification: SUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 50  # Number of variables (default 50, but can also be 10 or 25)

    def objective(self, y, args):
        del args

        # ERRINRSM formulation from SIF file:
        # For each i from 2 to N:
        #   Group SQ(I) = X(I-1) - AI * X(I)^2, where AI = 16 * (sin(i) * 1.5)^2
        #   Group B(I) = X(I) - 1.0
        # Objective = sum_{i=2}^N [SQ(I)^2 + B(I)^2]
        #           = sum_{i=2}^N [(X(I-1) - AI * X(I)^2)^2 + (X(I) - 1)^2]

        total = 0.0

        # Loop from i=2 to N (1 to n-1 in 0-based indexing)
        for i in range(1, self.n):
            # Alpha value: sin(i) * 1.5 where i is 1-based (i+1 in our 0-based loop)
            alpha_i = jnp.sin(float(i + 1)) * 1.5
            # AI = 16 * alpha_i^2 = 16 * (sin(i) * 1.5)^2
            ai = 16.0 * alpha_i**2

            # First term: (X(I-1) - AI * X(I)^2)^2
            # In 0-based indexing: (y[i-1] - ai * y[i]^2)^2
            sq_term = (y[i - 1] - ai * y[i] ** 2) ** 2

            # Second term: (X(I) - 1)^2
            # In 0-based indexing: (y[i] - 1)^2
            b_term = (y[i] - 1.0) ** 2

            total += sq_term + b_term

        return jnp.array(total)

    def y0(self):
        # Initial values from SIF file (all -1.0)
        return jnp.full(self.n, -1.0)

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution has all components equal to 1
        return jnp.ones(self.n)

    def expected_objective_value(self):
        # Solution values from the SIF file comments
        if self.n == 10:
            return jnp.array(6.69463214)
        elif self.n == 25:
            return jnp.array(18.4609060)
        elif self.n == 50:
            return jnp.array(39.9041540)
        else:
            assert False, "Invalid n value"
