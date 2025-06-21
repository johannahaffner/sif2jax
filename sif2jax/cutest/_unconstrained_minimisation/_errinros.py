import jax
import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractUnconstrainedMinimisation


# TODO: needs human review
class ERRINROS(AbstractUnconstrainedMinimisation):
    """A nonlinear function similar to the chained Rosenbrock problem CHNROSNB.

    This problem is actually an error in specifying the CHNROSNB problem,
    but it has been included in the CUTEst collection as a separate problem.

    Source:
    An error in specifying problem CHNROSNB.
    SIF input: Ph. Toint, Sept 1990.

    Classification: SUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 50  # Number of variables (default 50, but can also be 10 or 25)

    def objective(self, y, args):
        del args

        # From AMPL model:
        # sum {i in 2..N} (x[i-1]-16*alpha[i]^2*x[i]^2)^2 + sum {i in 2..N} (x[i]-1.0)^2
        # Converting to 0-based indexing: i from 1 to N-1

        # Alpha values from the AMPL file
        alphas = jnp.array(
            [
                1.25,
                1.40,
                2.40,
                1.40,
                1.75,
                1.20,
                2.25,
                1.20,
                1.00,
                1.10,
                1.50,
                1.60,
                1.25,
                1.25,
                1.20,
                1.20,
                1.40,
                0.50,
                0.50,
                1.25,
                1.80,
                0.75,
                1.25,
                1.40,
                1.60,
                2.00,
                1.00,
                1.60,
                1.25,
                2.75,
                1.25,
                1.25,
                1.25,
                3.00,
                1.50,
                2.00,
                1.25,
                1.40,
                1.80,
                1.50,
                2.20,
                1.40,
                1.50,
                1.25,
                2.00,
                1.50,
                1.25,
                1.40,
                0.60,
                1.50,
            ]
        )

        # Use correct number of alpha values based on problem dimension
        alphas = alphas[: self.n]

        # First sum: sum {i in 2..N} (x[i-1]-16*alpha[i]^2*x[i]^2)^2
        # In 0-based indexing: i from 1 to n-1, but alphas need correct indexing
        def compute_first_term(i):
            alpha_i = alphas[
                i + 1
            ]  # alphas[i+1] corresponds to alpha[i+2] in AMPL (1-indexed)
            return (y[i] - 16.0 * alpha_i**2 * y[i + 1] ** 2) ** 2

        indices = jnp.arange(0, self.n - 1)
        first_terms = jax.vmap(compute_first_term)(indices)

        # Second sum: sum {i in 2..N} (x[i]-1.0)^2
        # In 0-based indexing: i from 1 to n-1
        second_terms = (y[1:] - 1.0) ** 2

        return jnp.sum(first_terms) + jnp.sum(second_terms)

    def y0(self):
        # Initial values from SIF file (all -1.0)
        return inexact_asarray(jnp.full(self.n, -1.0))

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


# TODO: needs human review
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
        return inexact_asarray(jnp.full(self.n, -1.0))

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
