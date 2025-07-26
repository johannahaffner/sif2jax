import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class BDEXP(AbstractUnconstrainedMinimisation):
    """BDEXP problem.

    A banded exponential problem.

    Source: Problem 56 in
    A.R. Conn, N.I.M. Gould, M. Lescrenier and Ph.L. Toint,
    "Performance of a multifrontal scheme for partially separable
    optimization",
    Report 88/4, Dept of Mathematics, FUNDP (Namur, B), 1988.

    SIF input: Ph. Toint, Dec 1989.

    classification OBR2-AY-V-0
    """

    # Default parameter
    N: int = 5000  # Number of variables

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        """Compute the objective function."""
        del args

        x = y

        # Vectorized computation of DEXP elements
        # Element A(i) uses variables X(i), X(i+1), X(i+2)
        v1 = x[:-2]  # x[0] to x[n-3]
        v2 = x[1:-1]  # x[1] to x[n-2]
        v3 = x[2:]  # x[2] to x[n-1]

        # From DEXP element:
        # U1 = V1 + V2
        # U2 = -V3 (note the negative sign from SIF: R  U2  V3  -1.0)
        u1 = v1 + v2
        u2 = -v3

        # F = exp(U1 * U2) * U1
        u1_times_u2 = u1 * u2
        exp_u1u2 = jnp.exp(u1_times_u2)

        # Sum all elements
        return jnp.sum(exp_u1u2 * u1)

    @property
    def y0(self):
        """Initial guess for variables."""
        return jnp.ones(self.N)

    @property
    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    @property
    def expected_result(self):
        """Expected optimal solution."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        # From SIF file: SOLTN = 0.0
        return jnp.array(0.0)

    @property
    def n(self):
        """Number of variables."""
        return self.N
