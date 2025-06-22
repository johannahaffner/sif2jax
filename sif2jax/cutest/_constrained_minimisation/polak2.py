import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class POLAK2(AbstractConstrainedMinimisation):
    """POLAK2 problem - A nonlinear minmax problem.

    A nonlinear minmax problem in ten variables.

    Source:
    E. Polak, D.H. Mayne and J.E. Higgins,
    "Superlinearly convergent algorithm for min-max problems"
    JOTA 69, pp. 407-439, 1991.

    SIF input: Ph. Toint, Nov 1993.

    Classification: LOR2-AN-11-2
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return 11  # x1, ..., x10, u

    @property
    def m(self):
        """Number of constraints."""
        return 2  # Two inequality constraints

    def objective(self, y, args):
        """Compute the objective (minimize u)."""
        del args
        u = y[10]
        return u

    def constraint(self, y):
        """Compute the constraints."""
        x = y[:10]
        u = y[10]

        # Compute the exponential terms
        def compute_exp_term(x, p):
            a = 1.0e-8 * x[0] ** 2 + (x[1] + p) ** 2
            a = a + x[2] ** 2 + 4.0 * x[3] ** 2
            a = a + x[4] ** 2 + x[5] ** 2 + x[6] ** 2
            a = a + x[7] ** 2 + x[8] ** 2 + x[9] ** 2
            return jnp.exp(a)

        # Constraints:
        # F1: -u + exp(a) with p=2.0 <= 0
        # F2: -u + exp(a) with p=-2.0 <= 0
        e1 = compute_exp_term(x, 2.0)
        e2 = compute_exp_term(x, -2.0)

        c1 = -u + e1
        c2 = -u + e2

        # Both are inequality constraints (<=)
        constraints = jnp.array([c1, c2])

        # Return as inequality constraints
        return None, constraints

    def y0(self):
        """Initial guess."""
        x0 = jnp.ones(11) * 0.1
        x0 = x0.at[0].set(100.0)  # x1 = 100.0
        return x0

    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    def bounds(self):
        """Variable bounds (all free)."""
        lower = jnp.full(11, -jnp.inf)
        upper = jnp.full(11, jnp.inf)
        return lower, upper

    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return None

    def expected_objective_value(self):
        """Expected optimal objective value."""
        return jnp.array(54.598146)
