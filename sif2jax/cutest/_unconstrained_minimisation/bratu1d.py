import equinox as eqx
import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class BRATU1D(AbstractUnconstrainedMinimisation):
    """Bratu's problem in one dimension, according to Osborne.

    Source: Problem 121 (p. 99) in
    A.R. Buckley,
    "Test functions for unconstrained minimization",
    TR 1989CS-3, Mathematics, statistics and computing centre,
    Dalhousie University, Halifax (CDN), 1989.

    SIF input: Ph. Toint, Dec 1989.

    classification OXR2-MN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Problem parameter
    lambda_param: float = -3.4

    # Number of variables (must be odd)
    n: int = eqx.field(default=5001, init=False)

    def objective(self, y, args):
        """Compute the objective function (vectorized)."""
        n = self.n
        h = 1.0 / (n + 1)
        lambda_h = self.lambda_param * h
        two_lambda_h = 2.0 * lambda_h
        two_over_h = 2.0 / h

        # Variables include fixed boundary values x[0] = 0, x[n+1] = 0
        # but we only work with interior points x[1] to x[n]
        x = jnp.concatenate([jnp.array([0.0]), y, jnp.array([0.0])])

        # Vectorized computation of all terms
        # GA terms: 2/h * x[i]^2 for i=1 to n
        ga_terms = two_over_h * y * y

        # GB terms: -2/h * x[i] * x[i-1] for i=1 to n
        gb_terms = -two_over_h * x[1:-1] * x[:-2]

        # GC terms: BRA(x[i], x[i+1]) for i=0 to n
        # Compute differences
        d = x[1:] - x[:-1]

        # Compute exponentials
        exp_x = jnp.exp(x)

        # BRA element: (exp(y) - exp(x)) / (y - x)
        # Use where to handle the case d == 0 (though it shouldn't happen)
        bra_vals = jnp.where(
            d != 0.0,
            (exp_x[1:] - exp_x[:-1]) / d,
            exp_x[:-1],  # limit as d->0 is exp(x)
        )

        gc_terms = two_lambda_h * bra_vals

        # Sum all contributions
        obj = jnp.sum(ga_terms) + jnp.sum(gb_terms) + jnp.sum(gc_terms)

        return obj

    @property
    def y0(self):
        """Starting point."""
        n = self.n
        h = 1.0 / (n + 1)

        # Vectorized computation
        i_values = jnp.arange(1, n + 1)
        x0 = -0.1 * h * (i_values * i_values)

        return x0

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        """Expected solution (not provided)."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value for n=5001."""
        # From the SIF file comments:
        # n=11: -8.49454553
        # n=75: -8.51831187
        # n=101: -8.51859
        # n=501: -8.51892
        # n=1001: -8.51893
        # For n=5001, we can extrapolate
        return jnp.array(-8.51893)
