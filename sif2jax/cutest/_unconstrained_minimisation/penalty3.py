"""A penalty problem by Gill, Murray and Pitfield.

It has a dense Hessian matrix.

Source:  problem 114 (p. 81) in
A.R. Buckley,
"Test functions for unconstrained minimization",
TR 1989CS-3, Mathematics, statistics and computing centre,
Dalhousie University, Halifax (CDN), 1989.

SIF input: Nick Gould, Dec 1990.

classification OUR2-AY-V-0
"""

import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class PENALTY3(AbstractUnconstrainedMinimisation):
    """A penalty problem by Gill, Murray and Pitfield."""

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})
    n: int = 200  # Default to n=200 (N/2=100)

    @property
    def y0(self):
        """Initial guess: (1, -1, 1, -1, ..., 1, -1)."""
        x = jnp.ones(self.n)
        # Set every other element to -1
        x = x.at[1::2].set(-1.0)
        return x

    @property
    def args(self):
        """No additional arguments."""
        return None

    def objective(self, y, args):
        """Compute the objective function."""
        del args  # Not used

        x = y
        n = self.n
        n_half = n // 2
        rn = float(n)
        a = 0.001

        # Initialize objective value
        obj = 0.0

        # A group: constant -a
        obj += -a

        # REXP group: sum over i=1 to n-2
        # Vectorized: u1 = x[i] + 2.0 * x[i+1] + 10.0 * x[i+2]
        u1 = x[:-2] + 2.0 * x[1:-1] + 10.0 * x[2:]
        u3 = x[n - 1]
        u1m1 = u1 - 1.0
        rexp_sum = jnp.sum(a * u1m1 * u1m1 * jnp.exp(u3))
        obj += rexp_sum

        # SEXP group: sum over j=1 to n-2
        # Vectorized: u2 = 2.0 * x[j] + x[j+1]
        u2 = 2.0 * x[:-2] + x[1:-1]
        u3 = x[n - 2]
        u2m3 = u2 - 3.0
        sexp_sum = jnp.sum(a * u2m3 * u2m3 * jnp.exp(u3))
        obj += sexp_sum

        # RS group: double sum over i,j=1 to n-2
        # Vectorized using outer product
        u1 = x[:-2] + 2.0 * x[1:-1] + 10.0 * x[2:]
        u2 = 2.0 * x[:-2] + x[1:-1]
        u1m1 = u1 - 1.0
        u2m3 = u2 - 3.0
        # Outer product creates (n-2) x (n-2) matrix
        rs_sum = jnp.sum(a * jnp.outer(u1m1 * u1m1, u2m3 * u2m3))
        obj += rs_sum

        # T**2 group: double sum over i,j=1 to n
        # Vectorized using outer product
        x2mrn = x * x - rn
        t_sum = jnp.sum(jnp.outer(x2mrn, x2mrn))
        obj += t_sum

        # U group: sum over i=1 to n/2
        xim1 = x[:n_half] - 1.0
        u_sum = jnp.sum(xim1 * xim1)
        obj += u_sum

        return obj

    @property
    def expected_result(self):
        """Expected optimal solution."""
        # The expected objective value is approximately 0.001
        # No specific expected result provided in SIF
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value is approximately 0.001."""
        return jnp.array(0.001)
