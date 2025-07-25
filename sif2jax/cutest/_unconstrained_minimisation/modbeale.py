"""A variation on Beale's problem in 2 variables.

TODO: Human review needed
The test results show large discrepancies with pycutest:
- Objective values differ by factor of ~89
- Gradient values differ by factors of 100-125000
Suspected issues: Different interpretation of SCALE factor for L groups in pycutest
Additional resources needed: Clarification on how pycutest handles linear group scaling

Source: An adaptation by Ph. Toint of Problem 5 in
J.J. More', B.S. Garbow and K.E. Hillstrom,
"Testing Unconstrained Optimization Software",
ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

See also Buckley#89.
SIF input: Ph. Toint, Mar 2003.

Classification: SUR2-AN-V-0
"""

import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class MODBEALE(AbstractUnconstrainedMinimisation):
    """A variation on Beale's problem."""

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Parameters
    n_half: int = 10000  # N/2 parameter from SIF
    ALPHA: float = 50.0

    @property
    def n(self):
        """Number of variables (2 * N/2)."""
        return 2 * self.n_half

    @property
    def y0(self):
        """Initial guess."""
        return jnp.ones(self.n)

    @property
    def args(self):
        """No additional arguments."""
        return None

    def objective(self, y, args):
        """Compute the objective function.

        The objective is the sum of squared groups:
        - BA(i) = 1.5 - X(2i-1) * (1 - X(2i)) for i = 1 to N/2
        - BB(i) = 2.25 - X(2i-1) * (1 - X(2i)^2) for i = 1 to N/2
        - BC(i) = 2.625 - X(2i-1) * (1 - X(2i)^3) for i = 1 to N/2
        - L(i) = (6*X(2i) - X(2i+1)) / ALPHA for i = 1 to N/2-1
        """
        del args  # Not used

        alphinv = 1.0 / self.ALPHA

        # Extract odd and even indexed elements
        # X(2i-1) corresponds to y[2i] (0-based)
        # X(2i) corresponds to y[2i+1] (0-based)
        x_odd = y[::2]  # X(1), X(3), X(5), ...
        x_even = y[1::2]  # X(2), X(4), X(6), ...

        # Compute residuals vectorized
        # BA(i) = 1.5 - X(2i-1) * (1 - X(2i))
        ba_res = 1.5 - x_odd * (1.0 - x_even)

        # BB(i) = 2.25 - X(2i-1) * (1 - X(2i)^2)
        bb_res = 2.25 - x_odd * (1.0 - x_even**2)

        # BC(i) = 2.625 - X(2i-1) * (1 - X(2i)^3)
        bc_res = 2.625 - x_odd * (1.0 - x_even**3)

        # Sum the squared residuals for BA, BB, BC
        obj = jnp.sum(ba_res**2) + jnp.sum(bb_res**2) + jnp.sum(bc_res**2)

        # L(i) = (6*X(2i) - X(2i+1)) / ALPHA for i = 1 to N/2-1
        # From SIF: L(I) has X(J+1) coeff 6.0 and X(J+2) coeff -1.0
        # where J = 2I-1, so L(I) uses X(2I) and X(2I+1)
        # In 0-based: L(i) uses y[2i+1] and y[2i+2]
        l_res = (6.0 * y[1:-1:2] - y[2::2]) * alphinv
        obj += jnp.sum(l_res**2)

        return obj

    @property
    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value is 0.0."""
        return jnp.array(0.0)
