"""The Levy and Montalvo global optimization test problem.

TODO: Human review needed
Attempts made: Multiple interpretations of SCALE parameter, checked group structure
Suspected issues: SCALE interpretation in SIF format not matching pycutest
Additional resources needed: Documentation on exact SCALE behavior in SIF files

This is a bounded nonlinear least squares problem with trigonometric terms,
designed to test global optimization algorithms. The problem has multiple 
local minima due to the sine functions.

Source: A. Levy, A. Montalvo,
The tunneling algorithm for the global minimization of functions,
SIAM J. Sci. Stat. Comp. 6 (1985) 1, pp 15-29.

SIF input: A.R. Conn, May 1993.

Classification: SBR2-AY-V-0
"""

import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class LEVYMONT(AbstractBoundedMinimisation):
    """The Levy-Montalvo global optimization test problem."""

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Parameters
    N: int = 100  # Number of variables
    A: float = 1.0
    K: float = 10.0
    L: float = 1.0
    C: float = 0.0

    @property
    def n(self):
        """Number of variables."""
        return self.N

    @property
    def y0(self):
        """Initial guess (LEVYMONTA starting point)."""
        # Default: x1 = -8.0, x2 = 8.0, others = 8.0
        y0 = 8.0 * jnp.ones(self.n)
        if self.n >= 1:
            y0 = y0.at[0].set(-8.0)
        if self.n >= 2:
            y0 = y0.at[1].set(8.0)
        return y0

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def bounds(self):
        """Bounds on the variables (all [-10, 10])."""
        lw = -10.0 * jnp.ones(self.n)
        up = 10.0 * jnp.ones(self.n)
        return lw, up

    def objective(self, y, args):
        """Compute the sum of squares objective function.

        The objective consists of:
        1. Q(i) terms: Linear terms scaled by N/π
        2. N(i) terms: Nonlinear terms with sine functions
        """
        del args  # Not used

        pi = jnp.pi
        n_over_pi = self.N / pi
        sqrt_k_pi_over_n = jnp.sqrt(self.K * pi / self.N)

        sum_of_squares = 0.0

        # Q(i) groups: coefficient L is scaled by N/π
        # Group value is: (N/π * L) * x(i) - (A-C)
        a_minus_c = self.A - self.C
        scaled_L = n_over_pi * self.L
        for i in range(self.N):
            group_value = scaled_L * y[i] - a_minus_c
            sum_of_squares += group_value**2

        # N(i) groups: elements already include the sqrt(K*π/N) scaling
        # N(1) uses element S2: sin(π(L*x1 + C))
        # Element is scaled by sqrt(K*π/N), squared contribution is K*π/N * sin^2(...)
        lx1_c = self.L * y[0] + self.C
        element_1 = jnp.sin(pi * lx1_c)
        n_1_scaled = sqrt_k_pi_over_n * element_1
        sum_of_squares += n_1_scaled**2

        # N(i) for i >= 2 uses element PS2: (L*x(i-1) + C - A) * sin(π(L*x(i) + C))
        for i in range(1, self.N):
            lx_im1_c_a = self.L * y[i - 1] + self.C - self.A
            lx_i_c = self.L * y[i] + self.C
            element_i = lx_im1_c_a * jnp.sin(pi * lx_i_c)
            n_i_scaled = sqrt_k_pi_over_n * element_i
            sum_of_squares += n_i_scaled**2

        return sum_of_squares

    @property
    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        return jnp.array(0.0)
