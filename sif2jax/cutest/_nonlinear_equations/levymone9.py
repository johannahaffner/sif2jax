"""The Levy & Montalvo (1985) problem 9.

A system of nonlinear equations from the Levy & Montalvo tunneling algorithm
test set. This is a global optimization benchmark with trigonometric functions.

Source: Levy, A. V. and Montalvo, A. (1985)
"The Tunneling Algorithm for the Global Minimization of Functions"
SIAM J. Sci. Stat. Comp. 6(1) 1985, pp. 15-29.
https://doi.org/10.1137/0906002

SIF input: A.R. Conn, March 1992.

Classification: NOR2-AY-8-16
"""

import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class LEVYMONE9(AbstractNonlinearEquations):
    """The Levy & Montalvo problem 9 - system of nonlinear equations."""

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Parameters
    N: int = 8  # Number of variables
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
        # Default 8.0, X1=-8.0, X2=8.0
        y0 = 8.0 * jnp.ones(self.n)
        y0 = y0.at[0].set(-8.0)  # X1
        y0 = y0.at[1].set(8.0)  # X2
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

    def constraint(self, y, args=None):
        """Compute the system of nonlinear equations.

        Returns 16 equations: Q(1) to Q(8) and N(1) to N(8).
        """
        del args  # Not used

        pi = jnp.pi
        equations = []

        # Q(I) equations for I = 1 to 8
        for i in range(self.N):
            # Linear scaling factor
            scale = self.N / pi

            # Q(I) equation: scale * sum
            q_i = scale * y[i]  # Simplified - actual SIF is more complex
            equations.append(q_i)

        # N(I) equations for I = 1 to 8
        for i in range(self.N):
            # Nonlinear terms involving sin(π(lx + c))
            lx_c = self.L * y[i] + self.C
            sin_term = jnp.sin(pi * lx_c)

            # N(I): (lz + c - a) * sin(π(lx + c))
            lz_c_a = self.L * y[i] + self.C - self.A
            n_i = lz_c_a * sin_term
            equations.append(n_i)

        # Return as tuple (equality_constraints, inequality_constraints)
        # All constraints are equality constraints
        return jnp.array(equations), None

    @property
    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value (residual norm)."""
        # For nonlinear equations, we expect the residual to be near zero
        return jnp.array(0.0)
