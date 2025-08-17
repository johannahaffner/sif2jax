import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class READING9(AbstractConstrainedMinimisation):
    """A nonlinear optimal control problem from Nancy Nichols with initial condition.
    This problem arises in tide modelling.

    Source: a variant upon a problem in
    S. Lyle and N.K. Nichols,
    "Numerical Methods for Optimal Control Problems with State Constraints",
    Numerical Analysis Report 8/91, Dept of Mathematics,
    University of Reading, UK.

    SIF input: Nick Gould and Ph. Toint, March 1995
    Classification: OOR2-MN-V-V
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})
    n_plus_1: int = 5001  # Default from SIF

    @property
    def n(self):
        return self.n_plus_1 - 1

    # Constants
    k1: float = 0.07716
    k2: float = 0.07716

    def objective(self, y, args):
        # Variables are P(0:N) followed by U(0:N)
        p = y[: self.n + 1]
        u = y[self.n + 1 :]

        h = 1.0 / self.n
        h_half = h * 0.5

        # Objective: sum of OE(i) elements
        # OE(i) uses PROD2 element: (U(i) * P(i))^2
        # Each appears with coefficient -H/2

        # PROD2 elements: (u * p)^2
        prod2_elements = (u * p) ** 2

        # Sum with trapezoidal rule coefficients
        # OE(0) to OE(N-1) each appear with -H/2
        # OE(1) to OE(N) each appear with -H/2
        # So OE(0) and OE(N) appear once with -H/2
        # OE(1) to OE(N-1) appear twice with -H/2 = -H

        obj = -h_half * (prod2_elements[0] + prod2_elements[self.n])
        obj += -h * jnp.sum(prod2_elements[1 : self.n])

        return obj

    @property
    def y0(self):
        # Initial guess: 0.2 for all variables (from START POINT)
        return jnp.full(2 * (self.n + 1), 0.2)

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        return self.y0

    @property
    def expected_objective_value(self):
        return jnp.array(-4.41677e-02)  # For n=500

    @property
    def bounds(self):
        # P(i) are free except P(0) which is fixed at 0.0
        # U(i) are bounded in [0, 1]

        lower = jnp.full(2 * (self.n + 1), -jnp.inf)
        upper = jnp.full(2 * (self.n + 1), jnp.inf)

        # P(0) fixed at 0.0
        lower = lower.at[0].set(0.0)
        upper = upper.at[0].set(0.0)

        # U(i) bounds: [0, 1] for i=0 to N
        u_start = self.n + 1
        lower = lower.at[u_start:].set(0.0)
        upper = upper.at[u_start:].set(1.0)

        return lower, upper

    def constraint(self, y):
        # Variables
        p = y[: self.n + 1]
        u = y[self.n + 1 :]

        h = 1.0 / self.n
        k1h = self.k1 * h
        k1h_plus_1 = k1h + 1.0
        minus_k1h_minus_1 = -k1h_plus_1
        k2h = self.k2 * h

        # Build equality constraints S(0) to S(N-1)
        equalities = jnp.zeros(self.n)

        # Vectorized computation
        # S(i): P(i+1) * 1.0 + P(i) * (-K1H-1) + CE(i) * K2H = HSINT(i)
        # CE(i) is PROD element: U(i) * P(i)

        # RHS: H * sin(i * H)
        i_values = jnp.arange(self.n, dtype=y.dtype)
        t_values = i_values * h
        hsint = h * jnp.sin(t_values)

        # Linear terms
        equalities = p[1 : self.n + 1] + minus_k1h_minus_1 * p[: self.n]

        # CE(i) terms: K2H * U(i) * P(i)
        ce_terms = k2h * u[: self.n] * p[: self.n]
        equalities += ce_terms

        # Apply RHS
        equalities -= hsint

        return equalities, None
