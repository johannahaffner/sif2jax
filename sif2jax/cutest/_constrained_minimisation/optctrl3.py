"""
Problem OPTCTRL3: Modified optimal control with penalty terms.

This is a modification of OPTCNTRL where bound constraints on y and u have been
removed and replaced with penalty terms in the objective to handle what would
have been active bounds.

Source:
B. Murtagh and M. Saunders,
Mathematical Programming studies 16, pp 84-117,
(example 5.11)
Modified by Todd Plantenga, December 1992.

SIF input: Nick Gould, June 1990 and T. Plantenga December 1992.

classification QQR2-AN-V-V
"""

import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class OPTCTRL3(AbstractConstrainedMinimisation):
    """OPTCTRL3: Modified optimal control with penalty terms.

    Problem size depends on discretization parameter T:
    - Variables: 3*T + 2 total
    - Constraints: 2*T equality constraints

    Variable ordering (interleaved as in pycutest):
    x(0), y(0), x(1), y(1), ..., x(T), y(T), u(0), ..., u(T-1)
    """

    # Problem parameters
    T: int = 1500  # Default from SIF file
    TDP_mu: float = 1.0e6  # Penalty parameter

    @property
    def name(self) -> str:
        return "OPTCTRL3"

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self) -> int:
        # x(0), y(0), ..., x(T), y(T), u(0), ..., u(T-1)
        return 2 * (self.T + 1) + self.T

    @property
    def m(self) -> int:
        # B(0) to B(T-1), C(0) to C(T-1)
        return 2 * self.T

    def _extract_variables(self, y):
        """Extract x, y_vel, u from interleaved variable vector."""
        # Variables are interleaved: x(0), y(0), x(1), y(1), ..., x(T), y(T)
        # Then u(0), ..., u(T-1)

        # Extract x and y_vel from interleaved section using slicing
        # Reshape to (T+1, 2) to get pairs, then extract columns
        interleaved = y[: 2 * (self.T + 1)].reshape(self.T + 1, 2)
        x = interleaved[:, 0]  # x(t) values
        y_vel = interleaved[:, 1]  # y(t) values

        # Extract u from the end
        u_start = 2 * (self.T + 1)
        u = y[u_start : u_start + self.T]

        return x, y_vel, u

    @property
    def y0(self):
        # Initial point from SIF file with interleaved ordering
        y0_vals = jnp.zeros(self.n)

        # Set y(t) = -1.0 for t=1 to T-1 in interleaved positions
        # y(t) is at position 2*t + 1
        # Create indices for positions where y(t) should be -1.0
        indices = 2 * jnp.arange(1, self.T) + 1
        y0_vals = y0_vals.at[indices].set(-1.0)

        return y0_vals

    @property
    def args(self):
        return None

    def objective(self, y, args):
        """Minimize sum of 0.5*x(t)^2 + penalty terms for u bound violations.

        From SIF file:
        - 0.5 * x(t)^2 for t=0 to T
        - TDP_mu * (u(t) - 0.2)^2 for t=1 to T-1 (note: starts at t=1, not t=0)
        """
        x, _, u = self._extract_variables(y)

        # Original objective
        obj = 0.5 * jnp.sum(x**2)

        # Penalty terms for u constraints
        # From SIF: penalty is (u - 0.2)^2 for u(1) to u(T-1), not u(0)
        # The SQR_TDP element computes (X - 0.2)^2
        penalty = self.TDP_mu * jnp.sum((u[1:] - 0.2) ** 2)

        return obj + penalty

    def constraint(self, y):
        """Equality constraints from discretized dynamics.

        B(t): x(t+1) - x(t) - 0.2*y(t) = 0
        C(t): y(t+1) - y(t) + 0.004*x(t) - 0.2*u(t) + 0.01*y(t)^2 = 0
        """
        x, y_vel, u = self._extract_variables(y)

        # B constraints: x(t+1) - x(t) - 0.2*y(t) = 0 for t=0 to T-1
        B_constraints = x[1:] - x[:-1] - 0.2 * y_vel[:-1]

        # C constraints: y(t+1) - y(t) + 0.004*x(t) - 0.2*u(t) + 0.01*y(t)^2 = 0
        C_constraints = (
            y_vel[1:] - y_vel[:-1] + 0.004 * x[:-1] - 0.2 * u + 0.01 * y_vel[:-1] ** 2
        )

        # Interleave constraints as B(0), C(0), B(1), C(1), ...
        # Stack B and C constraints and reshape to interleave them
        equality_constraints = jnp.stack(
            [B_constraints, C_constraints], axis=1
        ).reshape(2 * self.T)

        return equality_constraints, None

    @property
    def bounds(self):
        """Variable bounds from SIF file with interleaved ordering.

        From SIF file:
        - All x(t), y(t), u(t) are free (XR in DO loop)
        - x(0) = 10.0 (XX override)
        - x(T) is explicitly free (XR statement)
        - y(0) = 0.0, y(T) = 0.0 (XX overrides)
        """
        lower = jnp.full(self.n, -jnp.inf)
        upper = jnp.full(self.n, jnp.inf)

        # x(0) = 10.0 (fixed via XX) at position 0
        lower = lower.at[0].set(10.0)
        upper = upper.at[0].set(10.0)

        # y(0) = 0.0 (fixed via XX) at position 1
        lower = lower.at[1].set(0.0)
        upper = upper.at[1].set(0.0)

        # y(T) = 0.0 (fixed via XX) at position 2*T + 1
        lower = lower.at[2 * self.T + 1].set(0.0)
        upper = upper.at[2 * self.T + 1].set(0.0)

        # x(T) is explicitly free - no additional bounds needed

        return lower, upper

    @property
    def expected_result(self):
        """Solution not provided in SIF file."""
        return None

    @property
    def expected_objective_value(self):
        """From SIF file comments for different mu and T values."""
        # For T=1500, mu=1e6: not provided
        # For T=10, mu=1e6: ~549.999986
        # For T=40, mu=1e6: 2048.003
        return None
