"""
Problem OPTCNTRL: An optimal control problem.

A spring-mass-damper system optimal control problem from Murtagh and Saunders.
The problem determines the applied force that restores the system to equilibrium.

Variables:
- x(t): displacement from equilibrium at time t (t=0 to T)
- y(t): velocity at time t (t=0 to T)
- u(t): control force at time t (t=0 to T-1)

Source:
B. Murtagh and M. Saunders,
Mathematical Programming studies 16, pp 84-117,
(example 5.11)

SIF input: Nick Gould, June 1990.

classification QQR2-AN-32-20
"""

import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class OPTCNTRL(AbstractConstrainedMinimisation):
    """OPTCNTRL: Optimal control of spring-mass-damper system.

    Problem size depends on discretization parameter T:
    - Variables: 3*T + 2 total (2*(T+1) for x,y and T for u)
    - Constraints: 2*T equality constraints

    Variable ordering (interleaved as in pycutest):
    x(0), y(0), x(1), y(1), ..., x(T), y(T), u(0), ..., u(T-1)
    """

    # Problem parameters
    T: int = 10  # Time discretization steps

    @property
    def name(self) -> str:
        return "OPTCNTRL"

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

        # Extract x and y_vel from interleaved section
        x = jnp.zeros(self.T + 1)
        y_vel = jnp.zeros(self.T + 1)

        for t in range(self.T + 1):
            x = x.at[t].set(y[2 * t])  # x(t) at position 2*t
            y_vel = y_vel.at[t].set(y[2 * t + 1])  # y(t) at position 2*t + 1

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
        for t in range(1, self.T):  # t=1 to 9
            y0_vals = y0_vals.at[2 * t + 1].set(-1.0)

        return y0_vals

    @property
    def args(self):
        return None

    def objective(self, y, args):
        """Minimize sum of 0.5 * x(t)^2 for t=0 to T."""
        x, _, _ = self._extract_variables(y)
        return 0.5 * jnp.sum(x**2)

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
        equality_constraints = jnp.zeros(2 * self.T)
        for t in range(self.T):
            equality_constraints = equality_constraints.at[2 * t].set(B_constraints[t])
            equality_constraints = equality_constraints.at[2 * t + 1].set(
                C_constraints[t]
            )

        return equality_constraints, None

    @property
    def bounds(self):
        """Variable bounds from SIF file with interleaved ordering.

        From SIF file:
        - x(t) free for t=0 to T-1 (DO loop), but x(0) is fixed at 10.0 (XX override)
        - x(T) has no explicit bounds specified, so it's free
        - y(t) >= -1.0 for t=0 to T-1 (DO loop), but y(0)=0.0 and y(T)=0.0
          (XX overrides)
        - u(t) in [-0.2, 0.2] for t=0 to T-1
        """
        lower = jnp.full(self.n, -jnp.inf)
        upper = jnp.full(self.n, jnp.inf)

        # x(0) = 10.0 (fixed via XX) at position 0
        lower = lower.at[0].set(10.0)
        upper = upper.at[0].set(10.0)

        # x(t) for t=1 to T-1 are free (XR in DO loop)
        # x(T) at position 2*T: pycutest seems to default to lower bound 0
        # when no explicit bound is set (outside DO loop)
        lower = lower.at[2 * self.T].set(0.0)

        # y(t) >= -1.0 for t=0 to T-1 from DO loop
        # But y(0) and y(T) have XX overrides
        for t in range(self.T):  # t=0 to T-1
            y_pos = 2 * t + 1
            lower = lower.at[y_pos].set(-1.0)

        # y(0) = 0.0 (fixed via XX) at position 1
        lower = lower.at[1].set(0.0)
        upper = upper.at[1].set(0.0)

        # y(T) = 0.0 (fixed via XX) at position 2*T + 1
        lower = lower.at[2 * self.T + 1].set(0.0)
        upper = upper.at[2 * self.T + 1].set(0.0)

        # u(t) in [-0.2, 0.2] for t=0 to T-1
        u_start = 2 * (self.T + 1)
        for t in range(self.T):
            lower = lower.at[u_start + t].set(-0.2)
            upper = upper.at[u_start + t].set(0.2)

        return lower, upper

    @property
    def expected_result(self):
        """Solution not provided in SIF file."""
        return None

    @property
    def expected_objective_value(self):
        """From SIF file comment: 549.9999869"""
        return jnp.array(549.9999869)
