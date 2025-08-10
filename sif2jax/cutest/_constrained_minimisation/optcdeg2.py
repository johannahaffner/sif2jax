"""
Problem OPTCDEG2: Corrected optimal control problem for spring-mass-damper system.

This is a corrected version of OPTCNTRL that properly scales finite difference
derivatives by the discretization parameter DT = 20/T, making it physically
meaningful for any value of T. Additionally, x(T) is a free variable.

Variables:
- x(t): displacement from equilibrium at time t (t=0 to T)
- y(t): velocity at time t (t=0 to T)
- u(t): control force at time t (t=0 to T-1)

Source:
B. Murtagh and M. Saunders,
Mathematical Programming studies 16, pp 84-117,
(example 5.11)

SIF input: Nick Gould, June 1990.
           Todd Plantenga, August 1995.

classification QQR2-AN-V-V
"""

import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class OPTCDEG2(AbstractConstrainedMinimisation):
    """OPTCDEG2: Corrected optimal control of spring-mass-damper system.

    Problem size depends on discretization parameter T:
    - Variables: 3*T + 2 total
    - Constraints: 2*T equality constraints

    Variable ordering (interleaved as in pycutest):
    x(0), y(0), x(1), y(1), ..., x(T), y(T), u(0), ..., u(T-1)
    """

    # Problem parameters
    T: int = 1500  # Default from SIF file
    SPRINGKM: float = 0.02
    DAMPING: float = 0.05

    @property
    def DT(self):
        """Time step size."""
        return 20.0 / self.T

    @property
    def name(self) -> str:
        return "OPTCDEG2"

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
        """Minimize sum of (DT/2) * x(t)^2 for t=0 to T."""
        x, _, _ = self._extract_variables(y)
        return (self.DT / 2.0) * jnp.sum(x**2)

    def constraint(self, y):
        """Equality constraints from discretized dynamics.

        B(t): x(t+1) - x(t) - DT*y(t) = 0
        C(t): y(t+1) - y(t) + C1*x(t) - DT*u(t) + C2*y(t)^2 = 0
        where C1 = SPRINGKM * DT, C2 = DAMPING * DT
        """
        x, y_vel, u = self._extract_variables(y)

        # Compute coefficients
        DT = self.DT
        C1 = self.SPRINGKM * DT
        C2 = self.DAMPING * DT

        # B constraints: x(t+1) - x(t) - DT*y(t) = 0 for t=0 to T-1
        B_constraints = x[1:] - x[:-1] - DT * y_vel[:-1]

        # C constraints: y(t+1) - y(t) + C1*x(t) - DT*u(t) + C2*y(t)^2 = 0
        C_constraints = (
            y_vel[1:] - y_vel[:-1] + C1 * x[:-1] - DT * u + C2 * y_vel[:-1] ** 2
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
        - x(t) free for t=0 to T-1 (DO loop), but x(0) is fixed at 10.0 (XX override)
        - x(T) is explicitly free (XR statement)
        - y(t) >= -1.0 for t=0 to T-1 (DO loop), but y(0)=0.0 and y(T)=0.0
          (XX overrides)
        - u(t) in [-0.2, 0.2] for t=0 to T-1
        """
        lower = jnp.full(self.n, -jnp.inf)
        upper = jnp.full(self.n, jnp.inf)

        # x(0) = 10.0 (fixed via XX) at position 0
        lower = lower.at[0].set(10.0)
        upper = upper.at[0].set(10.0)

        # x(t) for t=1 to T are free (XR in DO loop and explicit XR for x(T))

        # y(t) >= -1.0 for t=0 to T-1 from DO loop
        # Create indices for y positions
        y_positions = 2 * jnp.arange(self.T) + 1
        lower = lower.at[y_positions].set(-1.0)

        # y(0) = 0.0 (fixed via XX) at position 1 - overrides the -1.0
        lower = lower.at[1].set(0.0)
        upper = upper.at[1].set(0.0)

        # y(T) = 0.0 (fixed via XX) at position 2*T + 1
        lower = lower.at[2 * self.T + 1].set(0.0)
        upper = upper.at[2 * self.T + 1].set(0.0)

        # u(t) in [-0.2, 0.2] for t=0 to T-1
        u_start = 2 * (self.T + 1)
        u_indices = jnp.arange(u_start, u_start + self.T)
        lower = lower.at[u_indices].set(-0.2)
        upper = upper.at[u_indices].set(0.2)

        return lower, upper

    @property
    def expected_result(self):
        """Solution not provided in SIF file."""
        return None

    @property
    def expected_objective_value(self):
        """From SIF file comments for different T values."""
        # T=10: 340.6053, T=40: 253.2756, T=100: 237.2764, T=400: 229.573
        # Default T=1500 not provided
        return None
