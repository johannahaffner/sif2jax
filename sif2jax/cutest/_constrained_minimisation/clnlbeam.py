"""Clamped nonlinear beam optimal control problem.

An optimal control version of the CLamped NonLinear BEAM problem.
The energy of a beam of length 1 compressed by a force P is to be
minimized.  The control variable is the derivative of the deflection angle.

The problem is discretized using the trapezoidal rule. It is non-convex.

Source:
H. Maurer and H.D. Mittelman,
"The non-linear beam via optimal control with bound state variables",
Optimal Control Applications and Methods 12, pp. 19-31, 1991.

SIF input: Ph. Toint, Nov 1993.

Classification: OOR2-MN-V-V
"""

import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class CLNLBEAM(AbstractConstrainedMinimisation):
    """Clamped nonlinear beam optimal control problem."""

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Parameters
    NI: int = 1999  # Number of interior points + 1
    ALPHA: float = 350.0  # Force divided by bending stiffness

    @property
    def n(self):
        """Number of variables."""
        # Variables: T(0) to T(NI), X(0) to X(NI), U(0) to U(NI)
        return 3 * (self.NI + 1)

    @property
    def m(self):
        """Number of constraints."""
        # State equations: EX(0) to EX(NI-1), ET(0) to ET(NI-1)
        return 2 * self.NI

    @property
    def m_linear(self):
        """Number of linear constraints."""
        return 0

    @property
    def m_nonlinear(self):
        """Number of nonlinear constraints."""
        return self.m

    def _get_indices(self):
        """Get variable indices."""
        ni = self.NI
        # T indices: 0 to NI
        t_idx = list(range(ni + 1))
        # X indices: NI+1 to 2*NI+1
        x_idx = list(range(ni + 1, 2 * ni + 2))
        # U indices: 2*NI+2 to 3*NI+2
        u_idx = list(range(2 * ni + 2, 3 * ni + 3))
        return t_idx, x_idx, u_idx

    @property
    def y0(self):
        """Initial guess (perturbed from origin)."""
        y = jnp.zeros(self.n)

        t_idx, x_idx, u_idx = self._get_indices()
        h = 1.0 / self.NI

        # Perturb the origin
        for i in range(self.NI + 1):
            tt = i * h
            ctt = jnp.cos(tt)
            sctt = 0.05 * ctt
            y = y.at[t_idx[i]].set(sctt)
            y = y.at[x_idx[i]].set(sctt)

        return y

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def bounds(self):
        """Bounds on the variables."""
        lw = -jnp.inf * jnp.ones(self.n)
        up = jnp.inf * jnp.ones(self.n)

        t_idx, x_idx, u_idx = self._get_indices()

        # Bounds on displacements X(i): [-0.05, 0.05]
        for i in range(self.NI + 1):
            lw = lw.at[x_idx[i]].set(-0.05)
            up = up.at[x_idx[i]].set(0.05)

        # Bounds on deflection angles T(i): [-1.0, 1.0]
        for i in range(self.NI + 1):
            lw = lw.at[t_idx[i]].set(-1.0)
            up = up.at[t_idx[i]].set(1.0)

        # Boundary conditions (fixed values)
        lw = lw.at[x_idx[0]].set(0.0)
        up = up.at[x_idx[0]].set(0.0)
        lw = lw.at[x_idx[self.NI]].set(0.0)
        up = up.at[x_idx[self.NI]].set(0.0)

        lw = lw.at[t_idx[0]].set(0.0)
        up = up.at[t_idx[0]].set(0.0)
        lw = lw.at[t_idx[self.NI]].set(0.0)
        up = up.at[t_idx[self.NI]].set(0.0)

        return lw, up

    def objective(self, y, args):
        """Compute the energy objective function."""
        del args  # Not used

        t_idx, x_idx, u_idx = self._get_indices()
        h = 1.0 / self.NI
        ah = self.ALPHA * h

        energy = 0.0

        # Sum over intervals
        for i in range(self.NI):
            # Extract variables
            t_i = y[t_idx[i]]
            t_ip1 = y[t_idx[i + 1]]
            u_i = y[u_idx[i]]
            u_ip1 = y[u_idx[i + 1]]

            # Energy terms
            energy += (h / 2.0) * (u_ip1 * u_ip1 + u_i * u_i)
            energy += (ah / 2.0) * (jnp.cos(t_ip1) + jnp.cos(t_i))

        return jnp.array(energy)

    def constraint(self, y):
        """Compute the constraints (state equations)."""

        t_idx, x_idx, u_idx = self._get_indices()
        h = 1.0 / self.NI

        constraints = []

        # State equations
        for i in range(self.NI):
            # EX(i): X(i+1) - X(i) - h/2 * (sin(T(i+1)) + sin(T(i))) = 0
            x_i = y[x_idx[i]]
            x_ip1 = y[x_idx[i + 1]]
            t_i = y[t_idx[i]]
            t_ip1 = y[t_idx[i + 1]]

            ex_i = x_ip1 - x_i - (h / 2.0) * (jnp.sin(t_ip1) + jnp.sin(t_i))
            constraints.append(ex_i)

            # ET(i): T(i+1) - T(i) - h/2 * (U(i+1) + U(i)) = 0
            u_i = y[u_idx[i]]
            u_ip1 = y[u_idx[i + 1]]

            et_i = t_ip1 - t_i - (h / 2.0) * (u_ip1 + u_i)
            constraints.append(et_i)

        # Return as tuple (equality_constraints, inequality_constraints)
        # All constraints are equality constraints
        return jnp.array(constraints), None

    @property
    def expected_result(self):
        """Expected optimal value for NI=50."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value for NI=50."""
        return jnp.array(344.8673691861)
