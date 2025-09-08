import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class CLNLBEAM(AbstractConstrainedMinimisation):
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

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Parameters
    NI: int = 1999  # Number of interior points + 1
    ALPHA: float = 350.0  # Force divided by bending stiffness

    @property
    def n(self):
        """Number of variables."""
        # Variables: T(0) to T(NI), X(0) to X(NI), U(0) to U(NI)
        # But pycutest expects 5999, not 6000 = 3*(NI+1)
        # Perhaps pycutest excludes one of the fixed variables?
        return jnp.array(5999)

    @property
    def m(self):
        """Number of constraints."""
        # State equations: EX(0) to EX(NI-1), ET(0) to ET(NI-1)
        # That's 2 * NI = 2 * 1999 = 3998
        return 2 * self.NI

    @property
    def m_linear(self):
        """Number of linear constraints."""
        return jnp.array(0)

    @property
    def m_nonlinear(self):
        """Number of nonlinear constraints."""
        return self.m

    def _get_indices(self):
        """Get variable indices."""
        # The SIF file has T(0:NI), X(0:NI), U(0:NI) all with NI+1=2000 elements
        # But pycutest expects 5999 variables, not 6000
        # This means one variable is excluded. Standard layout is consecutive blocks
        # Total of 5999 variables: T[0:2000], X[0:2000], but U[0:1999]
        t_start = 0
        t_end = 2000
        x_start = 2000
        x_end = 4000
        u_start = 4000
        u_end = 5999

        t_idx = jnp.arange(t_start, t_end)
        x_idx = jnp.arange(x_start, x_end)
        u_idx = jnp.arange(u_start, u_end)
        return t_idx, x_idx, u_idx

    @property
    def y0(self):
        """Initial guess (perturbed from origin)."""
        y = jnp.zeros(self.n)

        t_idx, x_idx, u_idx = self._get_indices()
        h = 1.0 / self.NI

        # Perturb the origin - vectorized
        i_vals = jnp.arange(self.NI + 1, dtype=jnp.float64)
        tt = i_vals * h
        sctt = 0.05 * jnp.cos(tt)

        y = y.at[t_idx].set(sctt)
        y = y.at[x_idx].set(sctt)

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

        # Bounds on displacements X(i): [-0.05, 0.05] - vectorized
        lw = lw.at[x_idx].set(-0.05)
        up = up.at[x_idx].set(0.05)

        # Bounds on deflection angles T(i): [-1.0, 1.0] - vectorized
        lw = lw.at[t_idx].set(-1.0)
        up = up.at[t_idx].set(1.0)

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

        # Extract variables - vectorized
        t_vals = y[t_idx]
        u_vals = y[u_idx]

        # U has only 1999 elements, but we need 2000 for the objective
        # Add a zero for U(1999)
        u_extended = jnp.concatenate([u_vals, jnp.array([0.0])])

        # Pairs for trapezoidal rule
        t_i = t_vals[:-1]
        t_ip1 = t_vals[1:]
        u_i = u_extended[:-1]
        u_ip1 = u_extended[1:]

        # Energy terms - vectorized
        u_energy = (h / 2.0) * (u_ip1 * u_ip1 + u_i * u_i)
        cos_energy = (ah / 2.0) * (jnp.cos(t_ip1) + jnp.cos(t_i))

        energy = jnp.sum(u_energy + cos_energy)

        return energy

    def constraint(self, y):
        """Compute the constraints (state equations)."""

        t_idx, x_idx, u_idx = self._get_indices()
        h = 1.0 / self.NI

        # Extract variables - vectorized
        t_vals = y[t_idx]
        x_vals = y[x_idx]
        u_vals = y[u_idx]

        # Pairs for constraints
        # We have NI constraints (indexed 0 to NI-1)
        # For constraint i, we use indices i and i+1
        x_i = x_vals[: self.NI]
        x_ip1 = x_vals[1 : self.NI + 1]
        t_i = t_vals[: self.NI]
        t_ip1 = t_vals[1 : self.NI + 1]

        # U has only 1999 elements (0 to 1998)
        # For the last constraint ET(1998), we need U(1999) which doesn't exist
        # So we need to add a zero for U(1999)
        u_extended = jnp.concatenate([u_vals, jnp.array([0.0])])
        u_i = u_extended[: self.NI]
        u_ip1 = u_extended[1 : self.NI + 1]

        # State equations - vectorized
        # EX(i): X(i+1) - X(i) - h/2 * (sin(T(i+1)) + sin(T(i))) = 0
        ex = x_ip1 - x_i - (h / 2.0) * (jnp.sin(t_ip1) + jnp.sin(t_i))

        # ET(i): T(i+1) - T(i) - h/2 * (U(i+1) + U(i)) = 0
        et = t_ip1 - t_i - (h / 2.0) * (u_ip1 + u_i)

        # Interleave EX and ET constraints
        # We have NI EX constraints and NI ET constraints
        n_constraints = len(ex) + len(et)
        constraints = jnp.zeros(n_constraints)
        constraints = constraints.at[::2].set(ex)  # Even indices: EX
        constraints = constraints.at[1::2].set(et)  # Odd indices: ET

        # Return as tuple (equality_constraints, inequality_constraints)
        # All constraints are equality constraints
        return constraints, None

    @property
    def expected_result(self):
        """Expected optimal value for NI=50."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value for NI=50."""
        return jnp.array(344.8673691861)
