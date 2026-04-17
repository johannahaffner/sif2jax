import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class ROCKET(AbstractConstrainedMinimisation):
    """ROCKET problem - Maximize final altitude of vertically-launched rocket.

    Maximize the final altitude of a vertically-launched rocket, using
    the thrust as a control and given the initial mass, the fuel mass
    and the drag characteristics of the rocket.

    This is problem 10 in the COPS (Version 2) collection of
    E. Dolan and J. More'
    see "Benchmarking Optimization Software with COPS"
    Argonne National Labs Technical Report ANL/MCS-246 (2000)

    SIF input: Nick Gould, November 2000

    Classification: OOR2-AN-V-V

    Problem structure:
    - NH+1 time points in discretization (default NH=400)
    - Variables: step size, and at each time point i:
      - H(i): height
      - V(i): velocity
      - M(i): mass
      - T(i): thrust (control variable)
      - D(i): drag
      - G(i): gravity
    - Constraints: physics equations
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Number of subintervals
    NH = 400

    # Physical parameters (normalized)
    V0 = 0.0  # Initial velocity
    G0 = 1.0  # Surface gravity
    H0 = 1.0  # Initial height
    M0 = 1.0  # Initial mass

    # Model parameters
    TC = 3.5
    HC = 500.0
    VC = 620.0
    MC = 0.6

    # Derived parameters (computed as class attributes)
    G0H0 = G0 * H0
    G0H02 = G0H0 * H0
    C = jnp.sqrt(G0H0) * 0.5
    MF = MC * M0
    DC = (M0 / G0) * VC * 0.5
    TMAX = M0 * G0 * TC

    # Other useful values
    HC_over_H0 = HC / H0
    inv_2C = -0.5 / C

    @property
    def n(self):
        """Number of variables."""
        # Variables: step + (H, V, M, T, D, G) for each of NH+1 points
        return 1 + 6 * (self.NH + 1)

    @property
    def m(self):
        """Number of constraints."""
        # Constraints: 2*(NH+1) for drag and gravity + 3*NH for motion equations
        return 2 * (self.NH + 1) + 3 * self.NH

    def _unpack_variables(self, y):
        """Unpack flat array into problem variables using reshape (no loops)."""
        step = y[0]
        # y[1:] has layout [H0,V0,M0,T0,D0,G0, H1,V1,...] — reshape to (n_points, 6)
        vars_2d = y[1:].reshape(self.NH + 1, 6)
        H = vars_2d[:, 0]
        V = vars_2d[:, 1]
        M = vars_2d[:, 2]
        T = vars_2d[:, 3]
        D = vars_2d[:, 4]
        G = vars_2d[:, 5]
        return step, H, V, M, T, D, G

    def objective(self, y, args):
        """Compute the objective function (negative final height)."""
        del args
        step, H, V, M, T, D, G = self._unpack_variables(y)
        # Maximize H(NH) = minimize -H(NH)
        return -H[self.NH]

    def constraint(self, y):
        """Compute the constraints in pycutest order."""
        step, H, V, M, T, D, G = self._unpack_variables(y)

        # Drag constraints: DC * V^2 * exp(-HC*(H-H0)/H0) - D = 0
        drag = self.DC * V**2 * jnp.exp(-self.HC_over_H0 * (H - self.H0) / self.H0) - D

        # Gravity constraints: G0*(H0/H)^2 - G = 0
        grav = self.G0H02 / H**2 - G

        # Interleave drag and gravity: [drag0, grav0, drag1, grav1, ...]
        dg = jnp.stack([drag, grav], axis=1).flatten()

        # Motion equations for j=1..NH (using slices)
        # Height:
        h_eq = -H[1:] + H[:-1] + 0.5 * step * (V[1:] + V[:-1])

        # Acceleration:
        accel = (T - D - M * G) / M
        # Velocity:
        v_eq = -V[1:] + V[:-1] + 0.5 * step * (accel[1:] + accel[:-1])

        # Mass:
        m_eq = -M[1:] + M[:-1] - 0.5 * step * (T[1:] + T[:-1]) / self.C

        # Motion equations interleaved: [h_eq1, v_eq1, m_eq1, h_eq2, ...]
        motion = jnp.stack([h_eq, v_eq, m_eq], axis=1).flatten()

        return jnp.concatenate([dg, motion]), None

    @property
    def y0(self):
        """Initial guess."""
        n_points = self.NH + 1
        t_frac = jnp.arange(n_points) / self.NH  # i/NH

        H_init = jnp.ones(n_points)
        V_init = t_frac * (1.0 - t_frac)
        M_init = self.M0 + (self.MF - self.M0) * t_frac
        T_init = jnp.full(n_points, self.TMAX / 2.0)
        # D = DC * V^2 * exp(0) = DC * V^2 since H=H0=1
        D_init = self.DC * V_init**2
        G_init = jnp.full(n_points, self.G0)

        # Interleave: [H0,V0,M0,T0,D0,G0, H1,V1,...]
        vars_2d = jnp.stack([H_init, V_init, M_init, T_init, D_init, G_init], axis=1)
        return jnp.concatenate([jnp.array([1.0 / self.NH]), vars_2d.flatten()])

    @property
    def args(self):
        """Additional arguments."""
        return None

    @property
    def bounds(self):
        """Bounds on variables."""
        n = self.n
        lower = jnp.full(n, -jnp.inf)
        upper = jnp.full(n, jnp.inf)

        # Step >= 0
        lower = lower.at[0].set(0.0)

        # Variables ordered by time point: H(i), V(i), M(i), T(i), D(i), G(i)
        for i in range(self.NH + 1):
            idx = 1 + 6 * i  # Base index for time point i

            # H(i) >= H0
            lower = lower.at[idx].set(self.H0)

            # V(i) >= 0
            lower = lower.at[idx + 1].set(0.0)

            # MF <= M(i) <= M0
            lower = lower.at[idx + 2].set(self.MF)
            upper = upper.at[idx + 2].set(self.M0)

            # 0 <= T(i) <= TMAX
            lower = lower.at[idx + 3].set(0.0)
            upper = upper.at[idx + 3].set(self.TMAX)

            # D(i) and G(i) are unbounded (idx + 4 and idx + 5)

        # Fixed values
        # H(0) = H0, V(0) = V0, M(0) = M0
        lower = lower.at[1].set(self.H0)  # H(0)
        upper = upper.at[1].set(self.H0)
        lower = lower.at[2].set(self.V0)  # V(0)
        upper = upper.at[2].set(self.V0)
        lower = lower.at[3].set(self.M0)  # M(0)
        upper = upper.at[3].set(self.M0)

        # M(NH) = MF
        nh_idx = 1 + 6 * self.NH + 2  # M(NH) position
        lower = lower.at[nh_idx].set(self.MF)
        upper = upper.at[nh_idx].set(self.MF)

        return lower, upper

    @property
    def expected_result(self):
        """Expected optimal solution."""
        # Not provided in SIF file
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        # From SIF comment: -1.0128 for all NH values
        return jnp.array(-1.0128)
