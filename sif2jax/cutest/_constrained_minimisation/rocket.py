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
        """Unpack flat array into problem variables."""
        step = y[0]
        idx = 1

        # Variables ordered by time point: H(0), V(0), M(0), T(0), D(0), G(0), H(1), ...
        n_points = self.NH + 1
        H = jnp.zeros(n_points)
        V = jnp.zeros(n_points)
        M = jnp.zeros(n_points)
        T = jnp.zeros(n_points)
        D = jnp.zeros(n_points)
        G = jnp.zeros(n_points)

        # Extract variables for each time point
        for i in range(n_points):
            H = H.at[i].set(y[idx])
            idx += 1
            V = V.at[i].set(y[idx])
            idx += 1
            M = M.at[i].set(y[idx])
            idx += 1
            T = T.at[i].set(y[idx])
            idx += 1
            D = D.at[i].set(y[idx])
            idx += 1
            G = G.at[i].set(y[idx])
            idx += 1

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

        equality_constraints = []

        # First: Interleaved drag and gravity constraints for i=0..NH
        for i in range(self.NH + 1):
            # Drag constraint: DC * V[i]^2 * exp(-HC * (H[i] - H0) / H0) - D[i] = 0
            drag_val = (
                self.DC
                * V[i] ** 2
                * jnp.exp(-self.HC_over_H0 * (H[i] - self.H0) / self.H0)
                - D[i]
            )
            equality_constraints.append(drag_val)

            # Gravity constraint: G0 * (H0 / H[i])^2 - G[i] = 0
            grav_val = self.G0H02 / (H[i] ** 2) - G[i]
            equality_constraints.append(grav_val)

        # Then: Motion equations for j=1..NH
        for j in range(1, self.NH + 1):
            # Height equation: -H[j] + H[j-1] + 0.5 * step * (V[j] + V[j-1]) = 0
            # From SIF: XE H(J) H(J) -1.0 H(J-1) 1.0 and XE H(J) H(J) 0.5
            h_eq = -H[j] + H[j - 1] + 0.5 * step * (V[j] + V[j - 1])
            equality_constraints.append(h_eq)

            # Velocity eq: -V[j] + V[j-1] + 0.5 * step * (accel_j + accel_j_minus_1) = 0
            # From SIF: XE V(J) V(J) -1.0 V(J-1) 1.0 and XE V(J) V(J) 0.5 V(J-1) 0.5
            accel_j = (T[j] - D[j] - M[j] * G[j]) / M[j]
            accel_j_minus_1 = (T[j - 1] - D[j - 1] - M[j - 1] * G[j - 1]) / M[j - 1]
            v_eq = -V[j] + V[j - 1] + 0.5 * step * (accel_j + accel_j_minus_1)
            equality_constraints.append(v_eq)

            # Mass equation: -M[j] + M[j-1] - 0.5 * step * (T[j] + T[j-1]) / C = 0
            # From SIF: XE M(J) M(J) -1.0 M(J-1) 1.0 and ZE M(J) M(J) -1/2C
            m_eq = -M[j] + M[j - 1] - 0.5 * step * (T[j] + T[j - 1]) / self.C
            equality_constraints.append(m_eq)

        return jnp.array(equality_constraints), None

    @property
    def y0(self):
        """Initial guess."""
        y0_list = []

        # Step size
        y0_list.append(1.0 / self.NH)

        # Variables ordered by time point: H(i), V(i), M(i), T(i), D(i), G(i) for each i
        for i in range(self.NH + 1):
            ri = float(i)
            i_over_nh = ri / self.NH if self.NH > 0 else 0.0

            # H(i) = 1.0 (HI = ONE from SIF)
            hi = 1.0
            y0_list.append(hi)

            # V(i) = i/nh * (1 - i/nh)
            vi = i_over_nh * (1.0 - i_over_nh)
            y0_list.append(vi)

            # M(i) = M0 + (MF - M0) * i/nh
            mi = self.M0 + (self.MF - self.M0) * i_over_nh
            y0_list.append(mi)

            # T(i) = TMAX/2
            ti = self.TMAX / 2.0
            y0_list.append(ti)

            # D(i) = DC * vi^2 * exp(-HC * (hi - H0) / H0)
            # From SIF: DI = DC * VI^2 * exp(-HC * (HI - H0) / H0)
            # Since hi = H0 = 1.0, exp(-HC * 0 / H0) = exp(0) = 1
            di = self.DC * vi**2
            y0_list.append(di)

            # G(i) = G0 * (H0 / hi)^2
            # From SIF: GI = G0 * (H0 / HI)^2 = 1.0 * (1.0 / 1.0)^2 = 1.0
            gi = self.G0
            y0_list.append(gi)

        return jnp.array(y0_list)

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
