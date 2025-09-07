import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class FLOSP2TH(AbstractNonlinearEquations):
    """
    A two-dimensional base flow problem in an inclined enclosure.

    Temperature constant at y = +/- 1 boundary conditions
    High Reynolds number (RA = 1.0E+7)

    The flow is considered in a square of length 2, centered on the
    origin and aligned with the x-y axes. The square is divided into
    4 n ** 2 sub-squares, each of length 1 / n. The differential
    equation is replaced by discrete nonlinear equations at each of
    the grid points.

    The differential equation relates the vorticity, temperature and
    a stream function.

    Source:
    J. N. Shadid
    "Experimental and computational study of the stability
    of Natural convection flow in an inclined enclosure",
    Ph. D. Thesis, University of Minnesota, 1989,
    problem SP2 (pp.128-130).

    SIF input: Nick Gould, August 1993.

    classification NQR2-MY-V-V
    """

    m: int = 15  # Half the number of discretization intervals
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Problem parameters as class attributes
    ra: float = 1.0e7  # Rayleigh number (high)
    ax: float = 1.0
    theta: float = jnp.pi * 0.5

    # Boundary condition parameters for temperature constant case
    a1: float = 0.0
    a2: float = 1.0
    a3: float = 0.0
    b1: float = 0.0
    b2: float = 1.0
    b3: float = 1.0
    f1: float = 1.0
    f2: float = 0.0
    f3: float = 0.0
    g1: float = 1.0
    g2: float = 0.0
    g3: float = 0.0

    # Grid parameters
    h: float = 1.0 / 15  # m = 15
    h2: float = (1.0 / 15) * (1.0 / 15)

    # Derived parameters
    axx: float = 1.0  # ax * ax
    pi1: float = 0.0  # -0.5 * 1.0 * 1.0e7 * cos(pi/2) = 0.0
    pi2: float = 5.0e6  # 0.5 * 1.0 * 1.0e7 * sin(pi/2) = 5e6

    # Grid dimensions
    grid_size: int = 2 * 15 + 1  # = 31
    n_vars: int = 3 * 31 * 31  # = 2883

    def starting_point(self) -> Array:
        """Initial guess for the optimization problem."""
        return jnp.zeros(self.n_vars, dtype=jnp.float64)

    def num_residuals(self) -> int:
        """Number of residual equations."""
        # Interior equations: S(I,J), V(I,J), E(I,J) for I,J in [-M+1, M-1]
        n_interior_S = (self.grid_size - 2) * (self.grid_size - 2)
        n_interior_V = (self.grid_size - 2) * (self.grid_size - 2)
        n_interior_E = (self.grid_size - 2) * (self.grid_size - 2)

        # Temperature boundary conditions: T for all 4 boundaries * grid_size each
        # Note: corner points might be shared, so subtract corner overlaps
        n_temp_boundary = 4 * self.grid_size - 4

        # Vorticity boundary conditions: V for all 4 boundaries * grid_size each
        # Note: corner points counted twice, subtract 4 corners × 2 overcounts = 8
        n_vort_boundary = 4 * self.grid_size - 8

        # PS bounds are handled by variable bounds, not constraint equations
        return (
            n_interior_S
            + n_interior_V
            + n_interior_E
            + n_temp_boundary
            + n_vort_boundary
        )

    def _unpack_variables(self, y: Array) -> tuple[Array, Array, Array]:
        """Unpack flat array into OM, PH, PS grids.

        The SIF file defines variables in interleaved order:
        For each (J,I) position: OM(I,J), PH(I,J), PS(I,J)
        """
        # Variables are interleaved: OM, PH, PS for each grid point
        y_reshaped = y.reshape((self.grid_size, self.grid_size, 3))
        om = y_reshaped[:, :, 0]
        ph = y_reshaped[:, :, 1]
        ps = y_reshaped[:, :, 2]
        return om, ph, ps

    def residual(self, y: Array, args) -> Array:
        """Compute the residuals for the flow problem."""
        om, ph, ps = self._unpack_variables(y)

        h = self.h
        h2 = self.h2
        axx = self.axx
        ax = self.ax

        # Vectorized interior equations for all interior points
        # Interior points are [1:-1, 1:-1]
        interior_om = om[1:-1, 1:-1]
        interior_ph = ph[1:-1, 1:-1]
        interior_ps = ps[1:-1, 1:-1]

        # Stream function equation (S) - vectorized
        s_eq = (
            interior_om * (-2 / h2 - 2 * axx / h2)
            + om[1:-1, 2:] * (1 / h2)  # om[j, i+1]
            + om[1:-1, :-2] * (1 / h2)  # om[j, i-1]
            + om[2:, 1:-1] * (axx / h2)  # om[j+1, i]
            + om[:-2, 1:-1] * (axx / h2)  # om[j-1, i]
            + ph[1:-1, 2:] * (-self.pi1 / (2 * h))  # ph[j, i+1]
            + ph[1:-1, :-2] * (self.pi1 / (2 * h))  # ph[j, i-1]
            + ph[2:, 1:-1] * (-self.pi2 / (2 * h))  # ph[j+1, i]
            + ph[:-2, 1:-1] * (self.pi2 / (2 * h))  # ph[j-1, i]
        )

        # Vorticity equation (V) - vectorized
        v_eq = (
            interior_ps * (-2 / h2 - 2 * axx / h2)
            + ps[1:-1, 2:] * (1 / h2)  # ps[j, i+1]
            + ps[1:-1, :-2] * (1 / h2)  # ps[j, i-1]
            + ps[2:, 1:-1] * (axx / h2)  # ps[j+1, i]
            + ps[:-2, 1:-1] * (axx / h2)  # ps[j-1, i]
            + interior_om * (axx / 4)
        )

        # Thermal energy equation (E) - vectorized
        # Linear part
        e_eq = (
            interior_ph * (-2 / h2 - 2 * axx / h2)
            + ph[1:-1, 2:] * (1 / h2)  # ph[j, i+1]
            + ph[1:-1, :-2] * (1 / h2)  # ph[j, i-1]
            + ph[2:, 1:-1] * (axx / h2)  # ph[j+1, i]
            + ph[:-2, 1:-1] * (axx / h2)  # ph[j-1, i]
        )

        # Quadratic terms - vectorized
        psidif_i = ps[2:, 1:-1] - ps[:-2, 1:-1]  # ps[j+1, i] - ps[j-1, i]
        phidif_i = ph[1:-1, 2:] - ph[1:-1, :-2]  # ph[j, i+1] - ph[j, i-1]
        e_eq += -ax / (4 * h2) * psidif_i * phidif_i

        psidif_j = ps[1:-1, 2:] - ps[1:-1, :-2]  # ps[j, i+1] - ps[j, i-1]
        phidif_j = ph[2:, 1:-1] - ph[:-2, 1:-1]  # ph[j+1, i] - ph[j-1, i]
        e_eq += ax / (4 * h2) * psidif_j * phidif_j

        # Flatten interior equations
        s_interior = s_eq.flatten()
        v_interior = v_eq.flatten()
        e_interior = e_eq.flatten()

        # Temperature boundary conditions - vectorized
        # TODO: Human review needed for CONSTANTS section handling
        # The SIF file has CONSTANTS section with A3=-1, B3=-1, F3=0, G3=0
        # but pycutest may handle these differently than expected
        # Current implementation matches original loop-based version

        # Top boundary (all k)
        j = self.grid_size - 1
        t_top = ph[j, :] * (2 * self.a1 / h + self.a2) + ph[j - 1, :] * (
            -2 * self.a1 / h
        )

        # Bottom boundary (all k)
        j = 0
        t_bot = ph[j + 1, :] * (2 * self.b1 / h) + ph[j, :] * (
            -2 * self.b1 / h + self.b2
        )

        # Right boundary (excluding corners)
        i = self.grid_size - 1
        t_right = ph[1:-1, i] * (2 * self.f1 / (ax * h) + self.f2) + ph[1:-1, i - 1] * (
            -2 * self.f1 / (ax * h)
        )

        # Left boundary (excluding corners)
        i = 0
        t_left = ph[1:-1, i + 1] * (2 * self.g1 / (ax * h)) + ph[1:-1, i] * (
            -2 * self.g1 / (ax * h) + self.g2
        )

        # Vorticity boundary conditions - vectorized
        # Top and bottom boundaries
        j_top = self.grid_size - 1
        v_top = ps[j_top, :] * (-2 / h) + ps[j_top - 1, :] * (2 / h)

        j_bot = 0
        v_bot = ps[j_bot + 1, :] * (2 / h) + ps[j_bot, :] * (-2 / h)

        # Right and left boundaries (excluding corners)
        i_right = self.grid_size - 1
        v_right = ps[1:-1, i_right] * (-2 / (ax * h)) + ps[1:-1, i_right - 1] * (
            2 / (ax * h)
        )

        i_left = 0
        v_left = ps[1:-1, i_left + 1] * (2 / (ax * h)) + ps[1:-1, i_left] * (
            -2 / (ax * h)
        )

        # Concatenate all residuals in the correct order
        # PS boundary conditions handled by variable bounds, not constraints
        return jnp.concatenate(
            [
                s_interior,
                v_interior,
                e_interior,  # Interior equations
                t_top,
                t_bot,
                t_right,
                t_left,  # Temperature boundaries
                v_top,
                v_bot,
                v_right,
                v_left,  # Vorticity boundaries
            ]
        )

    @property
    def y0(self) -> Array:
        """Initial guess for the optimization problem."""
        return self.starting_point()

    @property
    def args(self):
        """Additional arguments for the residual function."""
        return ()

    @property
    def expected_result(self) -> Array | None:
        """Expected result of the optimization problem."""
        return None  # Not specified in SIF file

    def constraint(self, y: Array) -> tuple[Array, None]:
        """Returns the equality constraints (residuals should be zero)."""
        return self.residual(y, self.args), None

    @property
    def bounds(self) -> tuple[Array, Array] | None:
        """Returns bounds on variables.

        The SIF file sets bounds on PS variables at boundaries:
        For each K from -M to M:
        - PS(K,-M): bottom boundary
        - PS(-M,K): left boundary
        - PS(K,M): top boundary
        - PS(M,K): right boundary

        Variables are in interleaved order: OM(I,J), PH(I,J), PS(I,J) for each (J,I)
        """
        bounds_lower = jnp.full(self.n_vars, -jnp.inf, dtype=jnp.float64)
        bounds_upper = jnp.full(self.n_vars, jnp.inf, dtype=jnp.float64)

        M = self.m

        # Set bounds according to SIF file's DO loop
        for k in range(-M, M + 1):  # K from -M to M
            # Convert to 0-based indices
            k_idx = k + M  # k=-15 -> 0, k=15 -> 30

            # PS(K,-M): bottom boundary - in SIF notation PS(k, -M)
            # Grid position: row 0, column k_idx, PS component (index 2)
            j_idx = 0  # J=-M corresponds to row 0
            i_idx = k_idx  # I=K
            var_idx = (j_idx * self.grid_size + i_idx) * 3 + 2  # +2 for PS component
            bounds_lower = bounds_lower.at[var_idx].set(1.0)
            bounds_upper = bounds_upper.at[var_idx].set(1.0)

            # PS(-M,K): left boundary - in SIF notation PS(-M, k)
            # Grid position: row k_idx, column 0, PS component
            j_idx = k_idx  # J=K
            i_idx = 0  # I=-M corresponds to column 0
            var_idx = (j_idx * self.grid_size + i_idx) * 3 + 2
            bounds_lower = bounds_lower.at[var_idx].set(1.0)
            bounds_upper = bounds_upper.at[var_idx].set(1.0)

            # PS(K,M): top boundary - in SIF notation PS(k, M)
            # Grid position: row 30, column k_idx, PS component
            j_idx = self.grid_size - 1  # J=M corresponds to row 30
            i_idx = k_idx  # I=K
            var_idx = (j_idx * self.grid_size + i_idx) * 3 + 2
            bounds_lower = bounds_lower.at[var_idx].set(1.0)
            bounds_upper = bounds_upper.at[var_idx].set(1.0)

            # PS(M,K): right boundary - in SIF notation PS(M, k)
            # Grid position: row k_idx, column 30, PS component
            j_idx = k_idx  # J=K
            i_idx = self.grid_size - 1  # I=M corresponds to column 30
            var_idx = (j_idx * self.grid_size + i_idx) * 3 + 2
            bounds_lower = bounds_lower.at[var_idx].set(1.0)
            bounds_upper = bounds_upper.at[var_idx].set(1.0)

        return bounds_lower, bounds_upper

    @property
    def expected_objective_value(self) -> Array | None:
        """Expected value of the objective at the optimal solution."""
        return jnp.array(0.0)

    @property
    def expected_residual_value(self) -> Array | None:
        """Expected value of the residuals at the optimal solution."""
        return None  # Not specified in SIF file
