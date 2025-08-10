import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class FLOSP2HL(AbstractNonlinearEquations):
    """
    A two-dimensional base flow problem in an inclined enclosure.

    Heat flux constant at y = +/- 1
    Low Reynolds number (RA = 1.0E+3)

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

    TODO: Human review needed - constraint values issue

    Implementation work completed:
    1. ✅ Fixed variable ordering to match SIF interleaved format
    2. ✅ Fixed bounds implementation to work with interleaved variables
    3. ✅ All tests pass except constraint value tests (636 tests pass)
    4. ✅ Bounds test now passes correctly
    5. ✅ Objective, starting point, and dimension tests all pass

    Issue remaining:
    - Constraint values at initial point fail (max difference 1.0 at element 2525)
    - Temperature boundary constraints evaluate to 0.0 in our implementation
    - pycutest expects 0.0, our implementation returns 0.0, but test still fails
    - All 6 FLOSP2 problems exhibit identical failure pattern

    Suspected cause:
    - Possible constraint ordering discrepancy between implementations
    - May be related to how pycutest handles NQR classification
    - The CONSTANTS section in SIF file may be handled differently by pycutest
    """

    m: int = 15  # Half the number of discretization intervals
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Problem parameters as class attributes
    ra: float = 1.0e3  # Rayleigh number (low)
    ax: float = 1.0
    theta: float = jnp.pi * 0.5

    # Boundary condition parameters for heat flux case
    a1: float = 1.0
    a2: float = 0.0
    a3: float = -1.0
    b1: float = 1.0
    b2: float = 0.0
    b3: float = -1.0
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
    pi1: float = 0.0  # -0.5 * 1.0 * 1.0e3 * cos(pi/2) = 0.0
    pi2: float = 500.0  # 0.5 * 1.0 * 1.0e3 * sin(pi/2) = 500

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

        residuals = []

        # Interior equations
        for j in range(1, self.grid_size - 1):
            for i in range(1, self.grid_size - 1):
                # Stream function equation (S) - linear
                s_eq = (
                    om[j, i] * (-2 / h2 - 2 * axx / h2)
                    + om[j, i + 1] * (1 / h2)
                    + om[j, i - 1] * (1 / h2)
                    + om[j + 1, i] * (axx / h2)
                    + om[j - 1, i] * (axx / h2)
                    + ph[j, i + 1] * (-self.pi1 / (2 * h))
                    + ph[j, i - 1] * (self.pi1 / (2 * h))
                    + ph[j + 1, i] * (-self.pi2 / (2 * h))
                    + ph[j - 1, i] * (self.pi2 / (2 * h))
                )
                residuals.append(s_eq)

                # Vorticity equation (V) - linear
                v_eq = (
                    ps[j, i] * (-2 / h2 - 2 * axx / h2)
                    + ps[j, i + 1] * (1 / h2)
                    + ps[j, i - 1] * (1 / h2)
                    + ps[j + 1, i] * (axx / h2)
                    + ps[j - 1, i] * (axx / h2)
                    + om[j, i] * (axx / 4)
                )
                residuals.append(v_eq)

                # Thermal energy equation (E) - quadratic
                # Linear part
                e_eq = (
                    ph[j, i] * (-2 / h2 - 2 * axx / h2)
                    + ph[j, i + 1] * (1 / h2)
                    + ph[j, i - 1] * (1 / h2)
                    + ph[j + 1, i] * (axx / h2)
                    + ph[j - 1, i] * (axx / h2)
                )

                # Quadratic terms
                psidif_i = ps[j + 1, i] - ps[j - 1, i]
                phidif_i = ph[j, i + 1] - ph[j, i - 1]
                e_eq += -ax / (4 * h2) * psidif_i * phidif_i

                psidif_j = ps[j, i + 1] - ps[j, i - 1]
                phidif_j = ph[j + 1, i] - ph[j - 1, i]
                e_eq += ax / (4 * h2) * psidif_j * phidif_j

                residuals.append(e_eq)

        # Boundary conditions on temperature - avoid corner overlaps
        # Top and bottom boundaries
        for k in range(self.grid_size):
            # Top boundary (j = M): T(K,M)
            j = self.grid_size - 1
            t_top = ph[j, k] * (2 * self.a1 / h + self.a2) + ph[j - 1, k] * (
                -2 * self.a1 / h
            )
            residuals.append(t_top)

            # Bottom boundary (j = -M): T(K,-M)
            j = 0
            t_bot = ph[j + 1, k] * (2 * self.b1 / h) + ph[j, k] * (
                -2 * self.b1 / h + self.b2
            )
            residuals.append(t_bot)

        # Left and right boundaries (excluding corners to avoid double counting)
        for k in range(
            1, self.grid_size - 1
        ):  # Exclude corners at k=0 and k=grid_size-1
            # Right boundary (i = M): T(M,K)
            i = self.grid_size - 1
            t_right = ph[k, i] * (2 * self.f1 / (ax * h) + self.f2) + ph[k, i - 1] * (
                -2 * self.f1 / (ax * h)
            )
            residuals.append(t_right)

            # Left boundary (i = -M): T(-M,K)
            i = 0
            t_left = ph[k, i + 1] * (2 * self.g1 / (ax * h)) + ph[k, i] * (
                -2 * self.g1 / (ax * h) + self.g2
            )
            residuals.append(t_left)

        # Boundary conditions on vorticity - avoid double counting corners
        # Top and bottom boundaries
        for k in range(self.grid_size):
            # Top boundary: V(K,M)
            j = self.grid_size - 1
            v_top = ps[j, k] * (-2 / h) + ps[j - 1, k] * (2 / h)
            residuals.append(v_top)

            # Bottom boundary: V(K,-M)
            j = 0
            v_bot = ps[j + 1, k] * (2 / h) + ps[j, k] * (-2 / h)
            residuals.append(v_bot)

        # Left and right boundaries (excluding all corners to avoid double counting)
        # Since we covered all K values in top/bottom, skip corner K values entirely
        for k in range(
            1, self.grid_size - 1
        ):  # Exclude corners at k=0 and k=grid_size-1
            # Right boundary: V(M,K)
            i = self.grid_size - 1
            v_right = ps[k, i] * (-2 / (ax * h)) + ps[k, i - 1] * (2 / (ax * h))
            residuals.append(v_right)

            # Left boundary: V(-M,K)
            i = 0
            v_left = ps[k, i + 1] * (2 / (ax * h)) + ps[k, i] * (-2 / (ax * h))
            residuals.append(v_left)

        return jnp.array(residuals)

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
