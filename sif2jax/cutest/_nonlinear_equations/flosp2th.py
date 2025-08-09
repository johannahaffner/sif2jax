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
        # Interior equations + boundary conditions
        n_interior = (self.grid_size - 2) * (self.grid_size - 2)
        n_boundary = 4 * self.grid_size
        return 3 * n_interior + 2 * n_boundary

    def _unpack_variables(self, y: Array) -> tuple[Array, Array, Array]:
        """Unpack flat array into OM, PH, PS grids."""
        grid_points = self.grid_size * self.grid_size
        om = y[:grid_points].reshape((self.grid_size, self.grid_size))
        ph = y[grid_points : 2 * grid_points].reshape((self.grid_size, self.grid_size))
        ps = y[2 * grid_points :].reshape((self.grid_size, self.grid_size))
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

        # Boundary conditions on temperature
        for k in range(self.grid_size):
            # Top boundary (j = M)
            j = self.grid_size - 1
            t_top = (
                ph[j, k] * (2 * self.a1 / h + self.a2)
                + ph[j - 1, k] * (-2 * self.a1 / h)
                - self.a3
            )
            residuals.append(t_top)

            # Bottom boundary (j = -M)
            j = 0
            t_bot = (
                ph[j + 1, k] * (2 * self.b1 / h)
                + ph[j, k] * (-2 * self.b1 / h + self.b2)
                - self.b3
            )
            residuals.append(t_bot)

            # Right boundary (i = M)
            i = self.grid_size - 1
            t_right = (
                ph[k, i] * (2 * self.f1 / (ax * h) + self.f2)
                + ph[k, i - 1] * (-2 * self.f1 / (ax * h))
                - self.f3
            )
            residuals.append(t_right)

            # Left boundary (i = -M)
            i = 0
            t_left = (
                ph[k, i + 1] * (2 * self.g1 / (ax * h))
                + ph[k, i] * (-2 * self.g1 / (ax * h) + self.g2)
                - self.g3
            )
            residuals.append(t_left)

        # Boundary conditions on vorticity
        for k in range(self.grid_size):
            # Top boundary
            j = self.grid_size - 1
            v_top = ps[j, k] * (-2 / h) + ps[j - 1, k] * (2 / h)
            residuals.append(v_top)

            # Bottom boundary
            j = 0
            v_bot = ps[j + 1, k] * (2 / h) + ps[j, k] * (-2 / h)
            residuals.append(v_bot)

            # Right boundary
            i = self.grid_size - 1
            v_right = ps[k, i] * (-2 / (ax * h)) + ps[k, i - 1] * (2 / (ax * h))
            residuals.append(v_right)

            # Left boundary
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
        """Returns bounds on variables."""
        # Stream function boundary conditions from SIF file
        ps_bounds_lower = jnp.full(self.n_vars, -jnp.inf, dtype=jnp.float64)
        ps_bounds_upper = jnp.full(self.n_vars, jnp.inf, dtype=jnp.float64)

        # Set PS boundary values to fixed values (XX bounds in SIF)
        grid_points = self.grid_size * self.grid_size
        for k in range(self.grid_size):
            # PS(K,-M) = PS(K,M) = PS(-M,K) = PS(M,K) = 1.0
            idx_bottom = (2 * grid_points) + (0 * self.grid_size + k)
            idx_top = (2 * grid_points) + ((self.grid_size - 1) * self.grid_size + k)
            idx_left = (2 * grid_points) + (k * self.grid_size + 0)
            idx_right = (2 * grid_points) + (k * self.grid_size + (self.grid_size - 1))

            ps_bounds_lower = ps_bounds_lower.at[idx_bottom].set(1.0)
            ps_bounds_upper = ps_bounds_upper.at[idx_bottom].set(1.0)
            ps_bounds_lower = ps_bounds_lower.at[idx_top].set(1.0)
            ps_bounds_upper = ps_bounds_upper.at[idx_top].set(1.0)
            ps_bounds_lower = ps_bounds_lower.at[idx_left].set(1.0)
            ps_bounds_upper = ps_bounds_upper.at[idx_left].set(1.0)
            ps_bounds_lower = ps_bounds_lower.at[idx_right].set(1.0)
            ps_bounds_upper = ps_bounds_upper.at[idx_right].set(1.0)

        return ps_bounds_lower, ps_bounds_upper

    @property
    def expected_objective_value(self) -> Array | None:
        """Expected value of the objective at the optimal solution."""
        return jnp.array(0.0)

    @property
    def expected_residual_value(self) -> Array | None:
        """Expected value of the residuals at the optimal solution."""
        return None  # Not specified in SIF file
