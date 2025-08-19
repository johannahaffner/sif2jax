import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class BDRY2(AbstractConstrainedMinimisation):
    """Boundary value problem - BDRY2.

    The bdry2_0 & _1.mod AMPL models from Hans Mittelmann (mittelmann@asu.edu)
    See: http://plato.asu.edu/ftp/barrier/

    SIF input: Nick Gould, April 25th 2012

    Classification: QLR2-AN-V-V

    This is a discretized boundary value problem on a square grid.
    Grid size N can be 3, 499, 1999, or 3499.
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Grid parameter - default small size for testing
    N: int = 3  # Can be 3, 499, 1999, 3499

    @property
    def n_var(self):
        """Number of variables: (N+2) x (N+2) grid points."""
        return (self.N + 2) ** 2

    @property
    def n_con(self):
        """Number of constraints: N^2 PDE equations + 3*N boundary conditions."""
        return self.N**2 + 3 * self.N

    def _get_grid_indices(self):
        """Get grid parameters and special indices."""
        N1 = self.N + 1
        N1_4 = N1 // 4
        three_N1_4 = 3 * N1_4
        return N1, N1_4, three_N1_4

    def _get_parameters(self):
        """Get problem parameters."""
        A = 0.0
        G = 0.0
        T = 5.0
        N1 = self.N + 1
        h = 1.0 / N1
        h2 = h * h
        half_h2 = 0.5 * h2
        half_ha = 0.5 * h * A
        one_minus_h = 1.0 - h
        gh2 = G * h2
        ht = h * T

        return {
            "A": A,
            "G": G,
            "T": T,
            "h": h,
            "h2": h2,
            "half_h2": half_h2,
            "half_ha": half_ha,
            "one_minus_h": one_minus_h,
            "gh2": gh2,
            "ht": ht,
        }

    def _var_index(self, i, j):
        """Convert 2D grid indices to 1D variable index."""
        N1 = self.N + 1
        return i * (N1 + 1) + j

    def objective(self, y, args):
        """Quadratic objective function."""
        del args

        N1, N1_4, three_N1_4 = self._get_grid_indices()
        params = self._get_parameters()

        obj = 0.0

        # Sum over interior box region
        for i in range(N1_4, three_N1_4 + 1):
            for j in range(N1_4, three_N1_4 + 1):
                idx = self._var_index(i, j)
                # SQD element: (Y - YP)^2 with YP = 1.0
                obj += params["half_h2"] * (y[idx] - 1.0) ** 2

        # Sum over top boundary
        for i in range(1, self.N + 1):
            idx = self._var_index(i, N1)
            # SQ element: U^2
            obj += params["half_ha"] * y[idx] ** 2

        return jnp.array(obj)

    @property
    def y0(self):
        """Initial point - default values within bounds."""
        N1 = self.N + 1
        x0 = jnp.ones(self.n_var) * 5.0  # Middle of [0, 10] range

        # Set corner values
        x0 = x0.at[self._var_index(0, 0)].set(0.0)
        x0 = x0.at[self._var_index(N1, 0)].set(0.0)
        x0 = x0.at[self._var_index(0, N1)].set(10.0)
        x0 = x0.at[self._var_index(N1, N1)].set(10.0)

        # Set interior box values closer to upper bound
        N1_4 = N1 // 4
        three_N1_4 = 3 * N1_4
        for i in range(N1_4, three_N1_4 + 1):
            for j in range(N1_4, three_N1_4 + 1):
                x0 = x0.at[self._var_index(i, j)].set(1.5)

        return x0

    @property
    def args(self):
        return None

    @property
    def bounds(self):
        """Variable bounds."""
        N1, N1_4, three_N1_4 = self._get_grid_indices()

        # Default bounds [0, 10]
        lower = jnp.zeros(self.n_var)
        upper = jnp.ones(self.n_var) * 10.0

        # Tighter upper bounds for interior box
        for i in range(N1_4, three_N1_4 + 1):
            for j in range(N1_4, three_N1_4 + 1):
                idx = self._var_index(i, j)
                upper = upper.at[idx].set(3.15)

        # Fixed corner values
        for i, j, val in [(0, 0, 0.0), (N1, 0, 0.0), (0, N1, 10.0), (N1, N1, 10.0)]:
            idx = self._var_index(i, j)
            lower = lower.at[idx].set(val)
            upper = upper.at[idx].set(val)

        return lower, upper

    def constraint(self, y):
        """Linear constraints from PDE discretization and boundary conditions."""
        params = self._get_parameters()
        N1 = self.N + 1

        equality_constraints = []

        # Interior PDE constraints: P(i,j)
        # 4*X(i,j) - X(i,j+1) - X(i,j-1) - X(i+1,j) - X(i-1,j) = gh2
        for i in range(1, self.N + 1):
            for j in range(1, self.N + 1):
                idx_ij = self._var_index(i, j)
                idx_ip = self._var_index(i + 1, j)
                idx_im = self._var_index(i - 1, j)
                idx_jp = self._var_index(i, j + 1)
                idx_jm = self._var_index(i, j - 1)

                constraint = (
                    4.0 * y[idx_ij]
                    - y[idx_jp]
                    - y[idx_jm]
                    - y[idx_ip]
                    - y[idx_im]
                    - params["gh2"]
                )
                equality_constraints.append(constraint)

        # Boundary constraints B1(i): X(i,0) - X(i,1) = 0
        for i in range(1, self.N + 1):
            idx_0 = self._var_index(i, 0)
            idx_1 = self._var_index(i, 1)
            constraint = y[idx_0] - y[idx_1]
            equality_constraints.append(constraint)

        # Boundary constraints B2(i): (1-h)*X(0,i) - X(1,i) = -ht
        for i in range(1, self.N + 1):
            idx_0 = self._var_index(0, i)
            idx_1 = self._var_index(1, i)
            constraint = params["one_minus_h"] * y[idx_0] - y[idx_1] + params["ht"]
            equality_constraints.append(constraint)

        # Boundary constraints B3(i): (1-h)*X(N1,i) - X(N,i) = -ht
        for i in range(1, self.N + 1):
            idx_N1 = self._var_index(N1, i)
            idx_N = self._var_index(self.N, i)
            constraint = params["one_minus_h"] * y[idx_N1] - y[idx_N] + params["ht"]
            equality_constraints.append(constraint)

        return jnp.array(equality_constraints), None

    @property
    def expected_result(self):
        """Expected solution (not provided in SIF file)."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value (not provided in SIF file)."""
        return None
