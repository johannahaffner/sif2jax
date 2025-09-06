import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class GASOIL(AbstractConstrainedMinimisation):
    """
    GASOIL - Catalytic cracking of gas oil optimization problem.

    # TODO: Human review needed
    # Attempts made: [
    #   1. Initial implementation based on SIF file structure
    #   2. Analyzed SIF collocation formulation with Legendre polynomials
    #   3. Implemented variable structure: theta + V + W + U + DU
    #   4. Set up objective with measurement residuals and polynomial corrections
    #   5. Implemented constraint system for ODE collocation method
    # ]
    # Suspected issues: [
    #   - Variable count mismatch (10,403 vs pycutest expected)
    #   - Complex optimal control collocation formulation not matching SIF exactly
    #   - Starting point calculation needs refinement for large-scale problem
    #   - Objective function polynomial correction terms may have wrong structure
    #   - Constraint indexing and time point associations need validation
    # ]
    # Resources needed: [
    #   - Optimal control theory expertise for collocation methods
    #   - Detailed analysis of SIF ELEMENT/GROUP sections for this problem type
    #   - Reference implementation or AMPL .mod file for comparison
    #   - Understanding of COPS collection problem 12 specifics
    # ]

    Determine the reaction coefficients for the catalytic cracking of gas oil
    and other byproducts. The nonlinear model that describes the process is:

        y_1' = -(theta_1 + theta_3) * y_1^2
        y_2' = theta_1 * y_1^2 + theta_2 * y_2

    with given initial conditions. The problem is to minimize:

        sum_{i=1,20} ||y(tau_i, theta) - z_i||^2

    where the z_i are concentration measurements for y at times tau_i (i=1,20).

    This is problem 12 in the COPS (Version 2) collection of
    E. Dolan and J. More'
    see "Benchmarking Optimization Software with COPS"
    Argonne National Labs Technical Report ANL/MCS-246 (2000)

    Source: Nick Gould, November 2000

    Classification: OOR2-AN-V-V

    SIF input: Nick Gould, November 2000.
    """

    # Problem parameters from SIF file
    ne: int = 2  # Number of differential equations
    nh: int = 400  # Number of subintervals
    np: int = 3  # Number of ODE parameters
    nm: int = 21  # Number of measurements
    nc: int = 4  # Number of collocation points

    @property
    def n_var(self) -> int:
        """Total number of variables."""
        # theta (np=3) + V (nh*ne) + W (nh*nc*ne) + U (nh*nc*ne) + DU (nh*nc*ne)
        return self.np + self.nh * self.ne + 3 * (self.nh * self.nc * self.ne)

    @property
    def n_con(self) -> int:
        """Total number of constraints."""
        # U constraints + DU constraints + continuity + collocation constraints
        n_u_constraints = self.nh * self.nc * self.ne
        n_du_constraints = self.nh * self.nc * self.ne
        n_continuity = (self.nh - 1) * self.ne
        n_collocation = self.nh * self.nc * self.ne
        return n_u_constraints + n_du_constraints + n_continuity + n_collocation

    # Required attributes
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Time points for observations
    _tau = jnp.array(
        [
            0.0,
            0.025,
            0.05,
            0.075,
            0.10,
            0.125,
            0.150,
            0.175,
            0.20,
            0.225,
            0.250,
            0.30,
            0.35,
            0.40,
            0.45,
            0.50,
            0.55,
            0.65,
            0.75,
            0.85,
            0.95,
        ]
    )

    # Measurement data (z_i values from SIF)
    _z_data = jnp.array(
        [
            [1.0000, 0.0000],
            [0.8105, 0.2000],
            [0.6208, 0.2886],
            [0.5258, 0.3010],
            [0.4345, 0.3215],
            [0.3903, 0.3123],
            [0.3342, 0.2716],
            [0.3034, 0.2551],
            [0.2735, 0.2258],
            [0.2405, 0.1959],
            [0.2283, 0.1789],
            [0.2071, 0.1457],
            [0.1669, 0.1198],
            [0.1530, 0.0909],
            [0.1339, 0.0719],
            [0.1265, 0.0561],
            [0.1200, 0.0460],
            [0.0990, 0.0280],
            [0.0870, 0.0190],
            [0.0770, 0.0140],
            [0.0690, 0.0100],
        ]
    )

    # Legendre polynomial roots for nc=4
    _rho = jnp.array([0.0694318442, 0.3300094782, 0.6699905218, 0.9305681558])

    # ODE initial conditions
    _bc = jnp.array([1.0, 0.0])

    @property
    def initial_guess(self) -> jnp.ndarray:
        """Initial guess from SIF file."""
        y0 = jnp.zeros(self.n_var)

        # Set theta parameters to 0.0
        y0 = y0.at[: self.np].set(0.0)

        # Initialize V variables based on SIF starting point logic
        nh, ne, nc = self.nh, self.ne, self.nc  # noqa: F841
        v_start = self.np

        # Compute initial values for V based on measurement data
        tf = self._tau[-1]  # Final time = 0.95
        h = tf / nh  # Uniform interval length

        # Initialize V with boundary conditions and measurement interpolations
        v_vals = jnp.zeros((nh, ne))

        # Set initial values based on measurements
        for i in range(nh):
            t_i = i * h
            # Find closest measurement time
            idx = jnp.argmin(jnp.abs(self._tau - t_i))
            if i == 0:
                v_vals = v_vals.at[i, :].set(self._bc)
            else:
                v_vals = v_vals.at[i, :].set(self._z_data[idx])

        # Flatten and set V values
        y0 = y0.at[v_start : v_start + nh * ne].set(v_vals.ravel())

        # Initialize W, U, DU to zero (as per SIF)
        # W starts after V, U after W, DU after U
        w_start = v_start + nh * ne  # noqa: F841
        # u_start = w_start + nh * nc * ne  # Not used in current implementation
        # du_start = u_start + nh * nc * ne  # Not used in current implementation

        # Set all remaining variables to 0
        return y0

    @property
    def y0(self) -> jnp.ndarray:
        """Initial guess."""
        return self.initial_guess

    @property
    def bounds(self):
        """Variable bounds."""
        n_var = self.n_var
        lower = jnp.full(n_var, -jnp.inf)
        upper = jnp.full(n_var, jnp.inf)

        # Theta parameters have lower bound of 0.0
        lower = lower.at[: self.np].set(0.0)

        # V variables have fixed values at boundaries (handled in constraints)
        # No explicit bounds for other variables

        return lower, upper

    def objective(self, y: jnp.ndarray, args=None) -> jnp.ndarray:
        """
        Objective function: sum of squared residuals between model and measurements.
        """
        nh, ne, nc, nm = self.nh, self.ne, self.nc, self.nm

        # Extract variables
        theta = y[: self.np]  # noqa: F841 (used in constraint method)
        v_start = self.np
        v = y[v_start : v_start + nh * ne].reshape((nh, ne))

        w_start = v_start + nh * ne
        w = y[w_start : w_start + nh * nc * ne].reshape((nh, nc, ne))

        # Compute objective: sum of squared differences
        # obj(j,s) = v[itau[j],s] - z[j,s] + polynomial correction terms

        tf = self._tau[-1]
        h = tf / nh

        # Compute factorial values
        fact = jnp.array([1.0, 1.0, 2.0, 6.0])  # fact[0] to fact[3]

        obj_sum = 0.0

        for j in range(nm):
            tau_j = self._tau[j]
            # Find interval index
            itau_j = jnp.minimum(nh - 1, jnp.floor(tau_j / h).astype(int))

            # Time at interval start
            t_itau = itau_j * h
            diff = tau_j - t_itau

            for s in range(ne):
                # Base term: v[itau[j], s] - z[j, s]
                residual = v[itau_j, s] - self._z_data[j, s]

                # Add polynomial correction terms
                ratio = diff
                for k in range(nc):
                    coef = ratio / fact[k]
                    if k > 0:
                        coef = coef / (h ** (k - 1))
                    residual += w[itau_j, k, s] * coef
                    ratio *= diff

                # Add squared residual
                obj_sum += residual**2

        return jnp.array(obj_sum)

    def constraint(self, y: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Constraint functions for the optimal control problem.
        """
        # Extract theta parameters for collocation constraints
        theta = y[: self.np]
        nh, ne, nc, np = self.nh, self.ne, self.nc, self.np

        # Extract variables (theta not used in objective)
        v_start = np
        v = y[v_start : v_start + nh * ne].reshape((nh, ne))

        w_start = v_start + nh * ne
        w = y[w_start : w_start + nh * nc * ne].reshape((nh, nc, ne))

        u_start = w_start + nh * nc * ne
        u = y[u_start : u_start + nh * nc * ne].reshape((nh, nc, ne))

        du_start = u_start + nh * nc * ne
        du = y[du_start : du_start + nh * nc * ne].reshape((nh, nc, ne))

        tf = self._tau[-1]
        h = tf / nh

        # Factorial values
        fact = jnp.array([1.0, 1.0, 2.0, 6.0])

        constraints = []

        # U constraints: -u[i,j,s] + v[i,s] + h*sum_k w[i,k,s]*(rho[j]^k/fact[k]) = 0
        for i in range(nh):
            for j in range(nc):
                rho_j = self._rho[j]
                for s in range(ne):
                    u_constraint = -u[i, j, s] + v[i, s]

                    # Add polynomial terms
                    prod = h
                    for k in range(nc):
                        coef = prod / fact[k]
                        u_constraint += w[i, k, s] * coef
                        prod *= rho_j

                    constraints.append(u_constraint)

        # DU constraints: -du[i,j,s] + sum_k w[i,k,s]*(rho[j]^(k-1)/fact[k-1]) = 0
        for i in range(nh):
            for j in range(nc):
                rho_j = self._rho[j]
                for s in range(ne):
                    du_constraint = -du[i, j, s]

                    # Add polynomial terms
                    prod = 1.0
                    for k in range(nc):
                        if k > 0:
                            coef = prod / fact[k - 1]
                            du_constraint += w[i, k, s] * coef
                        prod *= rho_j

                    constraints.append(du_constraint)

        # Continuity constraints: v[i,s] - v[i+1,s] + sum_j w[i,j,s]*h/fact[j] = 0
        for i in range(nh - 1):
            for s in range(ne):
                cont_constraint = v[i, s] - v[i + 1, s]

                for j in range(nc):
                    coef = h / fact[j]
                    cont_constraint += w[i, j, s] * coef

                constraints.append(cont_constraint)

        # Collocation constraints (nonlinear ODE terms)
        for i in range(nh):
            for j in range(nc):
                # Constraint 1: du[i,j,0] + (theta[0]+theta[2])*u[i,j,0]^2 = 0
                col_constraint1 = du[i, j, 0] + (theta[0] + theta[2]) * u[i, j, 0] ** 2
                constraints.append(col_constraint1)

                # Constraint 2: du[i,j,1] - theta[0]*u[i,j,0]^2 + theta[1]*u[i,j,1] = 0
                col_constraint2 = (
                    du[i, j, 1] - theta[0] * u[i, j, 0] ** 2 + theta[1] * u[i, j, 1]
                )
                constraints.append(col_constraint2)

        c_eq = jnp.array(constraints)
        c_ineq = jnp.array([])  # No inequality constraints

        return c_eq, c_ineq

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value from SIF file."""
        return jnp.array(5.23659e-03)  # NH=400

    @property
    def expected_result(self):
        """Expected result from SIF file."""
        return None
