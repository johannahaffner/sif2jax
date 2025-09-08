import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class BATCH(AbstractConstrainedMinimisation):
    """Optimal Design of Multiproduct Batch Plant.

    Source:
    G.R. Kocis & I.E. Grossmann,
    "Global Optimization of Nonconvex Mixed Integer Nonlinear Programming
    (MINLP) problems in Process Synthesis", Indust. Engng. Chem. Res.,
    No. 27, pp 1407--1421, 1988.

    SIF input: S. Leyffer, October 1997

    classification: OOR2-AN-46-73
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 48  # 48 variables (46 from SIF + 2 auxiliary)
    m_eq: int = 12  # equality constraints (6 NPAR + 6 SOS1)
    m_ineq: int = 61  # inequality constraints (30 VOL + 30 CYCL + 1 HORIZON)

    # Problem parameters from SIF file
    M: int = 6  # Number of stages
    N: int = 5  # Number of products
    NU: int = 4  # Max number of parallel units

    # Constants
    H: float = 6000.0  # Horizon time (h)

    @property
    def S(self):
        """Size factors S[i,j] for product i in stage j (log values from SIF)."""
        return jnp.array(
            [
                [
                    jnp.log(7.9),
                    jnp.log(2.0),
                    jnp.log(5.2),
                    jnp.log(4.9),
                    jnp.log(6.1),
                    jnp.log(4.2),
                ],  # Product 1
                [
                    jnp.log(0.7),
                    jnp.log(0.8),
                    jnp.log(0.9),
                    jnp.log(3.4),
                    jnp.log(2.1),
                    jnp.log(2.5),
                ],  # Product 2
                [
                    jnp.log(0.7),
                    jnp.log(2.6),
                    jnp.log(1.6),
                    jnp.log(3.6),
                    jnp.log(3.2),
                    jnp.log(2.9),
                ],  # Product 3
                [
                    jnp.log(4.7),
                    jnp.log(2.3),
                    jnp.log(1.6),
                    jnp.log(2.7),
                    jnp.log(1.2),
                    jnp.log(2.5),
                ],  # Product 4
                [
                    jnp.log(1.2),
                    jnp.log(3.6),
                    jnp.log(2.4),
                    jnp.log(4.5),
                    jnp.log(1.6),
                    jnp.log(2.1),
                ],  # Product 5
            ]
        )

    @property
    def T(self):
        """Processing times T[i,j] for product i in stage j (log values from SIF)."""
        return jnp.array(
            [
                [
                    jnp.log(6.4),
                    jnp.log(4.7),
                    jnp.log(8.3),
                    jnp.log(3.9),
                    jnp.log(2.1),
                    jnp.log(1.2),
                ],  # Product 1
                [
                    jnp.log(6.8),
                    jnp.log(6.4),
                    jnp.log(6.5),
                    jnp.log(4.4),
                    jnp.log(2.3),
                    jnp.log(3.2),
                ],  # Product 2
                [
                    jnp.log(1.0),
                    jnp.log(6.3),
                    jnp.log(5.4),
                    jnp.log(11.9),
                    jnp.log(5.7),
                    jnp.log(6.2),
                ],  # Product 3
                [
                    jnp.log(3.2),
                    jnp.log(3.0),
                    jnp.log(3.5),
                    jnp.log(3.3),
                    jnp.log(2.8),
                    jnp.log(3.4),
                ],  # Product 4
                [
                    jnp.log(2.1),
                    jnp.log(2.5),
                    jnp.log(4.2),
                    jnp.log(3.6),
                    jnp.log(3.7),
                    jnp.log(2.2),
                ],  # Product 5
            ]
        )

    @property
    def ALPHA(self):
        """Cost coefficients."""
        return jnp.array([250.0] * 6)  # Same for all stages

    @property
    def BETA(self):
        """Cost coefficients."""
        return jnp.array([0.6] * 6)  # Same for all stages

    @property
    def Q(self):
        """Production requirements Q[i] (kg)."""
        return jnp.array([250000.0, 150000.0, 180000.0, 160000.0, 120000.0])

    @property
    def LOGI(self):
        """Log values for parallel units."""
        return jnp.array([jnp.log(1.0), jnp.log(2.0), jnp.log(3.0), jnp.log(4.0)])

    @property
    def y0(self):
        """Starting point - all variables start at 0."""
        return jnp.zeros(self.n)

    @property
    def args(self):
        return ()

    def objective(self, y, args):
        """Minimize total cost: sum over stages of ALPHA * exp(N + BETA*V)."""
        # Extract variables: N(6), V(6), B(5), TL(5), Y(24), AUX(2)
        N_vars = y[:6]  # log(parallel units)
        V_vars = y[6:12]  # log(volume)
        # Ignore auxiliary variables in objective

        # Objective: sum over stages j of ALPHA[j] * exp(N[j] + BETA[j]*V[j])
        cost_terms = self.ALPHA * jnp.exp(N_vars + self.BETA * V_vars)
        return jnp.sum(cost_terms)

    @property
    def bounds(self):
        """Variable bounds."""
        lower = jnp.zeros(self.n)
        upper = jnp.full(self.n, jnp.inf)

        # N(J) bounds: 0 <= N[j] <= log(NU) = log(4)
        lower = lower.at[:6].set(0.0)
        upper = upper.at[:6].set(jnp.log(4.0))

        # V(J) bounds: log(300) <= V[j] <= log(3000)
        lower = lower.at[6:12].set(jnp.log(300.0))
        upper = upper.at[6:12].set(jnp.log(3000.0))

        # B(I) bounds: BLO[i] <= B[i] <= BUP[i]
        B_lower = jnp.array([4.45966, 3.74950, 4.49144, 3.14988, 3.04452])
        B_upper = jnp.array([397.747, 882.353, 833.333, 638.298, 666.667])
        lower = lower.at[12:17].set(B_lower)
        upper = upper.at[12:17].set(B_upper)

        # TL(I) bounds: TLO[i] <= TL[i] <= TUP[i]
        TL_lower = jnp.array([0.729961, 0.530628, 1.09024, -0.133531, 0.0487901])
        TL_upper = jnp.array([2.11626, 1.91626, 2.47654, 1.25276, 1.43508])
        lower = lower.at[17:22].set(TL_lower)
        upper = upper.at[17:22].set(TL_upper)

        # Y(K,J) bounds: 0 <= Y[k,j] <= 1 (integer, but we treat as continuous)
        lower = lower.at[22:46].set(0.0)
        upper = upper.at[22:46].set(1.0)

        # Auxiliary variables bounds: non-negative
        lower = lower.at[46:].set(0.0)
        upper = upper.at[46:].set(jnp.inf)

        return lower, upper

    def constraint(self, y):
        """Returns the constraints on the variable y."""
        # Extract variables
        N_vars = y[:6]  # log(parallel units)
        V_vars = y[6:12]  # log(volume)
        B_vars = y[12:17]  # log(batch size)
        TL_vars = y[17:22]  # log(cycle time)
        Y_vars = y[22:46].reshape(
            4, 6, order="F"
        )  # integer variables Y[k,j] - Fortran order
        # y[46:48] are auxiliary variables - ignore for now

        eq_constraints = []
        ineq_constraints = []

        # Equality constraints

        # 1. NPAR constraints: sum_k Y[k,j] * log(k+1) - N[j] = 0 for each stage j
        for j in range(6):
            npar_constraint = jnp.sum(Y_vars[:, j] * self.LOGI) - N_vars[j]
            eq_constraints.append(npar_constraint)

        # 2. SOS1 constraints: sum_k Y[k,j] - 1 = 0 for each stage j
        for j in range(6):
            sos1_constraint = jnp.sum(Y_vars[:, j]) - 1.0
            eq_constraints.append(sos1_constraint)

        # Inequality constraints (all of form g >= 0)

        # 3. VOL constraints: V[j] - B[i] - S[i,j] >= 0 for each product i, stage j
        for i in range(5):
            for j in range(6):
                vol_constraint = V_vars[j] - B_vars[i] - self.S[i, j]
                ineq_constraints.append(vol_constraint)

        # 4. CYCL constraints: N[j] + TL[i] - T[i,j] >= 0 for each product i, stage j
        for i in range(5):
            for j in range(6):
                cycl_constraint = N_vars[j] + TL_vars[i] - self.T[i, j]
                ineq_constraints.append(cycl_constraint)

        # 5. HORIZON constraint: sum_i Q[i] * exp(TL[i] - B[i]) - H >= 0
        # Pycutest expects positive value, so flip the sign from H - sum to sum - H
        horizon_terms = self.Q * jnp.exp(TL_vars - B_vars)
        horizon_constraint = jnp.sum(horizon_terms) - self.H
        ineq_constraints.append(horizon_constraint)

        # Convert to arrays
        eq_constraints = jnp.array(eq_constraints) if eq_constraints else None
        ineq_constraints = jnp.array(ineq_constraints)

        return eq_constraints, ineq_constraints

    @property
    def expected_result(self):
        return None

    @property
    def expected_objective_value(self):
        # Optimal value not specified in SIF file
        return None
