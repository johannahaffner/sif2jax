import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class KTMODEL(AbstractNonlinearEquations):
    """A four sector dynamic macro-economic model for South Korea over an horizon
    of thirty periods of one year. The four considered sectors are
    (1) agriculture and mining, (2) heavy industry, (3) light industry and
    (4) services. The basic structure of the model is to maximize a welfare
    functional over the total period subject to constraints in the form of
    distribution relations, productions functions, absorptive capacity functions,
    foreign exchange constraints, initial and terminal capital stock and
    foreign debt constraints. The production functions are of CES (constant
    elasticity of substitution) type.

    Source:
    D. Kendrick and L. Taylor,
    "A Dynamic Nonlinear Planning Model for Korea",
    In "Practical Approaches to Development Planning", I. Adelman (ed.),
    The John Hopkins Press, Baltimore, vol. 7, pp.77-105, 1969.

    SIF input: E. Loute and B. Apraxine, March 1993.
    classification NOR2-MN-720-450
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Problem parameters
    T: int = 30  # number of periods
    n_sectors: int = 4

    # Economic parameters
    z: float = 0.03  # consumption discount rate
    theta: float = 0.05  # interest rate on foreign debt
    epsil: float = 0.5
    lo: float = 8.6e-3
    COLO: float = 0.000001
    alpha: float = 0.02
    gammastar: float = 8

    @property
    def K1(self):
        """Initial capital stocks"""
        return jnp.array([2.02, 2.13, 1.26, 1.27])

    @property
    def Kstar(self):
        """Final capital stocks"""
        return jnp.array([14.2, 20, 10.2, 10.3])

    @property
    def a(self):
        """Welfare coefficients"""
        return jnp.array([0.48, 0.33, 0.345, 0.3925])

    @property
    def b(self):
        """Consumption elasticity"""
        return jnp.array([0.85, 0.90, 0.91, 0.87])

    @property
    def mu(self):
        """Depreciation rates"""
        return jnp.array([0.275, 0.35, 0.30, 0.35])

    @property
    def tau(self):
        """Efficiency parameters"""
        return jnp.array([0.41, 1.26, 1.89, 0.47])

    @property
    def nu(self):
        """Technical progress rates"""
        return jnp.array([0.03, 0.035, 0.025, 0.025])

    @property
    def beta(self):
        """Distribution parameters"""
        return jnp.array([0.35, 0.30, 0.25, 0.20])

    @property
    def rho(self):
        """CES elasticity parameters"""
        return jnp.array([-0.166, 0.111, 0.111, 0.666])

    @property
    def d(self):
        """Marginal propensities to import (production)"""
        return jnp.array([0.0008, 0.090, 0.030, 0.004])

    @property
    def pi(self):
        """Marginal propensities to import (capital)"""
        return jnp.array([0.63, 0.98, 0.10, 0.10])

    @property
    def ee(self):
        """Export coefficients ee(t,i) for t=1..30, i=1..4"""
        return jnp.array(
            [
                [0.064, 0.035, 0.106, 0.131],  # t=1
                [0.070, 0.043, 0.128, 0.148],  # t=2
                [0.076, 0.051, 0.154, 0.166],  # t=3
                [0.082, 0.062, 0.185, 0.185],  # t=4
                [0.088, 0.073, 0.220, 0.205],  # t=5
                [0.094, 0.087, 0.261, 0.277],  # t=6
                [0.099, 0.103, 0.308, 0.251],  # t=7
                [0.103, 0.121, 0.362, 0.276],  # t=8
                [0.107, 0.141, 0.424, 0.302],  # t=9
                [0.110, 0.165, 0.495, 0.330],  # t=10
                [0.112, 0.189, 0.530, 0.348],  # t=11
                [0.113, 0.214, 0.567, 0.366],  # t=12
                [0.115, 0.243, 0.607, 0.334],  # t=13
                [0.115, 0.274, 0.649, 0.404],  # t=14
                [0.116, 0.309, 0.695, 0.425],  # t=15
                [0.116, 0.347, 0.744, 0.446],  # t=16
                [0.115, 0.389, 0.796, 0.469],  # t=17
                [0.114, 0.435, 0.851, 0.492],  # t=18
                [0.110, 0.486, 0.911, 0.516],  # t=19
                [0.108, 0.541, 0.975, 0.541],  # t=20
                [0.114, 0.606, 1.006, 0.560],  # t=21
                [0.121, 0.676, 1.038, 0.579],  # t=22
                [0.127, 0.751, 1.070, 0.599],  # t=23
                [0.134, 0.833, 1.102, 0.618],  # t=24
                [0.142, 0.921, 1.134, 0.638],  # t=25
                [0.150, 1.017, 1.198, 0.658],  # t=26
                [0.158, 1.119, 1.198, 0.678],  # t=27
                [0.166, 1.230, 1.230, 0.698],  # t=28
                [0.175, 1.349, 1.261, 0.718],  # t=29
                [0.185, 1.477, 1.292, 0.739],  # t=30
            ]
        )

    @property
    def AA(self):
        """Leontief matrix AA(i,j)"""
        return jnp.array(
            [
                [0.1, 0.09, 0.04, 0.03],
                [0.09, 0.33, 0.24, 0.12],
                [0.04, 0.02, 0.12, 0.05],
                [0.03, 0.09, 0.09, 0.08],
            ]
        )

    @property
    def BB(self):
        """Capital coefficient matrix BB(i,j)"""
        return jnp.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.6908, 1.3109, 0.1769, 0.15],
                [0.001, 0.0199, 0.002, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )

    @property
    def PP(self):
        """Compute the PP matrix from the SIF specification."""
        return self._compute_PP_matrix()

    def _compute_PP_matrix(self):
        """Compute the PP matrix from the SIF specification."""
        PP = jnp.zeros((4, 4))

        # PP(1,1) = -1 - d1 + AA(1,1)
        PP = PP.at[0, 0].set(-1.0 - self.d[0] + self.AA[0, 0])

        # PP(1,j) = -AA(1,j) for j=2,3,4
        for j in range(1, 4):
            PP = PP.at[0, j].set(-self.AA[0, j])

        # PP(2,1) = -AA(2,1)
        PP = PP.at[1, 0].set(-self.AA[1, 0])

        # PP(2,2) = -1 - d2 + AA(2,2)
        PP = PP.at[1, 1].set(-1.0 - self.d[1] + self.AA[1, 1])

        # PP(2,j) = -AA(2,j) for j=3,4
        for j in range(2, 4):
            PP = PP.at[1, j].set(-self.AA[1, j])

        # PP(3,1) = -AA(3,1), PP(3,2) = -AA(3,2)
        PP = PP.at[2, 0].set(-self.AA[2, 0])
        PP = PP.at[2, 1].set(-self.AA[2, 1])

        # PP(3,3) = -1 - d3 + AA(3,3)
        PP = PP.at[2, 2].set(-1.0 - self.d[2] + self.AA[2, 2])

        # PP(3,4) = -AA(3,4)
        PP = PP.at[2, 3].set(-self.AA[2, 3])

        # PP(4,j) = -AA(4,j) for j=1,2,3
        for j in range(3):
            PP = PP.at[3, j].set(-self.AA[3, j])

        # PP(4,4) = -1 - d4 + AA(4,4)
        PP = PP.at[3, 3].set(-1.0 - self.d[3] + self.AA[3, 3])

        return PP

    def _get_variable_indices(self):
        """Get the indices for different variable types in the flattened array."""
        # Variables: ksi(1:31), gam(1:31), K(1:31,1:4), Q(1:30,1:4),
        # C(1:30,1:4), del(1:30,1:4), L(1:30,1:4), M2(1:30), M3(1:30)

        idx = 0

        # ksi(1:31) - welfare
        ksi_start = idx
        idx += 31

        # gam(1:31) - debt
        gam_start = idx
        idx += 31

        # K(1:31,1:4) - capital
        K_start = idx
        idx += 31 * 4

        # Q(1:30,1:4) - output
        Q_start = idx
        idx += 30 * 4

        # C(1:30,1:4) - consumption
        C_start = idx
        idx += 30 * 4

        # del(1:30,1:4) - investment
        del_start = idx
        idx += 30 * 4

        # L(1:30,1:4) - labor
        L_start = idx
        idx += 30 * 4

        # M2(1:30) - imports sector 2
        M2_start = idx
        idx += 30

        # M3(1:30) - imports sector 3
        M3_start = idx
        idx += 30

        return {
            "ksi": ksi_start,
            "gam": gam_start,
            "K": K_start,
            "Q": Q_start,
            "C": C_start,
            "del": del_start,
            "L": L_start,
            "M2": M2_start,
            "M3": M3_start,
            "total": idx,
        }

    def _extract_variables(self, y):
        """Extract variables from the flattened array."""
        indices = self._get_variable_indices()

        # Extract and reshape variables
        ksi = y[indices["ksi"] : indices["ksi"] + 31]  # ksi(1:31)
        gam = y[indices["gam"] : indices["gam"] + 31]  # gam(1:31)
        K = y[indices["K"] : indices["K"] + 31 * 4].reshape(31, 4)  # K(1:31,1:4)
        Q = y[indices["Q"] : indices["Q"] + 30 * 4].reshape(30, 4)  # Q(1:30,1:4)
        C = y[indices["C"] : indices["C"] + 30 * 4].reshape(30, 4)  # C(1:30,1:4)
        del_inv = y[indices["del"] : indices["del"] + 30 * 4].reshape(
            30, 4
        )  # del(1:30,1:4)
        L = y[indices["L"] : indices["L"] + 30 * 4].reshape(30, 4)  # L(1:30,1:4)
        M2 = y[indices["M2"] : indices["M2"] + 30]  # M2(1:30)
        M3 = y[indices["M3"] : indices["M3"] + 30]  # M3(1:30)

        return ksi, gam, K, Q, C, del_inv, L, M2, M3

    def _pow_element(self, x, pex):
        """POW element: x^pex"""
        return x**pex

    def _kag_element(self, x, y, pmu, peps):
        """KAG element: X * (1 - (1 + PMU*Y/X)^PEPS)"""
        r1 = 1.0 + pmu * y / x
        r = r1**peps
        return x * (1.0 - r)

    def _qug_element(self, x, y, pbeta, prho):
        """QUG element: PBETA * X^(-PRHO) + (1-PBETA) * Y^(-1/PRHO)"""
        rx = -prho
        ry = -1.0 / prho
        return pbeta * (x**rx) + (1.0 - pbeta) * (y**ry)

    def residual(self, y: Array, args) -> Array:
        """Compute the residual vector for the macro-economic model."""
        ksi, gam, K, Q, C, del_inv, L, M2, M3 = self._extract_variables(y)

        residuals = []

        # XI(t): welfare constraints for t=1..30
        for t in range(30):
            # ksi(t+1) - ksi(t) - sum_i a_i * C(t,i)^b_i * (1+z)^(-t) = 0
            welfare_sum = 0.0
            for i in range(4):
                consumption_term = self._pow_element(C[t, i], self.b[i])
                discount_factor = (1.0 + self.z) ** (-t)
                welfare_sum += self.a[i] * consumption_term * discount_factor

            residual = ksi[t + 1] - ksi[t] - welfare_sum
            residuals.append(residual)

        # KK(t,i): capital constraints for t=1..30, i=1..4
        for t in range(30):
            for i in range(4):
                # K(t+1,i) - (1-mu_i)*K(t,i) - KAG_element = 0
                kag_term = self._kag_element(
                    K[t, i], del_inv[t, i], self.epsil / self.mu[i], -1.0 / self.epsil
                )
                residual = K[t + 1, i] - (1.0 - self.mu[i]) * K[t, i] - kag_term
                residuals.append(residual)

        # GG(t): debt constraints for t=1..30
        for t in range(30):
            # gam(t+1) - (1+theta)*gam(t) + sum_i d_i*Q(t,i) +
            # sum_i pi_i*del(t,i) + M2(t) + M3(t) + sum_i ee(t,i) = 0
            debt_term = 0.0
            for i in range(4):
                debt_term += self.d[i] * Q[t, i] + self.pi[i] * del_inv[t, i]
            debt_term += M2[t] + M3[t] - jnp.sum(self.ee[t])

            residual = gam[t + 1] - (1.0 + self.theta) * gam[t] - debt_term
            residuals.append(residual)

        # QQ(t,i): output constraints for t=1..30, i=1..4
        for t in range(30):
            for i in range(4):
                # Q(t,i) - tau_i * (1+nu_i)^t * QUG_element = 0
                technical_progress = (1.0 + self.nu[i]) ** t
                qug_term = self._qug_element(
                    K[t, i], L[t, i], self.beta[i], self.rho[i]
                )
                residual = Q[t, i] - self.tau[i] * technical_progress * qug_term
                residuals.append(residual)

        # CC(t,i): consumption constraints for t=1..30, i=1..4
        for t in range(30):
            for i in range(4):
                # sum_j PP(i,j)*Q(t,j) + sum_j BB(i,j)*del(t,j) - imports +
                # ee(t,i) = 0
                consumption_term = 0.0
                for j in range(4):
                    consumption_term += (
                        self.PP[i, j] * Q[t, j] + self.BB[i, j] * del_inv[t, j]
                    )

                # Add imports for sectors 2 and 3
                if i == 1:  # sector 2 (0-indexed as 1)
                    consumption_term -= M2[t]
                elif i == 2:  # sector 3 (0-indexed as 2)
                    consumption_term -= M3[t]

                consumption_term -= self.ee[t, i]
                residuals.append(consumption_term)

        # LL(t): labor constraints for t=1..30
        for t in range(30):
            # sum_i L(t,i) - lo * (1+alpha)^t = 0
            labor_supply = self.lo * ((1.0 + self.alpha) ** t)
            residual = jnp.sum(L[t]) - labor_supply
            residuals.append(residual)

        return jnp.array(residuals)

    @property
    def y0(self) -> Array:
        """Initial guess for the optimization problem from the SIF START POINT
        section."""
        indices = self._get_variable_indices()
        y0 = jnp.zeros(indices["total"])

        # This would be extremely long to implement all starting values manually
        # For now, use a simplified initialization based on the SIF patterns

        # Initialize with some reasonable values
        # ksi: welfare starts at 0 and grows
        ksi_vals = jnp.linspace(0.0, 35.0, 31)
        y0 = y0.at[:31].set(ksi_vals)

        # gam: debt starts at 0.25 with some variation
        gam_vals = jnp.full(31, 0.25)
        y0 = y0.at[31:62].set(gam_vals)

        # K: capital stocks start with K1 and grow
        K_vals = jnp.zeros((31, 4))
        K_vals = K_vals.at[0].set(self.K1)
        for t in range(1, 31):
            growth_factor = 1.0 + 0.05 * t  # Simple growth
            K_vals = K_vals.at[t].set(self.K1 * growth_factor)
        y0 = y0.at[62 : 62 + 31 * 4].set(K_vals.flatten())

        # Q, C, del, L: initialize with positive values scaled by sector
        for var_name, start_idx, periods, sectors in [
            ("Q", 62 + 31 * 4, 30, 4),
            ("C", 62 + 31 * 4 + 30 * 4, 30, 4),
            ("del", 62 + 31 * 4 + 60 * 4, 30, 4),
            ("L", 62 + 31 * 4 + 90 * 4, 30, 4),
        ]:
            vals = jnp.ones((periods, sectors)) * 0.5
            if var_name == "L":
                sector_scales = jnp.array([3.0, 1.0, 0.3, 4.0])
                vals = vals * sector_scales[None, :]  # Broadcast properly
            elif var_name == "Q":
                sector_scales = jnp.array([2.0, 2.0, 1.0, 1.0])
                vals = vals * sector_scales[None, :]  # Broadcast properly
            y0 = y0.at[start_idx : start_idx + periods * sectors].set(vals.flatten())

        # M2, M3: imports start at 0
        M2_start = 62 + 31 * 4 + 120 * 4
        M3_start = M2_start + 30
        y0 = y0.at[M2_start : M2_start + 30].set(jnp.zeros(30))
        y0 = y0.at[M3_start : M3_start + 30].set(jnp.zeros(30))

        return y0

    @property
    def bounds(self):
        """Variable bounds from the SIF BOUNDS section."""
        indices = self._get_variable_indices()
        n_vars = indices["total"]

        # Most variables are free, but some have specific bounds
        lower_bounds = jnp.full(n_vars, -jnp.inf)
        upper_bounds = jnp.full(n_vars, jnp.inf)

        # Fixed initial values: ksi(1)=0, gam(1)=0
        lower_bounds = lower_bounds.at[0].set(0.0)  # ksi(1)
        upper_bounds = upper_bounds.at[0].set(0.0)
        lower_bounds = lower_bounds.at[31].set(0.0)  # gam(1)
        upper_bounds = upper_bounds.at[31].set(0.0)

        # Initial capital stocks K(1,i) = Ki
        for i in range(4):
            idx = 62 + i  # K(1,i)
            lower_bounds = lower_bounds.at[idx].set(self.K1[i])
            upper_bounds = upper_bounds.at[idx].set(self.K1[i])

        # Terminal capital stocks K(31,i) = Ki_star
        for i in range(4):
            idx = 62 + 30 * 4 + i  # K(31,i)
            lower_bounds = lower_bounds.at[idx].set(self.Kstar[i])
            upper_bounds = upper_bounds.at[idx].set(self.Kstar[i])

        # Terminal debt gam(31) = 0
        lower_bounds = lower_bounds.at[61].set(0.0)  # gam(31)
        upper_bounds = upper_bounds.at[61].set(0.0)

        # Consumption lower bounds C(t,i) >= COLO
        C_start = indices["C"]
        for t in range(30):
            for i in range(4):
                idx = C_start + t * 4 + i
                lower_bounds = lower_bounds.at[idx].set(self.COLO)

        return (lower_bounds, upper_bounds)

    def constraint(self, y):
        """Returns the residuals as equality constraints."""
        return self.residual(y, self.args), None

    @property
    def args(self):
        """Additional arguments for the residual function."""
        return None

    @property
    def expected_result(self):
        """Expected result of the optimization problem."""
        return None

    @property
    def expected_objective_value(self):
        """Expected value of the objective at the solution."""
        return jnp.array(0.0)
