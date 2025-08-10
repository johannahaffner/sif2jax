import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class TWIRISM1(AbstractConstrainedMinimisation):
    """
    TWIRISM1 - Small nuclear reactor core reload pattern optimization problem.

    # TODO: Human review needed
    # Attempts made: [
    #   1. Fixed initial values from SIF file (hardcoded pycutest array)
    #   2. Implemented basic constraint structure
    #      (Sumi, Sumlm, Norm, Kern, Burn, Plac, Dia)
    #   3. Added complete G matrix (76 non-zero entries) and V vector from SIF
    #   4. Fixed k/phi variable extraction (node-major, interleaved per time step)
    #   5. Corrected bounds (Kfresh=1.2 for this problem)
    #   6. Implemented trilinear elements xa, xb, xc using einsum operations
    # ]
    # Suspected issues: [
    #   - Trilinear ELEMENT variables (xa,xb,xc) formulation not matching
    #     SIF exactly
    #   - xa(i,j,m) = x(i,2,m) * x(j,1,m) * k(j,t_eoc) structure needs
    #     validation
    #   - Constraint values differ from pycutest (Sumi/Sumlm match ~1e-4,
    #     others have larger discrepancies)
    #   - Complex group-separable structure requires deep SIF ELEMENT analysis
    #   - All test categories fail: constraints, constraint Jacobians
    # ]
    # Resources needed: [
    #   - Detailed analysis of SIF ELEMENT and GROUP sections
    #   - Reference implementation or documentation for trilinear nuclear
    #     reactor constraints
    #   - Verification of variable indexing and time step associations
    #   - Comparison with AMPL .mod file if available
    # ]

    A relaxation of a nonlinear integer programming problem for finding
    optimal nuclear reactor core reload patterns (small size version).

    Source:
    A.J. Quist, E. de Klerk, C. Roos, T. Terlaky, R. van Geemert,
    J.E. Hoogenboom, T.Illes,
    "Finding Optimal Nuclear Reactor Core Reload Patterns Using Nonlinear
    Optimization and Search Heuristics",
    Delft University, January 1998.

    classification LOI2-RN-343-313

    SIF input: Arie Quist, Delft, 1998.
    """

    # Problem dimensions
    Nnod: int = 14  # Number of nodes
    Nred: int = 12  # Number of reduced nodes
    Ndia: int = 2  # Number of diagonal sets
    Ntim: int = 6  # Number of time steps
    Nage: int = 4  # Number of age groups
    Ntra: int = 3  # Number of trajectories

    # Required attributes for the abstract class
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Pre-computed constants
    Kfresh: float = 1.2
    alp_Pc_Dt: float = 0.01

    @property
    def initial_guess(self) -> jnp.ndarray:
        """Initial guess matching pycutest structure exactly."""
        # Use the exact pycutest x0 array
        pycutest_x0 = jnp.array(
            [
                0.0333,
                0.0333,
                0.0333,
                0.1849,
                0.1849,
                0.1849,
                0.1151,
                0.1151,
                0.1151,
                0.0,
                0.0,
                0.0,
                0.0333,
                0.0333,
                0.0333,
                0.1849,
                0.1849,
                0.1849,
                0.1151,
                0.1151,
                0.1151,
                0.0,
                0.0,
                0.0,
                0.1,
                0.1,
                0.1,
                0.1349,
                0.1349,
                0.1349,
                0.0651,
                0.0651,
                0.0651,
                0.0333,
                0.0333,
                0.0333,
                0.1,
                0.1,
                0.1,
                0.1349,
                0.1349,
                0.1349,
                0.0651,
                0.0651,
                0.0651,
                0.0333,
                0.0333,
                0.0333,
                0.1,
                0.1,
                0.1,
                0.1349,
                0.1349,
                0.1349,
                0.0651,
                0.0651,
                0.0651,
                0.0333,
                0.0333,
                0.0333,
                0.0,
                0.0,
                0.0,
                0.0083,
                0.0083,
                0.0083,
                0.025,
                0.025,
                0.025,
                0.3,
                0.3,
                0.3,
                0.0333,
                0.0333,
                0.0333,
                0.1849,
                0.1849,
                0.1849,
                0.1151,
                0.1151,
                0.1151,
                0.0,
                0.0,
                0.0,
                0.3,
                0.3,
                0.3,
                0.0083,
                0.0083,
                0.0083,
                0.025,
                0.025,
                0.025,
                0.0,
                0.0,
                0.0,
                0.3,
                0.3,
                0.3,
                0.0083,
                0.0083,
                0.0083,
                0.025,
                0.025,
                0.025,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0333,
                0.0333,
                0.0333,
                0.1186,
                0.1186,
                0.1186,
                0.1814,
                0.1814,
                0.1814,
                0.0333,
                0.0333,
                0.0333,
                0.1256,
                0.1256,
                0.1256,
                0.0558,
                0.0558,
                0.0558,
                0.1186,
                0.1186,
                0.1186,
                0.0,
                0.0,
                0.0,
                0.0333,
                0.0333,
                0.0333,
                0.3,
                0.3,
                0.3,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0083,
                0.0083,
                0.0083,
                0.025,
                0.025,
                0.025,
                0.3,
                0.3,
                0.3,
                0.0333,
                0.0333,
                0.0333,
                0.1256,
                0.1256,
                0.1256,
                0.0558,
                0.0558,
                0.0558,
                0.1186,
                0.1186,
                0.1186,
                1.0231,
                0.0855,
                1.0098,
                0.088,
                0.9962,
                0.0903,
                0.9824,
                0.0922,
                0.9686,
                0.0939,
                0.9547,
                0.0954,
                1.0231,
                0.0997,
                1.0075,
                0.1008,
                0.992,
                0.1016,
                0.9766,
                0.1023,
                0.9613,
                0.1028,
                0.9462,
                0.1031,
                1.0595,
                0.1206,
                1.04,
                0.1207,
                1.0208,
                0.1205,
                1.002,
                0.1203,
                0.9836,
                0.1199,
                0.9655,
                0.1195,
                1.0595,
                0.1008,
                1.0432,
                0.1032,
                1.0267,
                0.1054,
                1.0102,
                0.1076,
                0.9936,
                0.1095,
                0.9769,
                0.1113,
                1.0595,
                0.0534,
                1.0509,
                0.0573,
                1.0417,
                0.0612,
                1.0319,
                0.0653,
                1.0216,
                0.0695,
                1.0108,
                0.0738,
                0.9546,
                0.0147,
                0.9524,
                0.0165,
                0.95,
                0.0184,
                0.9474,
                0.0204,
                0.9444,
                0.0227,
                0.9411,
                0.0251,
                1.0231,
                0.1194,
                1.0045,
                0.1185,
                0.9863,
                0.1176,
                0.9685,
                0.1166,
                0.9513,
                0.1156,
                0.9344,
                0.1147,
                1.1779,
                0.1505,
                1.1508,
                0.1483,
                1.1247,
                0.1461,
                1.0996,
                0.1441,
                1.0753,
                0.1422,
                1.052,
                0.1403,
                1.1779,
                0.1151,
                1.1572,
                0.1166,
                1.1365,
                0.118,
                1.116,
                0.1193,
                1.0957,
                0.1206,
                1.0755,
                0.1218,
                0.9636,
                0.0409,
                0.9576,
                0.0433,
                0.9513,
                0.0458,
                0.9446,
                0.0484,
                0.9376,
                0.0511,
                0.9303,
                0.0539,
                1.0074,
                0.098,
                0.9923,
                0.0982,
                0.9774,
                0.0985,
                0.9627,
                0.0989,
                0.9481,
                0.0993,
                0.9338,
                0.0997,
                0.97,
                0.0559,
                0.9617,
                0.0578,
                0.9533,
                0.0598,
                0.9445,
                0.0619,
                0.9356,
                0.0641,
                0.9264,
                0.0664,
                0.9546,
                0.0207,
                0.9516,
                0.0222,
                0.9483,
                0.0238,
                0.9449,
                0.0255,
                0.9412,
                0.0273,
                0.9373,
                0.0293,
                1.0074,
                0.0245,
                1.0036,
                0.0262,
                0.9996,
                0.028,
                0.9953,
                0.0299,
                0.9908,
                0.032,
                0.9859,
                0.0343,
                1.0558,
                1.0373,
                1.0193,
                1.0017,
                0.9846,
                0.9678,
                0.01,
            ]
        )

        return pycutest_x0

    @property
    def y0(self) -> jnp.ndarray:
        """Initial guess."""
        return self.initial_guess

    @property
    def bounds(self):
        """Variable bounds matching pycutest structure."""
        Nnod = self.Nnod
        Ntim = self.Ntim
        Nage = self.Nage
        Ntra = self.Ntra

        # Sizes
        n_x = Nnod * Nage * Ntra  # 168
        n_kphi = 2 * Nnod * Ntim  # 168 (interleaved k and phi)
        n_keff = Ntim  # 6
        n_eps = 1  # 1

        # Physical constants
        Kfresh = 1.2

        # Bounds for x variables: [0, 1]
        x_lower = jnp.zeros(n_x)
        x_upper = jnp.ones(n_x)

        # Bounds for interleaved k and phi
        # Create interleaved bounds array
        kphi_lower = jnp.zeros(n_kphi)
        kphi_upper = jnp.zeros(n_kphi)

        # Set bounds for k and phi in interleaved pattern
        for i in range(Nnod):
            for t in range(Ntim):
                idx = 2 * (i * Ntim + t)
                # k bounds: [0, Kfresh]
                kphi_lower = kphi_lower.at[idx].set(0.0)
                kphi_upper = kphi_upper.at[idx].set(Kfresh)
                # phi bounds: [0.0001, 1.0]
                kphi_lower = kphi_lower.at[idx + 1].set(0.0001)
                kphi_upper = kphi_upper.at[idx + 1].set(1.0)

        # Bounds for keff: [0, Kfresh]
        keff_lower = jnp.zeros(n_keff)
        keff_upper = jnp.full(n_keff, Kfresh)

        # Bounds for epsilon: [0, 100]
        eps_lower = jnp.zeros(n_eps)
        eps_upper = jnp.full(n_eps, 100.0)

        lower = jnp.concatenate([x_lower, kphi_lower, keff_lower, eps_lower])
        upper = jnp.concatenate([x_upper, kphi_upper, keff_upper, eps_upper])

        return lower, upper

    def objective(self, y: jnp.ndarray, args=None) -> jnp.ndarray:
        """Objective: maximize keff(Ntim-1) - w1*epsilon."""
        Nnod = self.Nnod
        Ntim = self.Ntim
        Nage = self.Nage
        Ntra = self.Ntra

        # Extract variables matching pycutest structure
        n_x = Nnod * Nage * Ntra  # 168
        n_kphi = 2 * Nnod * Ntim  # 168 (interleaved k and phi)

        keff_start = n_x + n_kphi
        keff = y[keff_start : keff_start + Ntim]
        epsilon = y[-1]

        # Weight for epsilon penalty
        w1 = 1.0

        # Maximize keff(Ntim-1) which means minimize -keff(Ntim-1)
        return -keff[Ntim - 1] + w1 * epsilon

    def constraint(self, y: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Constraints for the nuclear reactor problem."""
        Nnod = self.Nnod
        Ntim = self.Ntim
        Nage = self.Nage
        Ntra = self.Ntra

        # Extract variables matching pycutest's interleaved structure
        n_x = Nnod * Nage * Ntra  # 168
        x = y[:n_x].reshape((Nnod, Nage, Ntra))

        # k and phi are interleaved in pycutest
        # Each node has: k[i,0], phi[i,0], k[i,1], phi[i,1], k[i,2], phi[i,2], ...
        n_kphi = 2 * Nnod * Ntim  # 168
        kphi_section = y[n_x : n_x + n_kphi]
        kphi_node_major = kphi_section.reshape((Nnod, 2 * Ntim))
        k = kphi_node_major[:, ::2]  # Every other column starting from 0 (k values)
        phi = kphi_node_major[
            :, 1::2
        ]  # Every other column starting from 1 (phi values)

        keff = y[n_x + n_kphi : n_x + n_kphi + Ntim]
        epsilon = y[-1]

        # Constants
        V = jnp.array(
            [0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 0.5]
        )
        G = _build_G_matrix()
        alp_Pc_Dt = self.alp_Pc_Dt
        Fl_Nred = 1.0 / self.Nred

        # Trilinear elements from SIF ELEMENT definitions
        # xa(i,j,m) = x(i,2,m) * x(j,1,m) * k(j,t) where t needs to be determined
        # xb(i,j,m) = x(i,3,m) * x(j,2,m) * k(j,t)
        # xc(i,j,m) = x(i,4,m) * x(j,3,m) * k(j,t)
        # From SIF comment "k(j,EoC)" suggests End of Cycle = last time step
        # Try t = Ntim-1 (last time step, 0-indexed)
        t_eoc = Ntim - 1  # End of Cycle
        xa = jnp.einsum(
            "im,jm,j->ijm", x[:, 1, :], x[:, 0, :], k[:, t_eoc]
        )  # x(i,2,m) * x(j,1,m) * k(j,EoC)
        xb = jnp.einsum(
            "im,jm,j->ijm", x[:, 2, :], x[:, 1, :], k[:, t_eoc]
        )  # x(i,3,m) * x(j,2,m) * k(j,EoC)
        xc = jnp.einsum(
            "im,jm,j->ijm", x[:, 3, :], x[:, 2, :], k[:, t_eoc]
        )  # x(i,4,m) * x(j,3,m) * k(j,EoC)

        # Vectorized constraints

        # Sumi constraints: sum over i of V(i)*x(i,l,m) = 1 for each l,m
        # Shape: (Nage, Ntra)
        sumi = jnp.einsum("i,ilm->lm", V, x) - 1.0
        sumi_flat = sumi.ravel()  # Flatten to 1D

        # Sumlm constraints: sum over l,m of x(i,l,m) = 1 for each i
        # Shape: (Nnod,)
        sumlm = jnp.sum(x, axis=(1, 2)) - 1.0

        # Norm constraints: sum over i of V(i)*k(i,t)*phi(i,t) = 1 for each t
        # Shape: (Ntim,)
        norm = jnp.einsum("i,it,it->t", V, k, phi) - 1.0

        # Kern constraints: keff(t)*phi(i,t) + sum_j G(i,j)*k(j,t)*phi(j,t) = 0
        # Shape: (Nnod, Ntim)
        kern = keff[None, :] * phi + jnp.einsum("ij,jt,jt->it", G, k, phi)
        kern_flat = kern.ravel()  # Flatten to 1D

        # Peak constraints: k(i,t)*phi(i,t) - epsilon/Fl_Nred <= 0
        # Shape: (Nnod, Ntim)
        peak = k * phi - epsilon / Fl_Nred
        peak_flat = peak.ravel()  # Flatten to 1D

        # Burn constraints: k(i,t+1) - k(i,t) + alp_Pc_Dt*k(i,t)*phi(i,t) = 0
        # Shape: (Nnod, Ntim-1)
        burn = k[:, 1:] - k[:, :-1] + alp_Pc_Dt * k[:, :-1] * phi[:, :-1]
        burn_flat = burn.ravel()  # Flatten to 1D

        # Plac constraints: k(i,1) - sum_m x(i,1,m)*Kfresh -
        # sum_{j,m} V(j)*[xa(i,j,m) + xb(i,j,m) + xc(i,j,m)] = 0
        # Note: k(i,1) in SIF is 1-indexed, so k(i,0) in 0-indexed
        # Shape: (Nnod,)
        trilinear_sum = jnp.einsum(
            "j,ijm->i", V, xa + xb + xc
        )  # sum_{j,m} V(j)*[xa + xb + xc]
        plac = k[:, 0] - jnp.sum(x[:, 0, :], axis=1) * self.Kfresh - trilinear_sum

        # Diagonal constraints: x(d1,l,m) = x(d2,l,m) for diagonal pairs
        # From SIF: D1,1=1 D2,1=11, D1,2=7 D2,2=14
        # Python uses 0-indexed, so: (0,10) and (6,13)
        dia1 = x[0, :, :] - x[10, :, :]  # First diagonal pair
        dia2 = x[6, :, :] - x[13, :, :]  # Second diagonal pair
        dia_flat = jnp.concatenate([dia1.ravel(), dia2.ravel()])

        # Kefford constraints: keff(t+1) - keff(t) <= 0
        # Shape: (Ntim-1,)
        kefford = keff[1:] - keff[:-1]

        # Concatenate all equality constraints
        c_eq = jnp.concatenate(
            [sumi_flat, sumlm, norm, kern_flat, burn_flat, plac, dia_flat]
        )

        # Concatenate all inequality constraints
        c_ineq = jnp.concatenate([peak_flat, kefford])

        return c_eq, c_ineq

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value from SIF file."""
        return None

    @property
    def expected_result(self):
        """Expected result from SIF file."""
        return None


def _build_G_matrix() -> jnp.ndarray:
    """Build the G matrix from SIF values."""
    G = jnp.zeros((14, 14))
    G = G.at[0, 0].set(-0.828)
    G = G.at[1, 0].set(-0.079)
    G = G.at[6, 0].set(-0.014)
    G = G.at[0, 1].set(-0.158)
    G = G.at[1, 1].set(-0.763)
    G = G.at[2, 1].set(-0.079)
    G = G.at[6, 1].set(-0.13)
    G = G.at[7, 1].set(-0.014)
    G = G.at[1, 2].set(-0.079)
    G = G.at[2, 2].set(-0.749)
    G = G.at[3, 2].set(-0.079)
    G = G.at[6, 2].set(-0.028)
    G = G.at[7, 2].set(-0.065)
    G = G.at[8, 2].set(-0.014)
    G = G.at[2, 3].set(-0.079)
    G = G.at[3, 3].set(-0.749)
    G = G.at[4, 3].set(-0.079)
    G = G.at[7, 3].set(-0.014)
    G = G.at[8, 3].set(-0.065)
    G = G.at[9, 3].set(-0.014)
    G = G.at[3, 4].set(-0.079)
    G = G.at[4, 4].set(-0.749)
    G = G.at[5, 4].set(-0.079)
    G = G.at[8, 4].set(-0.014)
    G = G.at[9, 4].set(-0.065)
    G = G.at[4, 5].set(-0.079)
    G = G.at[5, 5].set(-0.749)
    G = G.at[9, 5].set(-0.014)
    G = G.at[0, 6].set(-0.014)
    G = G.at[1, 6].set(-0.065)
    G = G.at[2, 6].set(-0.014)
    G = G.at[6, 6].set(-0.684)
    G = G.at[7, 6].set(-0.065)
    G = G.at[10, 6].set(-0.014)
    G = G.at[1, 7].set(-0.014)
    G = G.at[2, 7].set(-0.065)
    G = G.at[3, 7].set(-0.014)
    G = G.at[6, 7].set(-0.13)
    G = G.at[7, 7].set(-0.698)
    G = G.at[8, 7].set(-0.065)
    G = G.at[10, 7].set(-0.13)
    G = G.at[11, 7].set(-0.014)
    G = G.at[2, 8].set(-0.014)
    G = G.at[3, 8].set(-0.065)
    G = G.at[4, 8].set(-0.014)
    G = G.at[7, 8].set(-0.065)
    G = G.at[8, 8].set(-0.684)
    G = G.at[9, 8].set(-0.065)
    G = G.at[10, 8].set(-0.028)
    G = G.at[11, 8].set(-0.065)
    G = G.at[12, 8].set(-0.014)
    G = G.at[3, 9].set(-0.014)
    G = G.at[4, 9].set(-0.065)
    G = G.at[5, 9].set(-0.014)
    G = G.at[8, 9].set(-0.065)
    G = G.at[9, 9].set(-0.684)
    G = G.at[11, 9].set(-0.014)
    G = G.at[12, 9].set(-0.065)
    G = G.at[6, 10].set(-0.014)
    G = G.at[7, 10].set(-0.065)
    G = G.at[8, 10].set(-0.014)
    G = G.at[10, 10].set(-0.684)
    G = G.at[11, 10].set(-0.065)
    G = G.at[13, 10].set(-0.014)
    G = G.at[7, 11].set(-0.014)
    G = G.at[8, 11].set(-0.065)
    G = G.at[9, 11].set(-0.014)
    G = G.at[10, 11].set(-0.13)
    G = G.at[11, 11].set(-0.698)
    G = G.at[12, 11].set(-0.065)
    G = G.at[13, 11].set(-0.13)
    G = G.at[8, 12].set(-0.014)
    G = G.at[9, 12].set(-0.065)
    G = G.at[11, 12].set(-0.065)
    G = G.at[12, 12].set(-0.684)
    G = G.at[13, 12].set(-0.028)
    G = G.at[10, 13].set(-0.014)
    G = G.at[11, 13].set(-0.065)
    G = G.at[12, 13].set(-0.014)
    G = G.at[13, 13].set(-0.684)
    return G
