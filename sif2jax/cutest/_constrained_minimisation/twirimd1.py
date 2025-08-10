import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class TWIRIMD1(AbstractConstrainedMinimisation):
    """
    TWIRIMD1 - Medium nuclear reactor core reload pattern optimization problem.

    # TODO: Human review needed
    # Attempts made: [
    #   1. Fixed initial values from SIF file (parsed and stored 1247 exact
    #      values)
    #   2. Fixed Kfresh value from 1.2 to 1.3 as per SIF file
    #   3. Implemented constraint structure matching TWIRISM1 pattern
    #      (Sumi, Sumlm, Norm, Kern, Burn, Plac, Dia)
    #   4. Added complete G matrix (31x31, hundreds of entries) and V vector
    #      from SIF
    #   5. Fixed bounds to use correct Kfresh=1.3
    #   6. Added diagonal constraints (3 pairs for 31 nodes)
    #   7. Implemented trilinear elements xa, xb, xc using same einsum
    #      structure as TWIRISM1
    # ]
    # Suspected issues: [
    #   - Trilinear constraint formulation not matching pycutest
    #     (max diff ~0.034)
    #   - Scaling/indexing issues in constraint evaluation
    #   - Complex group-separable structure requires SIF ELEMENT analysis
    #   - All constraint-related tests fail despite correct dimensions/bounds
    #   - Same trilinear issues as TWIRISM1 but larger scale (31 nodes vs 14)
    # ]
    # Resources needed: [
    #   - Same as TWIRISM1: detailed SIF ELEMENT analysis
    #   - Verification of medium-scale problem structure vs small scale
    #   - Reference for 31-node reactor geometry and constraint relationships
    # ]

    A relaxation of a nonlinear integer programming problem for finding
    optimal nuclear reactor core reload patterns (medium size version).

    Source:
    A.J. Quist, E. de Klerk, C. Roos, T. Terlaky, R. van Geemert,
    J.E. Hoogenboom, T.Illes,
    "Finding Optimal Nuclear Reactor Core Reload Patterns Using Nonlinear
    Optimization and Search Heuristics",
    Delft University, January 1998.

    classification LOI2-RN-1247-544

    SIF input: Arie Quist, Delft, 1998.
    """

    # Problem dimensions
    Nnod: int = 31  # Number of nodes
    Nred: int = 28  # Number of reduced nodes
    Ndia: int = 3  # Number of diagonal sets
    Ntim: int = 6  # Number of time steps
    Nage: int = 4  # Number of age groups
    Ntra: int = 7  # Number of trajectories (Nred / Nage = 28 / 4 = 7)

    # Required attributes for the abstract class
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Pre-computed constants
    Kfresh: float = 1.3  # From SIF file
    alp_Pc_Dt: float = 0.01

    @property
    def initial_guess(self) -> jnp.ndarray:
        """Initial guess for TWIRIMD1 - exact values from SIF file."""
        # Load complete precomputed initial values from the numpy file
        import os

        import numpy as np

        # Try to load from numpy file if it exists
        npy_path = os.path.join(os.path.dirname(__file__), "twirimd1_y0.npy")
        if os.path.exists(npy_path):
            return jnp.array(np.load(npy_path))

        # Fallback to hardcoded values (abbreviated - full array has 1247
        # elements)
        # These are the exact values from pycutest/SIF file
        return _get_twirimd1_initial_values()

    @property
    def y0(self) -> jnp.ndarray:
        """Initial guess."""
        return self.initial_guess

    @property
    def bounds(self):
        """Variable bounds for TWIRIMD1."""
        Nnod = self.Nnod
        Ntim = self.Ntim
        Nage = self.Nage
        Ntra = self.Ntra

        # Sizes
        n_x = Nnod * Nage * Ntra
        n_kphi = 2 * Nnod * Ntim
        n_keff = Ntim
        n_eps = 1

        # Physical constants
        Kfresh = self.Kfresh  # 1.3

        # Bounds for x variables: [0, 1]
        x_lower = jnp.zeros(n_x)
        x_upper = jnp.ones(n_x)

        # Bounds for interleaved k and phi
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

        # Extract variables
        n_x = Nnod * Nage * Ntra
        n_kphi = 2 * Nnod * Ntim

        keff_start = n_x + n_kphi
        keff = y[keff_start : keff_start + Ntim]
        epsilon = y[-1]

        # Weight for epsilon penalty
        w1 = 1.0

        # Maximize keff(Ntim-1) which means minimize -keff(Ntim-1)
        return -keff[Ntim - 1] + w1 * epsilon

    def constraint(self, y: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Constraints for the medium nuclear reactor problem."""
        Nnod = self.Nnod
        Ntim = self.Ntim
        Nage = self.Nage
        Ntra = self.Ntra

        # Extract variables matching expected structure
        n_x = Nnod * Nage * Ntra
        x = y[:n_x].reshape((Nnod, Nage, Ntra))

        # k and phi are interleaved
        n_kphi = 2 * Nnod * Ntim
        kphi_section = y[n_x : n_x + n_kphi]
        kphi_node_major = kphi_section.reshape((Nnod, 2 * Ntim))
        k = kphi_node_major[:, ::2]
        phi = kphi_node_major[:, 1::2]

        keff = y[n_x + n_kphi : n_x + n_kphi + Ntim]
        epsilon = y[-1]

        # Constants (scaled for 31 nodes)
        V = _build_V_vector_31()
        G = _build_G_matrix_31()
        alp_Pc_Dt = self.alp_Pc_Dt
        Fl_Nred = 1.0 / self.Nred

        # Trilinear elements (same structure as TWIRISM1)
        t_eoc = Ntim - 1
        xa = jnp.einsum("im,jm,j->ijm", x[:, 1, :], x[:, 0, :], k[:, t_eoc])
        xb = jnp.einsum("im,jm,j->ijm", x[:, 2, :], x[:, 1, :], k[:, t_eoc])
        xc = jnp.einsum("im,jm,j->ijm", x[:, 3, :], x[:, 2, :], k[:, t_eoc])

        # Vectorized constraints (same structure as TWIRISM1)

        # Sumi constraints: sum over i of V(i)*x(i,l,m) = 1 for each l,m
        sumi = jnp.einsum("i,ilm->lm", V, x) - 1.0
        sumi_flat = sumi.ravel()

        # Sumlm constraints: sum over l,m of x(i,l,m) = 1 for each i
        sumlm = jnp.sum(x, axis=(1, 2)) - 1.0

        # Norm constraints: sum over i of V(i)*k(i,t)*phi(i,t) = 1 for each t
        norm = jnp.einsum("i,it,it->t", V, k, phi) - 1.0

        # Kern constraints: keff(t)*phi(i,t) + sum_j G(i,j)*k(j,t)*phi(j,t) = 0
        kern = keff[None, :] * phi + jnp.einsum("ij,jt,jt->it", G, k, phi)
        kern_flat = kern.ravel()

        # Peak constraints: k(i,t)*phi(i,t) - epsilon/Fl_Nred <= 0
        peak = k * phi - epsilon * Fl_Nred
        peak_flat = peak.ravel()

        # Burn constraints: k(i,t+1) - k(i,t) + alp_Pc_Dt*k(i,t)*phi(i,t) = 0
        burn = k[:, 1:] - k[:, :-1] + alp_Pc_Dt * k[:, :-1] * phi[:, :-1]
        burn_flat = burn.ravel()

        # Plac constraints: k(i,1) - sum_m x(i,1,m)*Kfresh -
        # sum_{j,m} V(j)*[xa(i,j,m) + xb(i,j,m) + xc(i,j,m)] = 0
        trilinear_sum = jnp.einsum("j,ijm->i", V, xa + xb + xc)
        plac = k[:, 0] - jnp.sum(x[:, 0, :], axis=1) * self.Kfresh - trilinear_sum

        # Diagonal constraints: x(d1,l,m) = x(d2,l,m) for diagonal pairs
        # From SIF: D1,1=1 D2,1=18 D3,1=28, D1,2=10 D2,2=24 D3,2=31
        # Python uses 0-indexed, so: (0,17,27) and (9,23,30)
        dia1 = x[0, :, :] - x[17, :, :]  # First diagonal pair
        dia2 = x[9, :, :] - x[23, :, :]  # Second diagonal pair
        dia3 = x[27, :, :] - x[30, :, :]  # Third diagonal pair
        dia_flat = jnp.concatenate([dia1.ravel(), dia2.ravel(), dia3.ravel()])

        # Kefford constraints: keff(t+1) - keff(t) <= 0
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


def _build_V_vector_31() -> jnp.ndarray:
    """Build the V vector for 31 nodes from SIF values."""
    return jnp.array(
        [
            0.5,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.5,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.5,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.5,
            1.0,
            1.0,
            1.0,
            0.5,
            1.0,
            1.0,
            0.5,
        ]
    )


def _build_G_matrix_31() -> jnp.ndarray:
    """Build the G matrix for 31 nodes from SIF values."""
    G = jnp.zeros((31, 31))
    G = G.at[0, 0].set(-0.677)
    G = G.at[0, 1].set(-0.22)
    G = G.at[0, 2].set(-0.03)
    G = G.at[0, 9].set(-0.05)
    G = G.at[0, 10].set(-0.02)
    G = G.at[0, 17].set(-0.003)
    G = G.at[1, 0].set(-0.11)
    G = G.at[1, 1].set(-0.632)
    G = G.at[1, 2].set(-0.105)
    G = G.at[1, 3].set(-0.015)
    G = G.at[1, 9].set(-0.07)
    G = G.at[1, 10].set(-0.048)
    G = G.at[1, 11].set(-0.01)
    G = G.at[1, 17].set(-0.007)
    G = G.at[1, 18].set(-0.003)
    G = G.at[2, 0].set(-0.015)
    G = G.at[2, 1].set(-0.105)
    G = G.at[2, 2].set(-0.585)
    G = G.at[2, 3].set(-0.095)
    G = G.at[2, 4].set(-0.015)
    G = G.at[2, 9].set(-0.04)
    G = G.at[2, 10].set(-0.077)
    G = G.at[2, 11].set(-0.04)
    G = G.at[2, 12].set(-0.01)
    G = G.at[2, 17].set(-0.008)
    G = G.at[2, 18].set(-0.007)
    G = G.at[2, 19].set(-0.003)
    G = G.at[3, 1].set(-0.015)
    G = G.at[3, 2].set(-0.095)
    G = G.at[3, 3].set(-0.582)
    G = G.at[3, 4].set(-0.095)
    G = G.at[3, 5].set(-0.015)
    G = G.at[3, 9].set(-0.01)
    G = G.at[3, 10].set(-0.043)
    G = G.at[3, 11].set(-0.07)
    G = G.at[3, 12].set(-0.04)
    G = G.at[3, 13].set(-0.01)
    G = G.at[3, 17].set(-0.007)
    G = G.at[3, 18].set(-0.008)
    G = G.at[3, 19].set(-0.007)
    G = G.at[3, 20].set(-0.003)
    G = G.at[4, 2].set(-0.015)
    G = G.at[4, 3].set(-0.095)
    G = G.at[4, 4].set(-0.582)
    G = G.at[4, 5].set(-0.095)
    G = G.at[4, 6].set(-0.015)
    G = G.at[4, 10].set(-0.01)
    G = G.at[4, 11].set(-0.04)
    G = G.at[4, 12].set(-0.07)
    G = G.at[4, 13].set(-0.04)
    G = G.at[4, 14].set(-0.01)
    G = G.at[4, 17].set(-0.003)
    G = G.at[4, 18].set(-0.007)
    G = G.at[4, 19].set(-0.008)
    G = G.at[4, 20].set(-0.007)
    G = G.at[4, 21].set(-0.003)
    G = G.at[5, 3].set(-0.015)
    G = G.at[5, 4].set(-0.095)
    G = G.at[5, 5].set(-0.582)
    G = G.at[5, 6].set(-0.095)
    G = G.at[5, 7].set(-0.015)
    G = G.at[5, 11].set(-0.01)
    G = G.at[5, 12].set(-0.04)
    G = G.at[5, 13].set(-0.07)
    G = G.at[5, 14].set(-0.04)
    G = G.at[5, 15].set(-0.01)
    G = G.at[5, 18].set(-0.003)
    G = G.at[5, 19].set(-0.007)
    G = G.at[5, 20].set(-0.008)
    G = G.at[5, 21].set(-0.007)
    G = G.at[5, 22].set(-0.003)
    G = G.at[6, 4].set(-0.015)
    G = G.at[6, 5].set(-0.095)
    G = G.at[6, 6].set(-0.582)
    G = G.at[6, 7].set(-0.095)
    G = G.at[6, 8].set(-0.015)
    G = G.at[6, 12].set(-0.01)
    G = G.at[6, 13].set(-0.04)
    G = G.at[6, 14].set(-0.07)
    G = G.at[6, 15].set(-0.04)
    G = G.at[6, 16].set(-0.01)
    G = G.at[6, 19].set(-0.003)
    G = G.at[6, 20].set(-0.007)
    G = G.at[6, 21].set(-0.008)
    G = G.at[6, 22].set(-0.007)
    G = G.at[7, 5].set(-0.015)
    G = G.at[7, 6].set(-0.095)
    G = G.at[7, 7].set(-0.582)
    G = G.at[7, 8].set(-0.095)
    G = G.at[7, 13].set(-0.01)
    G = G.at[7, 14].set(-0.04)
    G = G.at[7, 15].set(-0.07)
    G = G.at[7, 16].set(-0.04)
    G = G.at[7, 20].set(-0.003)
    G = G.at[7, 21].set(-0.007)
    G = G.at[7, 22].set(-0.008)
    G = G.at[8, 6].set(-0.015)
    G = G.at[8, 7].set(-0.095)
    G = G.at[8, 8].set(-0.582)
    G = G.at[8, 14].set(-0.01)
    G = G.at[8, 15].set(-0.04)
    G = G.at[8, 16].set(-0.07)
    G = G.at[8, 21].set(-0.003)
    G = G.at[8, 22].set(-0.007)
    G = G.at[9, 0].set(-0.05)
    G = G.at[9, 1].set(-0.14)
    G = G.at[9, 2].set(-0.08)
    G = G.at[9, 3].set(-0.02)
    G = G.at[9, 9].set(-0.52)
    G = G.at[9, 10].set(-0.124)
    G = G.at[9, 11].set(-0.016)
    G = G.at[9, 17].set(-0.033)
    G = G.at[9, 18].set(-0.014)
    G = G.at[9, 23].set(-0.003)
    G = G.at[10, 0].set(-0.01)
    G = G.at[10, 1].set(-0.048)
    G = G.at[10, 2].set(-0.077)
    G = G.at[10, 3].set(-0.043)
    G = G.at[10, 4].set(-0.01)
    G = G.at[10, 9].set(-0.062)
    G = G.at[10, 10].set(-0.553)
    G = G.at[10, 11].set(-0.069)
    G = G.at[10, 12].set(-0.008)
    G = G.at[10, 17].set(-0.062)
    G = G.at[10, 18].set(-0.041)
    G = G.at[10, 19].set(-0.007)
    G = G.at[10, 23].set(-0.007)
    G = G.at[10, 24].set(-0.003)
    G = G.at[11, 1].set(-0.01)
    G = G.at[11, 2].set(-0.04)
    G = G.at[11, 3].set(-0.07)
    G = G.at[11, 4].set(-0.04)
    G = G.at[11, 5].set(-0.01)
    G = G.at[11, 9].set(-0.008)
    G = G.at[11, 10].set(-0.069)
    G = G.at[11, 11].set(-0.523)
    G = G.at[11, 12].set(-0.062)
    G = G.at[11, 13].set(-0.008)
    G = G.at[11, 17].set(-0.033)
    G = G.at[11, 18].set(-0.069)
    G = G.at[11, 19].set(-0.033)
    G = G.at[11, 20].set(-0.007)
    G = G.at[11, 23].set(-0.008)
    G = G.at[11, 24].set(-0.007)
    G = G.at[11, 25].set(-0.003)
    G = G.at[12, 2].set(-0.01)
    G = G.at[12, 3].set(-0.04)
    G = G.at[12, 4].set(-0.07)
    G = G.at[12, 5].set(-0.04)
    G = G.at[12, 6].set(-0.01)
    G = G.at[12, 10].set(-0.008)
    G = G.at[12, 11].set(-0.062)
    G = G.at[12, 12].set(-0.52)
    G = G.at[12, 13].set(-0.062)
    G = G.at[12, 14].set(-0.008)
    G = G.at[12, 17].set(-0.007)
    G = G.at[12, 18].set(-0.036)
    G = G.at[12, 19].set(-0.062)
    G = G.at[12, 20].set(-0.033)
    G = G.at[12, 21].set(-0.007)
    G = G.at[12, 23].set(-0.007)
    G = G.at[12, 24].set(-0.008)
    G = G.at[12, 25].set(-0.007)
    G = G.at[12, 26].set(-0.003)
    G = G.at[13, 3].set(-0.01)
    G = G.at[13, 4].set(-0.04)
    G = G.at[13, 5].set(-0.07)
    G = G.at[13, 6].set(-0.04)
    G = G.at[13, 7].set(-0.01)
    G = G.at[13, 11].set(-0.008)
    G = G.at[13, 12].set(-0.062)
    G = G.at[13, 13].set(-0.52)
    G = G.at[13, 14].set(-0.062)
    G = G.at[13, 15].set(-0.008)
    G = G.at[13, 18].set(-0.007)
    G = G.at[13, 19].set(-0.033)
    G = G.at[13, 20].set(-0.062)
    G = G.at[13, 21].set(-0.033)
    G = G.at[13, 22].set(-0.007)
    G = G.at[13, 23].set(-0.003)
    G = G.at[13, 24].set(-0.007)
    G = G.at[13, 25].set(-0.008)
    G = G.at[13, 26].set(-0.007)
    G = G.at[14, 4].set(-0.01)
    G = G.at[14, 5].set(-0.04)
    G = G.at[14, 6].set(-0.07)
    G = G.at[14, 7].set(-0.04)
    G = G.at[14, 8].set(-0.01)
    G = G.at[14, 12].set(-0.008)
    G = G.at[14, 13].set(-0.062)
    G = G.at[14, 14].set(-0.52)
    G = G.at[14, 15].set(-0.062)
    G = G.at[14, 16].set(-0.008)
    G = G.at[14, 19].set(-0.007)
    G = G.at[14, 20].set(-0.033)
    G = G.at[14, 21].set(-0.062)
    G = G.at[14, 22].set(-0.033)
    G = G.at[14, 24].set(-0.003)
    G = G.at[14, 25].set(-0.007)
    G = G.at[14, 26].set(-0.008)
    G = G.at[15, 5].set(-0.01)
    G = G.at[15, 6].set(-0.04)
    G = G.at[15, 7].set(-0.07)
    G = G.at[15, 8].set(-0.04)
    G = G.at[15, 13].set(-0.008)
    G = G.at[15, 14].set(-0.062)
    G = G.at[15, 15].set(-0.52)
    G = G.at[15, 16].set(-0.062)
    G = G.at[15, 20].set(-0.007)
    G = G.at[15, 21].set(-0.033)
    G = G.at[15, 22].set(-0.062)
    G = G.at[15, 25].set(-0.003)
    G = G.at[15, 26].set(-0.007)
    G = G.at[16, 6].set(-0.01)
    G = G.at[16, 7].set(-0.04)
    G = G.at[16, 8].set(-0.07)
    G = G.at[16, 14].set(-0.008)
    G = G.at[16, 15].set(-0.062)
    G = G.at[16, 16].set(-0.52)
    G = G.at[16, 21].set(-0.007)
    G = G.at[16, 22].set(-0.033)
    G = G.at[16, 26].set(-0.003)
    G = G.at[17, 0].set(-0.003)
    G = G.at[17, 1].set(-0.014)
    G = G.at[17, 2].set(-0.016)
    G = G.at[17, 3].set(-0.014)
    G = G.at[17, 4].set(-0.006)
    G = G.at[17, 9].set(-0.033)
    G = G.at[17, 10].set(-0.124)
    G = G.at[17, 11].set(-0.066)
    G = G.at[17, 12].set(-0.014)
    G = G.at[17, 17].set(-0.52)
    G = G.at[17, 18].set(-0.124)
    G = G.at[17, 19].set(-0.016)
    G = G.at[17, 23].set(-0.033)
    G = G.at[17, 24].set(-0.014)
    G = G.at[17, 27].set(-0.003)
    G = G.at[18, 1].set(-0.003)
    G = G.at[18, 2].set(-0.007)
    G = G.at[18, 3].set(-0.008)
    G = G.at[18, 4].set(-0.007)
    G = G.at[18, 5].set(-0.003)
    G = G.at[18, 9].set(-0.007)
    G = G.at[18, 10].set(-0.041)
    G = G.at[18, 11].set(-0.069)
    G = G.at[18, 12].set(-0.036)
    G = G.at[18, 13].set(-0.007)
    G = G.at[18, 17].set(-0.062)
    G = G.at[18, 18].set(-0.553)
    G = G.at[18, 19].set(-0.069)
    G = G.at[18, 20].set(-0.008)
    G = G.at[18, 23].set(-0.062)
    G = G.at[18, 24].set(-0.041)
    G = G.at[18, 25].set(-0.007)
    G = G.at[18, 27].set(-0.007)
    G = G.at[18, 28].set(-0.003)
    G = G.at[19, 2].set(-0.003)
    G = G.at[19, 3].set(-0.007)
    G = G.at[19, 4].set(-0.008)
    G = G.at[19, 5].set(-0.007)
    G = G.at[19, 6].set(-0.003)
    G = G.at[19, 10].set(-0.007)
    G = G.at[19, 11].set(-0.033)
    G = G.at[19, 12].set(-0.062)
    G = G.at[19, 13].set(-0.033)
    G = G.at[19, 14].set(-0.007)
    G = G.at[19, 17].set(-0.008)
    G = G.at[19, 18].set(-0.069)
    G = G.at[19, 19].set(-0.523)
    G = G.at[19, 20].set(-0.062)
    G = G.at[19, 21].set(-0.008)
    G = G.at[19, 23].set(-0.033)
    G = G.at[19, 24].set(-0.069)
    G = G.at[19, 25].set(-0.033)
    G = G.at[19, 26].set(-0.007)
    G = G.at[19, 27].set(-0.008)
    G = G.at[19, 28].set(-0.007)
    G = G.at[19, 29].set(-0.003)
    G = G.at[20, 3].set(-0.003)
    G = G.at[20, 4].set(-0.007)
    G = G.at[20, 5].set(-0.008)
    G = G.at[20, 6].set(-0.007)
    G = G.at[20, 7].set(-0.003)
    G = G.at[20, 11].set(-0.007)
    G = G.at[20, 12].set(-0.033)
    G = G.at[20, 13].set(-0.062)
    G = G.at[20, 14].set(-0.033)
    G = G.at[20, 15].set(-0.007)
    G = G.at[20, 18].set(-0.008)
    G = G.at[20, 19].set(-0.062)
    G = G.at[20, 20].set(-0.52)
    G = G.at[20, 21].set(-0.062)
    G = G.at[20, 22].set(-0.008)
    G = G.at[20, 23].set(-0.007)
    G = G.at[20, 24].set(-0.036)
    G = G.at[20, 25].set(-0.062)
    G = G.at[20, 26].set(-0.033)
    G = G.at[20, 27].set(-0.007)
    G = G.at[20, 28].set(-0.008)
    G = G.at[20, 29].set(-0.007)
    G = G.at[21, 4].set(-0.003)
    G = G.at[21, 5].set(-0.007)
    G = G.at[21, 6].set(-0.008)
    G = G.at[21, 7].set(-0.007)
    G = G.at[21, 8].set(-0.003)
    G = G.at[21, 12].set(-0.007)
    G = G.at[21, 13].set(-0.033)
    G = G.at[21, 14].set(-0.062)
    G = G.at[21, 15].set(-0.033)
    G = G.at[21, 16].set(-0.007)
    G = G.at[21, 19].set(-0.008)
    G = G.at[21, 20].set(-0.062)
    G = G.at[21, 21].set(-0.52)
    G = G.at[21, 22].set(-0.062)
    G = G.at[21, 24].set(-0.007)
    G = G.at[21, 25].set(-0.033)
    G = G.at[21, 26].set(-0.062)
    G = G.at[21, 27].set(-0.003)
    G = G.at[21, 28].set(-0.007)
    G = G.at[21, 29].set(-0.008)
    G = G.at[22, 5].set(-0.003)
    G = G.at[22, 6].set(-0.007)
    G = G.at[22, 7].set(-0.008)
    G = G.at[22, 8].set(-0.007)
    G = G.at[22, 13].set(-0.007)
    G = G.at[22, 14].set(-0.033)
    G = G.at[22, 15].set(-0.062)
    G = G.at[22, 16].set(-0.033)
    G = G.at[22, 20].set(-0.008)
    G = G.at[22, 21].set(-0.062)
    G = G.at[22, 22].set(-0.52)
    G = G.at[22, 25].set(-0.007)
    G = G.at[22, 26].set(-0.033)
    G = G.at[22, 28].set(-0.003)
    G = G.at[22, 29].set(-0.007)
    G = G.at[23, 9].set(-0.003)
    G = G.at[23, 10].set(-0.014)
    G = G.at[23, 11].set(-0.016)
    G = G.at[23, 12].set(-0.014)
    G = G.at[23, 13].set(-0.006)
    G = G.at[23, 17].set(-0.033)
    G = G.at[23, 18].set(-0.124)
    G = G.at[23, 19].set(-0.066)
    G = G.at[23, 20].set(-0.014)
    G = G.at[23, 23].set(-0.52)
    G = G.at[23, 24].set(-0.124)
    G = G.at[23, 25].set(-0.016)
    G = G.at[23, 27].set(-0.033)
    G = G.at[23, 28].set(-0.014)
    G = G.at[23, 30].set(-0.003)
    G = G.at[24, 10].set(-0.003)
    G = G.at[24, 11].set(-0.007)
    G = G.at[24, 12].set(-0.008)
    G = G.at[24, 13].set(-0.007)
    G = G.at[24, 14].set(-0.003)
    G = G.at[24, 17].set(-0.007)
    G = G.at[24, 18].set(-0.041)
    G = G.at[24, 19].set(-0.069)
    G = G.at[24, 20].set(-0.036)
    G = G.at[24, 21].set(-0.007)
    G = G.at[24, 23].set(-0.062)
    G = G.at[24, 24].set(-0.553)
    G = G.at[24, 25].set(-0.069)
    G = G.at[24, 26].set(-0.008)
    G = G.at[24, 27].set(-0.062)
    G = G.at[24, 28].set(-0.041)
    G = G.at[24, 29].set(-0.007)
    G = G.at[24, 30].set(-0.007)
    G = G.at[25, 11].set(-0.003)
    G = G.at[25, 12].set(-0.007)
    G = G.at[25, 13].set(-0.008)
    G = G.at[25, 14].set(-0.007)
    G = G.at[25, 15].set(-0.003)
    G = G.at[25, 18].set(-0.007)
    G = G.at[25, 19].set(-0.033)
    G = G.at[25, 20].set(-0.062)
    G = G.at[25, 21].set(-0.033)
    G = G.at[25, 22].set(-0.007)
    G = G.at[25, 23].set(-0.008)
    G = G.at[25, 24].set(-0.069)
    G = G.at[25, 25].set(-0.523)
    G = G.at[25, 26].set(-0.062)
    G = G.at[25, 27].set(-0.033)
    G = G.at[25, 28].set(-0.069)
    G = G.at[25, 29].set(-0.033)
    G = G.at[25, 30].set(-0.008)
    G = G.at[26, 12].set(-0.003)
    G = G.at[26, 13].set(-0.007)
    G = G.at[26, 14].set(-0.008)
    G = G.at[26, 15].set(-0.007)
    G = G.at[26, 16].set(-0.003)
    G = G.at[26, 19].set(-0.007)
    G = G.at[26, 20].set(-0.033)
    G = G.at[26, 21].set(-0.062)
    G = G.at[26, 22].set(-0.033)
    G = G.at[26, 24].set(-0.008)
    G = G.at[26, 25].set(-0.062)
    G = G.at[26, 26].set(-0.52)
    G = G.at[26, 27].set(-0.007)
    G = G.at[26, 28].set(-0.036)
    G = G.at[26, 29].set(-0.062)
    G = G.at[26, 30].set(-0.007)
    G = G.at[27, 17].set(-0.003)
    G = G.at[27, 18].set(-0.014)
    G = G.at[27, 19].set(-0.016)
    G = G.at[27, 20].set(-0.014)
    G = G.at[27, 21].set(-0.006)
    G = G.at[27, 23].set(-0.033)
    G = G.at[27, 24].set(-0.124)
    G = G.at[27, 25].set(-0.066)
    G = G.at[27, 26].set(-0.014)
    G = G.at[27, 27].set(-0.52)
    G = G.at[27, 28].set(-0.124)
    G = G.at[27, 29].set(-0.016)
    G = G.at[27, 30].set(-0.033)
    G = G.at[28, 18].set(-0.003)
    G = G.at[28, 19].set(-0.007)
    G = G.at[28, 20].set(-0.008)
    G = G.at[28, 21].set(-0.007)
    G = G.at[28, 22].set(-0.003)
    G = G.at[28, 23].set(-0.007)
    G = G.at[28, 24].set(-0.041)
    G = G.at[28, 25].set(-0.069)
    G = G.at[28, 26].set(-0.036)
    G = G.at[28, 27].set(-0.062)
    G = G.at[28, 28].set(-0.553)
    G = G.at[28, 29].set(-0.069)
    G = G.at[28, 30].set(-0.062)
    G = G.at[29, 19].set(-0.003)
    G = G.at[29, 20].set(-0.007)
    G = G.at[29, 21].set(-0.008)
    G = G.at[29, 22].set(-0.007)
    G = G.at[29, 24].set(-0.007)
    G = G.at[29, 25].set(-0.033)
    G = G.at[29, 26].set(-0.062)
    G = G.at[29, 27].set(-0.008)
    G = G.at[29, 28].set(-0.069)
    G = G.at[29, 29].set(-0.523)
    G = G.at[29, 30].set(-0.033)
    G = G.at[30, 23].set(-0.003)
    G = G.at[30, 24].set(-0.014)
    G = G.at[30, 25].set(-0.016)
    G = G.at[30, 26].set(-0.014)
    G = G.at[30, 27].set(-0.033)
    G = G.at[30, 28].set(-0.124)
    G = G.at[30, 29].set(-0.066)
    G = G.at[30, 30].set(-0.52)
    return G


def _get_twirimd1_initial_values() -> jnp.ndarray:
    """Hardcoded initial values from SIF file for TWIRIMD1."""
    return jnp.array(
        [
            0.0089,
            0.0089,
            0.0089,
            0.0089,
            0.0089,
            0.0089,
            0.0089,
            0.0609,
            0.0609,
            0.0609,
            0.0609,
            0.0609,
            0.0609,
            0.0609,
            0.0677,
            0.0677,
            0.0677,
            0.0677,
            0.0677,
            0.0677,
            0.0677,
            0.0054,
            0.0054,
            0.0054,
            0.0054,
            0.0054,
            0.0054,
            0.0054,
            0.0053,
            0.0053,
            0.0053,
            0.0053,
            0.0053,
            0.0053,
            0.0053,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.0072,
            0.0072,
            0.0072,
            0.0072,
            0.0072,
            0.0072,
            0.0072,
            0.0018,
            0.0018,
            0.0018,
            0.0018,
            0.0018,
            0.0018,
            0.0018,
            0.0089,
            0.0089,
            0.0089,
            0.0089,
            0.0089,
            0.0089,
            0.0089,
            0.0609,
            0.0609,
            0.0609,
            0.0609,
            0.0609,
            0.0609,
            0.0609,
            0.0677,
            0.0677,
            0.0677,
            0.0677,
            0.0677,
            0.0677,
            0.0677,
            0.0054,
            0.0054,
            0.0054,
            0.0054,
            0.0054,
            0.0054,
            0.0054,
            0.0089,
            0.0089,
            0.0089,
            0.0089,
            0.0089,
            0.0089,
            0.0089,
            0.0609,
            0.0609,
            0.0609,
            0.0609,
            0.0609,
            0.0609,
            0.0609,
            0.0677,
            0.0677,
            0.0677,
            0.0677,
            0.0677,
            0.0677,
            0.0677,
            0.0054,
            0.0054,
            0.0054,
            0.0054,
            0.0054,
            0.0054,
            0.0054,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.0079,
            0.0079,
            0.0079,
            0.0079,
            0.0079,
            0.0079,
            0.0079,
            0.0059,
            0.0059,
            0.0059,
            0.0059,
            0.0059,
            0.0059,
            0.0059,
            0.0004,
            0.0004,
            0.0004,
            0.0004,
            0.0004,
            0.0004,
            0.0004,
            0.0089,
            0.0089,
            0.0089,
            0.0089,
            0.0089,
            0.0089,
            0.0089,
            0.0609,
            0.0609,
            0.0609,
            0.0609,
            0.0609,
            0.0609,
            0.0609,
            0.0677,
            0.0677,
            0.0677,
            0.0677,
            0.0677,
            0.0677,
            0.0677,
            0.0054,
            0.0054,
            0.0054,
            0.0054,
            0.0054,
            0.0054,
            0.0054,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.0079,
            0.0079,
            0.0079,
            0.0079,
            0.0079,
            0.0079,
            0.0079,
            0.0059,
            0.0059,
            0.0059,
            0.0059,
            0.0059,
            0.0059,
            0.0059,
            0.0004,
            0.0004,
            0.0004,
            0.0004,
            0.0004,
            0.0004,
            0.0004,
            0.0632,
            0.0632,
            0.0632,
            0.0632,
            0.0632,
            0.0632,
            0.0632,
            0.0293,
            0.0293,
            0.0293,
            0.0293,
            0.0293,
            0.0293,
            0.0293,
            0.0361,
            0.0361,
            0.0361,
            0.0361,
            0.0361,
            0.0361,
            0.0361,
            0.0143,
            0.0143,
            0.0143,
            0.0143,
            0.0143,
            0.0143,
            0.0143,
            0.0028,
            0.0028,
            0.0028,
            0.0028,
            0.0028,
            0.0028,
            0.0028,
            0.0068,
            0.0068,
            0.0068,
            0.0068,
            0.0068,
            0.0068,
            0.0068,
            0.0047,
            0.0047,
            0.0047,
            0.0047,
            0.0047,
            0.0047,
            0.0047,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.0089,
            0.0089,
            0.0089,
            0.0089,
            0.0089,
            0.0089,
            0.0089,
            0.0609,
            0.0609,
            0.0609,
            0.0609,
            0.0609,
            0.0609,
            0.0609,
            0.0677,
            0.0677,
            0.0677,
            0.0677,
            0.0677,
            0.0677,
            0.0677,
            0.0054,
            0.0054,
            0.0054,
            0.0054,
            0.0054,
            0.0054,
            0.0054,
            0.0053,
            0.0053,
            0.0053,
            0.0053,
            0.0053,
            0.0053,
            0.0053,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.0072,
            0.0072,
            0.0072,
            0.0072,
            0.0072,
            0.0072,
            0.0072,
            0.0018,
            0.0018,
            0.0018,
            0.0018,
            0.0018,
            0.0018,
            0.0018,
            0.0046,
            0.0046,
            0.0046,
            0.0046,
            0.0046,
            0.0046,
            0.0046,
            0.0086,
            0.0086,
            0.0086,
            0.0086,
            0.0086,
            0.0086,
            0.0086,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.0011,
            0.0011,
            0.0011,
            0.0011,
            0.0011,
            0.0011,
            0.0011,
            0.0089,
            0.0089,
            0.0089,
            0.0089,
            0.0089,
            0.0089,
            0.0089,
            0.0609,
            0.0609,
            0.0609,
            0.0609,
            0.0609,
            0.0609,
            0.0609,
            0.0677,
            0.0677,
            0.0677,
            0.0677,
            0.0677,
            0.0677,
            0.0677,
            0.0054,
            0.0054,
            0.0054,
            0.0054,
            0.0054,
            0.0054,
            0.0054,
            0.0632,
            0.0632,
            0.0632,
            0.0632,
            0.0632,
            0.0632,
            0.0632,
            0.0293,
            0.0293,
            0.0293,
            0.0293,
            0.0293,
            0.0293,
            0.0293,
            0.0361,
            0.0361,
            0.0361,
            0.0361,
            0.0361,
            0.0361,
            0.0361,
            0.0143,
            0.0143,
            0.0143,
            0.0143,
            0.0143,
            0.0143,
            0.0143,
            0.0812,
            0.0812,
            0.0812,
            0.0812,
            0.0812,
            0.0812,
            0.0812,
            0.0474,
            0.0474,
            0.0474,
            0.0474,
            0.0474,
            0.0474,
            0.0474,
            0.0099,
            0.0099,
            0.0099,
            0.0099,
            0.0099,
            0.0099,
            0.0099,
            0.0044,
            0.0044,
            0.0044,
            0.0044,
            0.0044,
            0.0044,
            0.0044,
            0.0051,
            0.0051,
            0.0051,
            0.0051,
            0.0051,
            0.0051,
            0.0051,
            0.0091,
            0.0091,
            0.0091,
            0.0091,
            0.0091,
            0.0091,
            0.0091,
            0.0643,
            0.0643,
            0.0643,
            0.0643,
            0.0643,
            0.0643,
            0.0643,
            0.0643,
            0.0643,
            0.0643,
            0.0643,
            0.0643,
            0.0643,
            0.0643,
            0.0028,
            0.0028,
            0.0028,
            0.0028,
            0.0028,
            0.0028,
            0.0028,
            0.0068,
            0.0068,
            0.0068,
            0.0068,
            0.0068,
            0.0068,
            0.0068,
            0.0047,
            0.0047,
            0.0047,
            0.0047,
            0.0047,
            0.0047,
            0.0047,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.0089,
            0.0089,
            0.0089,
            0.0089,
            0.0089,
            0.0089,
            0.0089,
            0.0609,
            0.0609,
            0.0609,
            0.0609,
            0.0609,
            0.0609,
            0.0609,
            0.0677,
            0.0677,
            0.0677,
            0.0677,
            0.0677,
            0.0677,
            0.0677,
            0.0054,
            0.0054,
            0.0054,
            0.0054,
            0.0054,
            0.0054,
            0.0054,
            0.0812,
            0.0812,
            0.0812,
            0.0812,
            0.0812,
            0.0812,
            0.0812,
            0.0474,
            0.0474,
            0.0474,
            0.0474,
            0.0474,
            0.0474,
            0.0474,
            0.0099,
            0.0099,
            0.0099,
            0.0099,
            0.0099,
            0.0099,
            0.0099,
            0.0044,
            0.0044,
            0.0044,
            0.0044,
            0.0044,
            0.0044,
            0.0044,
            0.0812,
            0.0812,
            0.0812,
            0.0812,
            0.0812,
            0.0812,
            0.0812,
            0.0474,
            0.0474,
            0.0474,
            0.0474,
            0.0474,
            0.0474,
            0.0474,
            0.0099,
            0.0099,
            0.0099,
            0.0099,
            0.0099,
            0.0099,
            0.0099,
            0.0044,
            0.0044,
            0.0044,
            0.0044,
            0.0044,
            0.0044,
            0.0044,
            0.0812,
            0.0812,
            0.0812,
            0.0812,
            0.0812,
            0.0812,
            0.0812,
            0.0474,
            0.0474,
            0.0474,
            0.0474,
            0.0474,
            0.0474,
            0.0474,
            0.0099,
            0.0099,
            0.0099,
            0.0099,
            0.0099,
            0.0099,
            0.0099,
            0.0044,
            0.0044,
            0.0044,
            0.0044,
            0.0044,
            0.0044,
            0.0044,
            0.0046,
            0.0046,
            0.0046,
            0.0046,
            0.0046,
            0.0046,
            0.0046,
            0.0086,
            0.0086,
            0.0086,
            0.0086,
            0.0086,
            0.0086,
            0.0086,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.0011,
            0.0011,
            0.0011,
            0.0011,
            0.0011,
            0.0011,
            0.0011,
            0.0028,
            0.0028,
            0.0028,
            0.0028,
            0.0028,
            0.0028,
            0.0028,
            0.0068,
            0.0068,
            0.0068,
            0.0068,
            0.0068,
            0.0068,
            0.0068,
            0.0047,
            0.0047,
            0.0047,
            0.0047,
            0.0047,
            0.0047,
            0.0047,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.0089,
            0.0089,
            0.0089,
            0.0089,
            0.0089,
            0.0089,
            0.0089,
            0.0609,
            0.0609,
            0.0609,
            0.0609,
            0.0609,
            0.0609,
            0.0609,
            0.0677,
            0.0677,
            0.0677,
            0.0677,
            0.0677,
            0.0677,
            0.0677,
            0.0054,
            0.0054,
            0.0054,
            0.0054,
            0.0054,
            0.0054,
            0.0054,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.0079,
            0.0079,
            0.0079,
            0.0079,
            0.0079,
            0.0079,
            0.0079,
            0.0059,
            0.0059,
            0.0059,
            0.0059,
            0.0059,
            0.0059,
            0.0059,
            0.0004,
            0.0004,
            0.0004,
            0.0004,
            0.0004,
            0.0004,
            0.0004,
            0.0632,
            0.0632,
            0.0632,
            0.0632,
            0.0632,
            0.0632,
            0.0632,
            0.0293,
            0.0293,
            0.0293,
            0.0293,
            0.0293,
            0.0293,
            0.0293,
            0.0361,
            0.0361,
            0.0361,
            0.0361,
            0.0361,
            0.0361,
            0.0361,
            0.0143,
            0.0143,
            0.0143,
            0.0143,
            0.0143,
            0.0143,
            0.0143,
            0.0028,
            0.0028,
            0.0028,
            0.0028,
            0.0028,
            0.0028,
            0.0028,
            0.0068,
            0.0068,
            0.0068,
            0.0068,
            0.0068,
            0.0068,
            0.0068,
            0.0047,
            0.0047,
            0.0047,
            0.0047,
            0.0047,
            0.0047,
            0.0047,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.0051,
            0.0051,
            0.0051,
            0.0051,
            0.0051,
            0.0051,
            0.0051,
            0.0091,
            0.0091,
            0.0091,
            0.0091,
            0.0091,
            0.0091,
            0.0091,
            0.0643,
            0.0643,
            0.0643,
            0.0643,
            0.0643,
            0.0643,
            0.0643,
            0.0643,
            0.0643,
            0.0643,
            0.0643,
            0.0643,
            0.0643,
            0.0643,
            0.0028,
            0.0028,
            0.0028,
            0.0028,
            0.0028,
            0.0028,
            0.0028,
            0.0068,
            0.0068,
            0.0068,
            0.0068,
            0.0068,
            0.0068,
            0.0068,
            0.0047,
            0.0047,
            0.0047,
            0.0047,
            0.0047,
            0.0047,
            0.0047,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.0028,
            0.0028,
            0.0028,
            0.0028,
            0.0028,
            0.0028,
            0.0028,
            0.0068,
            0.0068,
            0.0068,
            0.0068,
            0.0068,
            0.0068,
            0.0068,
            0.0047,
            0.0047,
            0.0047,
            0.0047,
            0.0047,
            0.0047,
            0.0047,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.1286,
            0.0051,
            0.0051,
            0.0051,
            0.0051,
            0.0051,
            0.0051,
            0.0051,
            0.0091,
            0.0091,
            0.0091,
            0.0091,
            0.0091,
            0.0091,
            0.0091,
            0.0643,
            0.0643,
            0.0643,
            0.0643,
            0.0643,
            0.0643,
            0.0643,
            0.0643,
            0.0643,
            0.0643,
            0.0643,
            0.0643,
            0.0643,
            0.0643,
            1.0392,
            0.0283,
            1.0244,
            0.0302,
            1.0088,
            0.032,
            0.9925,
            0.0338,
            0.9756,
            0.0355,
            0.9582,
            0.037,
            1.0723,
            0.0308,
            1.0557,
            0.0325,
            1.0384,
            0.0341,
            1.0206,
            0.0356,
            1.0023,
            0.0371,
            0.9835,
            0.0384,
            1.0392,
            0.0339,
            1.0215,
            0.035,
            1.0034,
            0.0361,
            0.9852,
            0.0371,
            0.9667,
            0.0381,
            0.9482,
            0.039,
            1.0392,
            0.0407,
            1.0179,
            0.0411,
            0.9968,
            0.0415,
            0.976,
            0.0419,
            0.9554,
            0.0423,
            0.935,
            0.0427,
            1.2727,
            0.0548,
            1.2375,
            0.0545,
            1.2035,
            0.0542,
            1.1707,
            0.054,
            1.1388,
            0.0538,
            1.1079,
            0.0537,
            1.0392,
            0.0458,
            1.0152,
            0.0459,
            0.9917,
            0.0461,
            0.9687,
            0.0463,
            0.946,
            0.0466,
            0.9238,
            0.0469,
            1.2727,
            0.0454,
            1.2436,
            0.0462,
            1.2146,
            0.0469,
            1.1859,
            0.0477,
            1.1574,
            0.0485,
            1.1292,
            0.0493,
            1.1384,
            0.0269,
            1.123,
            0.028,
            1.1071,
            0.0292,
            1.0909,
            0.0304,
            1.0742,
            0.0317,
            1.057,
            0.033,
            0.9658,
            0.0107,
            0.9606,
            0.0114,
            0.955,
            0.0122,
            0.9492,
            0.013,
            0.943,
            0.0139,
            0.9364,
            0.0148,
            1.0392,
            0.0321,
            1.0224,
            0.0335,
            1.0052,
            0.0348,
            0.9876,
            0.0361,
            0.9696,
            0.0373,
            0.9514,
            0.0384,
            1.0723,
            0.0366,
            1.0526,
            0.0375,
            1.0327,
            0.0385,
            1.0126,
            0.0393,
            0.9926,
            0.0402,
            0.9725,
            0.0409,
            0.9996,
            0.0401,
            0.9794,
            0.0405,
            0.9594,
            0.0409,
            0.9396,
            0.0413,
            0.92,
            0.0417,
            0.9007,
            0.0421,
            1.0392,
            0.0448,
            1.0158,
            0.0449,
            0.9928,
            0.0451,
            0.9703,
            0.0452,
            0.9481,
            0.0454,
            0.9264,
            0.0457,
            1.1384,
            0.045,
            1.1126,
            0.0454,
            1.0872,
            0.0458,
            1.0621,
            0.0462,
            1.0374,
            0.0466,
            1.013,
            0.0471,
            1.1912,
            0.0366,
            1.1693,
            0.0374,
            1.1472,
            0.0383,
            1.1251,
            0.0393,
            1.1028,
            0.0403,
            1.0804,
            0.0413,
            0.9861,
            0.019,
            0.9766,
            0.0199,
            0.9668,
            0.0208,
            0.9566,
            0.0218,
            0.9461,
            0.0229,
            0.9352,
            0.024,
            0.9658,
            0.0079,
            0.962,
            0.0084,
            0.9579,
            0.009,
            0.9535,
            0.0096,
            0.9489,
            0.0103,
            0.944,
            0.0111,
            1.0392,
            0.0398,
            1.0184,
            0.0405,
            0.9976,
            0.0411,
            0.9769,
            0.0416,
            0.9564,
            0.0422,
            0.9361,
            0.0427,
            1.1912,
            0.0481,
            1.1624,
            0.0484,
            1.134,
            0.0486,
            1.1062,
            0.0489,
            1.0789,
            0.0492,
            1.0522,
            0.0494,
            1.1912,
            0.0486,
            1.162,
            0.0489,
            1.1334,
            0.0492,
            1.1053,
            0.0495,
            1.0777,
            0.0498,
            1.0507,
            0.0501,
            1.1912,
            0.0408,
            1.1667,
            0.0415,
            1.1423,
            0.0422,
            1.118,
            0.0429,
            1.0938,
            0.0437,
            1.0697,
            0.0444,
            0.9996,
            0.0238,
            0.9876,
            0.0247,
            0.9753,
            0.0255,
            0.9627,
            0.0265,
            0.9499,
            0.0274,
            0.9368,
            0.0285,
            0.9658,
            0.0116,
            0.9602,
            0.0122,
            0.9543,
            0.0129,
            0.9481,
            0.0136,
            0.9416,
            0.0143,
            0.9348,
            0.0152,
            1.0392,
            0.0443,
            1.016,
            0.0447,
            0.9932,
            0.045,
            0.9706,
            0.0454,
            0.9484,
            0.0457,
            0.9266,
            0.0461,
            1.2727,
            0.0462,
            1.2431,
            0.0468,
            1.2138,
            0.0473,
            1.1848,
            0.0479,
            1.1562,
            0.0485,
            1.128,
            0.0491,
            1.1384,
            0.0308,
            1.1208,
            0.0317,
            1.1029,
            0.0326,
            1.0847,
            0.0336,
            1.0664,
            0.0346,
            1.0478,
            0.0356,
            0.9658,
            0.0151,
            0.9584,
            0.0159,
            0.9508,
            0.0166,
            0.9428,
            0.0174,
            0.9345,
            0.0183,
            0.9259,
            0.0192,
            0.9861,
            0.0287,
            0.9718,
            0.0295,
            0.9573,
            0.0303,
            0.9427,
            0.0312,
            0.9279,
            0.0321,
            0.9129,
            0.033,
            0.9658,
            0.0175,
            0.9573,
            0.0183,
            0.9484,
            0.0191,
            0.9393,
            0.02,
            0.9298,
            0.0209,
            0.92,
            0.0219,
            0.9658,
            0.0085,
            0.9617,
            0.009,
            0.9573,
            0.0095,
            0.9527,
            0.0101,
            0.9478,
            0.0108,
            0.9427,
            0.0115,
            0.9861,
            0.0092,
            0.9815,
            0.0098,
            0.9766,
            0.0104,
            0.9715,
            0.011,
            0.9661,
            0.0117,
            0.9604,
            0.0125,
            1.0886,
            1.0657,
            1.0431,
            1.0209,
            0.999,
            0.9773,
            0.01,
        ]
    )
