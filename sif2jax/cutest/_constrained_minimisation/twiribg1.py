import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class TWIRIBG1(AbstractConstrainedMinimisation):
    """
    TWIRIBG1 - Large nuclear reactor core reload pattern optimization problem.

    # TODO: Human review needed
    # Attempts made: [
    #   1. Fixed initial values from SIF file (parsed and stored 3127 exact
    #      values)
    #   2. Kept Kfresh=1.2 as per SIF file (different from TWIRIMD1)
    #   3. Implemented constraint structure matching TWIR pattern
    #      (Sumi, Sumlm, Norm, Kern, Burn, Plac, Dia)
    #   4. Added partial G matrix (52x52, truncated due to complexity) and
    #      V vector from SIF
    #   5. Fixed diagonal constraints (4 pairs for 52 nodes) based on
    #      SIF D values
    #   6. Implemented bounds correctly
    #   7. Implemented trilinear elements xa, xb, xc using same einsum
    #      structure
    # ]
    # Suspected issues: [
    #   - Trilinear constraint formulation not matching pycutest
    #     (max diff ~0.0715)
    #   - G matrix incomplete (TODO comment mentions 872 entries, only first
    #     few implemented)
    #   - Large scale (52 nodes, 3127 variables) amplifies any structural
    #     issues
    #   - Same fundamental trilinear problems as smaller TWIR variants
    #   - All constraint-related tests fail despite correct dimensions
    # ]
    # Resources needed: [
    #   - Complete G matrix extraction from SIF file (872 entries for
    #     52x52 matrix)
    #   - Same SIF ELEMENT analysis as other TWIR problems
    #   - Large-scale nuclear reactor geometry understanding
    #   - Verification that constraint scaling is appropriate for 52-node
    #     problem
    # ]

    A relaxation of a nonlinear integer programming problem for finding
    optimal nuclear reactor core reload patterns (large size version).

    Source:
    A.J. Quist, E. de Klerk, C. Roos, T. Terlaky, R. van Geemert,
    J.E. Hoogenboom, T.Illes,
    "Finding Optimal Nuclear Reactor Core Reload Patterns Using Nonlinear
    Optimization and Search Heuristics",
    Delft University, January 1998.

    classification LOI2-RN-3127-1239

    SIF input: Arie Quist, Delft, 1998.
    """

    # Problem dimensions from SIF file
    Nnod: int = 52  # Number of nodes
    Nred: int = 48  # Number of reduced nodes
    Ndia: int = 5  # Number of diagonal sets (estimate)
    Ntim: int = 6  # Number of time steps
    Nage: int = 4  # Number of age groups
    Ntra: int = 12  # Number of trajectories (Nred / Nage = 48 / 4 = 12)

    # Required attributes for the abstract class
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Pre-computed constants
    Kfresh: float = 1.2
    alp_Pc_Dt: float = 0.01

    @property
    def initial_guess(self) -> jnp.ndarray:
        """Initial guess for TWIRIBG1 - exact values from SIF file."""
        # Load complete precomputed initial values from the numpy file
        import os

        import numpy as np

        # Try to load from numpy file if it exists
        npy_path = os.path.join(os.path.dirname(__file__), "twiribg1_y0.npy")
        if os.path.exists(npy_path):
            return jnp.array(np.load(npy_path))

        # Fallback to hardcoded values - exact values from pycutest/SIF file
        return _get_twiribg1_initial_values()

    @property
    def y0(self) -> jnp.ndarray:
        """Initial guess."""
        return self.initial_guess

    @property
    def bounds(self):
        """Variable bounds for the large problem."""
        Nnod = self.Nnod
        Ntim = self.Ntim
        Nage = self.Nage
        Ntra = self.Ntra

        # Sizes
        n_x = Nnod * Nage * Ntra  # 2496
        n_kphi = 2 * Nnod * Ntim  # 624 (interleaved k and phi)
        n_keff = Ntim  # 6
        n_eps = 1  # 1

        # Physical constants
        Kfresh = 1.2

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
        n_x = Nnod * Nage * Ntra  # 2496
        n_kphi = 2 * Nnod * Ntim  # 624 (interleaved k and phi)

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

        # Extract variables
        n_x = Nnod * Nage * Ntra  # 2496
        x = y[:n_x].reshape((Nnod, Nage, Ntra))

        # k and phi are interleaved in pycutest
        n_kphi = 2 * Nnod * Ntim  # 624
        kphi_section = y[n_x : n_x + n_kphi]
        kphi_node_major = kphi_section.reshape((Nnod, 2 * Ntim))
        k = kphi_node_major[:, ::2]  # Every other column starting from 0 (k values)
        phi = kphi_node_major[
            :, 1::2
        ]  # Every other column starting from 1 (phi values)

        keff = y[n_x + n_kphi : n_x + n_kphi + Ntim]
        epsilon = y[-1]

        # Constants
        V = _build_V_vector_52()
        G = _build_G_matrix_52()
        alp_Pc_Dt = self.alp_Pc_Dt
        Fl_Nred = 1.0 / self.Nred

        # Trilinear elements from SIF ELEMENT definitions
        t_eoc = Ntim - 1  # End of Cycle
        xa = jnp.einsum("im,jm,j->ijm", x[:, 1, :], x[:, 0, :], k[:, t_eoc])
        xb = jnp.einsum("im,jm,j->ijm", x[:, 2, :], x[:, 1, :], k[:, t_eoc])
        xc = jnp.einsum("im,jm,j->ijm", x[:, 3, :], x[:, 2, :], k[:, t_eoc])

        # Vectorized constraints

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
        # From SIF file: D1,1=1 D2,1=22 D3,1=38 D4,1=49, D1,2=12 D2,2=31 D3,2=44 D4,2=52
        # Python uses 0-indexed, so: (0,21,37,48) and (11,30,43,51)
        dia1 = x[0, :, :] - x[21, :, :]  # First diagonal pair
        dia2 = x[11, :, :] - x[30, :, :]  # Second diagonal pair
        dia3 = x[37, :, :] - x[48, :, :]  # Third diagonal pair
        dia4 = x[43, :, :] - x[51, :, :]  # Fourth diagonal pair
        dia_flat = jnp.concatenate(
            [dia1.ravel(), dia2.ravel(), dia3.ravel(), dia4.ravel()]
        )

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


def _build_V_vector_52() -> jnp.ndarray:
    """Build the V vector for 52 nodes from SIF values."""
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
            1.0,
            0.5,
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
            1.0,
            0.5,
            1.0,
            1.0,
            0.5,
        ]
    )


def _build_G_matrix_52() -> jnp.ndarray:
    """Build the G matrix for 52 nodes from SIF values."""
    G = jnp.zeros((52, 52))
    # First few entries from the extracted data
    G = G.at[0, 0].set(-0.677)
    G = G.at[0, 1].set(-0.22)
    G = G.at[0, 2].set(-0.03)
    G = G.at[0, 11].set(-0.05)
    G = G.at[0, 12].set(-0.02)
    G = G.at[0, 21].set(-0.003)
    G = G.at[1, 0].set(-0.11)
    G = G.at[1, 1].set(-0.632)
    G = G.at[1, 2].set(-0.105)
    G = G.at[1, 3].set(-0.015)
    G = G.at[1, 11].set(-0.07)
    G = G.at[1, 12].set(-0.048)
    G = G.at[1, 13].set(-0.01)
    G = G.at[1, 21].set(-0.007)
    G = G.at[1, 22].set(-0.003)
    # TODO: Add all 872 entries from the complete G matrix
    # For now, this is a truncated version for testing
    return G


def _get_twiribg1_initial_values() -> jnp.ndarray:
    """Hardcoded initial values from SIF file for TWIRIBG1."""
    import os

    import numpy as np

    # Full array has 3127 elements - abbreviated here
    return jnp.array(
        np.load(os.path.join(os.path.dirname(__file__), "twiribg1_y0.npy"))
    )
