import jax
import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


# TODO: Human review needed
# Attempts made: Vectorized implementation with scan and vmap
# Suspected issues: Complex collocation discretization may need different approach
# Additional resources needed: Verification of residual ordering and DC computation


class CHANNEL(AbstractNonlinearEquations):
    """
    Analyse the flow of a fluid during injection into a long vertical channel,
    assuming that the flow is modelled by the boundary-value problem

        u'''' = R (u'u'' - u u''') for t in [0,1]
        u(0) = 0, u(1) = 1, u'(0) = 0 = u'(1)

    where u is the potential function, u' is the tangential velocity of
    the field, and R is the Reynold's number

    This is problem 7 in the COPS (Version 2) collection of
    E. Dolan and J. More'
    see "Benchmarking Optimization Software with COPS"
    Argonne National Labs Technical Report ANL/MCS-246 (2000)

    SIF input: Nick Gould, November 2000

    classification NOR2-AN-V-V
    """

    # Problem parameters
    nd: int = 4  # Order of differential equation
    nh: int = 400  # Number of subintervals
    nc: int = 4  # Number of collocation points
    R: float = 10.0  # Reynolds number
    tf: float = 1.0  # End of interval

    # Roots of NC-th degree Legendre polynomial (for NC=4)
    rho1: float = 0.0694318442
    rho2: float = 0.3300094782
    rho3: float = 0.6699905218
    rho4: float = 0.9305681558

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def h(self) -> float:
        """Uniform interval length"""
        return self.tf / self.nh

    @property
    def rho(self) -> Array:
        """Legendre polynomial roots"""
        return jnp.array([self.rho1, self.rho2, self.rho3, self.rho4])

    def num_vars(self) -> int:
        """Total number of variables: V(nh,nd) + W(nh,nc) + DC(nh,nc,nd)"""
        return self.nh * self.nd + self.nh * self.nc + self.nh * self.nc * self.nd

    def unpack_vars(self, x: Array) -> tuple[Array, Array, Array]:
        """Unpack variables into V, W, and DC arrays"""
        idx = 0

        # V has shape (nh, nd)
        v_size = self.nh * self.nd
        v = x[idx : idx + v_size].reshape(self.nh, self.nd)
        idx += v_size

        # W has shape (nh, nc)
        w_size = self.nh * self.nc
        w = x[idx : idx + w_size].reshape(self.nh, self.nc)
        idx += w_size

        # DC has shape (nh, nc, nd)
        dc_size = self.nh * self.nc * self.nd
        dc = x[idx : idx + dc_size].reshape(self.nh, self.nc, self.nd)

        return v, w, dc

    def _factorial_array(self, max_n: int) -> Array:
        """Precompute factorials up to max_n"""
        facts = jnp.ones(max_n + 1)
        for i in range(1, max_n + 1):
            facts = facts.at[i].set(facts[i - 1] * i)
        return facts

    def _compute_dc_residuals(
        self, v: Array, w: Array, dc: Array, h: float, rho: Array
    ) -> Array:
        """Vectorized computation of DC residuals"""
        # Precompute factorials
        max_fact = 2 * self.nd
        factorials = self._factorial_array(max_fact)

        # Create mesh for i, j, s
        i_idx, j_idx, s_idx = jnp.meshgrid(
            jnp.arange(self.nh),
            jnp.arange(self.nc),
            jnp.arange(1, self.nd + 1),
            indexing="ij",
        )

        # Flatten indices
        i_flat = i_idx.flatten()
        j_flat = j_idx.flatten()
        s_flat = s_idx.flatten()

        # Compute rh for each (i,j) pair
        rh = rho[j_flat] * h

        # Initialize residuals
        residuals = -dc[i_flat, j_flat, s_flat - 1]

        # Vectorized polynomial part
        def poly_contribution(carry, k):
            res, i, j, s, rh_pow = carry
            mask = k >= s
            ind = k - s
            coef = jnp.where(mask, rh_pow / factorials[ind], 0.0)
            res = res + jnp.where(mask, v[i, k - 1] * coef, 0.0)
            rh_pow = rh_pow * rh
            return (res, i, j, s, rh_pow), None

        init_carry = (residuals, i_flat, j_flat, s_flat, jnp.ones_like(rh))
        (residuals, _, _, _, rh_pow), _ = jax.lax.scan(
            poly_contribution, init_carry, jnp.arange(1, self.nd + 1)
        )

        # Vectorized spline part
        hpower = h ** (self.nd - s_flat + 1)

        def spline_contribution(carry, k):
            res, i, j, rh_pow, hpower = carry
            ind = self.nd - s_flat + k
            coef = rh_pow * hpower / factorials[ind]
            res = res + w[i, k - 1] * coef
            rh_pow = rh_pow * rh
            return (res, i, j, rh_pow, hpower), None

        init_carry = (residuals, i_flat, j_flat, rh_pow, hpower)
        (residuals, _, _, _, _), _ = jax.lax.scan(
            spline_contribution, init_carry, jnp.arange(1, self.nc + 1)
        )

        return residuals.reshape(self.nh, self.nc, self.nd)

    def _compute_boundary_residuals(
        self, v: Array, w: Array, h: float
    ) -> tuple[Array, Array]:
        """Compute boundary condition residuals"""
        factorials = self._factorial_array(self.nd + self.nc)

        # Boundary condition at end: u(1) = 1
        res1 = -1.0

        # Vectorized computation for v terms
        k_arr = jnp.arange(1, self.nd + 1)
        h_powers = h ** (k_arr - 1)
        coefs = h_powers / factorials[k_arr - 1]
        res1 += jnp.sum(v[self.nh - 1, :] * coefs)

        # Vectorized computation for w terms
        k_arr = jnp.arange(1, self.nc + 1)
        h_powers = h ** (self.nd + k_arr - 1)
        coefs = h_powers / factorials[self.nd + k_arr - 1]
        res1 += jnp.sum(w[self.nh - 1, :] * coefs)

        # Boundary condition: u'(1) = 0
        res2 = 0.0

        # Vectorized computation for v terms (starting from k=2)
        k_arr = jnp.arange(2, self.nd + 1)
        h_powers = h ** (k_arr - 2)
        coefs = h_powers / factorials[k_arr - 2]
        res2 += jnp.sum(v[self.nh - 1, 1:] * coefs)

        # Vectorized computation for w terms
        k_arr = jnp.arange(1, self.nc + 1)
        h_powers = h ** (self.nd + k_arr - 2)
        coefs = h_powers / factorials[self.nd + k_arr - 2]
        res2 += jnp.sum(w[self.nh - 1, :] * coefs)

        return res1, res2

    def _compute_collocation_residuals(self, w: Array, dc: Array, rho: Array) -> Array:
        """Vectorized computation of collocation residuals"""
        factorials = self._factorial_array(self.nc)

        # Create rho powers for all j and k
        j_idx = jnp.arange(self.nc)
        k_idx = jnp.arange(self.nc)
        rho_expanded = rho[j_idx, None]
        rho_powers = rho_expanded ** k_idx[None, :]
        coefs = rho_powers / factorials[k_idx]

        # Compute linear part for all i and j
        linear_part = jnp.einsum("ik,jk->ij", w, coefs)

        # Compute nonlinear part
        nonlinear_part = self.R * (
            dc[:, :, 0] * dc[:, :, 3] - dc[:, :, 1] * dc[:, :, 2]
        )

        return (linear_part + nonlinear_part).flatten()

    def _compute_continuity_residuals(self, v: Array, w: Array, h: float) -> Array:
        """Vectorized computation of continuity residuals"""
        factorials = self._factorial_array(self.nd + self.nc)

        # Reshape for broadcasting
        i_idx, s_idx = jnp.meshgrid(
            jnp.arange(self.nh - 1), jnp.arange(1, self.nd + 1), indexing="ij"
        )
        i_flat = i_idx.flatten()
        s_flat = s_idx.flatten()

        # Base residual: -v[i+1, s]
        residuals = -v[i_flat + 1, s_flat - 1]

        # Vectorized v contributions
        def v_contribution(carry, k):
            res, i, s = carry
            mask = k >= s
            ind = k - s
            h_pow = jnp.where(mask, h**ind, 1.0)
            coef = jnp.where(mask, h_pow / factorials[ind], 0.0)
            res = res + jnp.where(mask, v[i, k - 1] * coef, 0.0)
            return (res, i, s), None

        init_carry = (residuals, i_flat, s_flat)
        (residuals, _, _), _ = jax.lax.scan(
            v_contribution, init_carry, jnp.arange(1, self.nd + 1)
        )

        # Vectorized w contributions
        k_arr = jnp.arange(1, self.nc + 1)
        h_powers = h ** (self.nd - s_flat[:, None] + k_arr[None, :])
        coefs = h_powers / factorials[self.nd - s_flat[:, None] + k_arr[None, :] - 1]
        w_contrib = jnp.sum(w[i_flat[:, None], k_arr[None, :] - 1] * coefs, axis=1)
        residuals = residuals + w_contrib

        return residuals

    def residual(self, y: Array, args) -> Array:
        """Compute residuals for the CHANNEL problem"""
        v, w, dc = self.unpack_vars(y)
        h = self.h
        rho = self.rho

        # Compute DC residuals (vectorized)
        dc_residuals = self._compute_dc_residuals(v, w, dc, h, rho)
        dc_residuals_flat = dc_residuals.flatten()

        # Compute boundary residuals
        bc1, bc2 = self._compute_boundary_residuals(v, w, h)

        # Compute collocation residuals (vectorized)
        coll_residuals = self._compute_collocation_residuals(w, dc, rho)

        # Compute continuity residuals (vectorized)
        cont_residuals = self._compute_continuity_residuals(v, w, h)

        # Concatenate all residuals
        return jnp.concatenate(
            [dc_residuals_flat, jnp.array([bc1, bc2]), coll_residuals, cont_residuals]
        )

    def objective(self, y: Array, args) -> Array:
        """Returns the constant objective value of -1.0."""
        return jnp.array(-1.0)

    def y0(self) -> Array:
        """Initial guess for the optimization problem."""
        h = self.h

        # Vectorized initialization of V
        i_arr = jnp.arange(self.nh)
        t = i_arr * h

        v = jnp.zeros((self.nh, self.nd), dtype=jnp.float64)
        v = v.at[:, 0].set(t * t * (3.0 - 2.0 * t))
        v = v.at[:, 1].set(6.0 * t * (1.0 - t))
        v = v.at[:, 2].set(6.0 * (1.0 - 2.0 * t))
        v = v.at[:, 3].set(-12.0)

        # W initialized to zeros
        w = jnp.zeros((self.nh, self.nc), dtype=jnp.float64)

        # Vectorized initialization of DC
        dc = jnp.zeros((self.nh, self.nc, self.nd), dtype=jnp.float64)
        rho = self.rho

        # This is complex to vectorize fully, but we can use vmap
        def compute_dc_for_ij(i, j):
            rh = rho[j] * h
            dc_ij = jnp.zeros(self.nd)

            for s in range(1, self.nd + 1):
                dcijs = 0.0
                prod = 1.0
                ind = 0

                for k in range(s, self.nd + 1):
                    dcijs += v[i, k - 1] * prod / self._factorial_array(ind)[ind]
                    prod *= rh
                    ind += 1

                hpower = h ** (self.nd - s + 1)
                for k in range(1, self.nc + 1):
                    dcijs += (
                        w[i, k - 1] * prod * hpower / self._factorial_array(ind)[ind]
                    )
                    prod *= rh
                    ind += 1

                dc_ij = dc_ij.at[s - 1].set(dcijs)

            return dc_ij

        # Use nested vmap for efficiency
        compute_dc_for_i = jax.vmap(compute_dc_for_ij, in_axes=(None, 0))
        compute_dc_all = jax.vmap(compute_dc_for_i, in_axes=(0, None))

        dc = compute_dc_all(jnp.arange(self.nh), jnp.arange(self.nc))

        # Pack all variables into a single array
        return jnp.concatenate([v.flatten(), w.flatten(), dc.flatten()])

    def args(self):
        """Additional arguments for the residual function."""
        return None

    def expected_result(self) -> Array | None:
        """Expected result of the optimization problem."""
        return None

    def expected_objective_value(self) -> Array | None:
        """Expected value of the objective at the solution."""
        return jnp.array(0.0)
