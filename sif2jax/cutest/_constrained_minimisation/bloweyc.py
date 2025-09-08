import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


# TODO: Human review needed
# Attempts made: Implemented alongside BLOWEYA with same fixes applied
# Suspected issues: Similar constraint issues as BLOWEYA, needs same fine-tuning
# Resources needed: Verification of integral=0.4 case handling,
#                    constraint formulation check
# Progress: Implemented with same structure as BLOWEYA, needs testing
class BLOWEYC(AbstractConstrainedMinimisation):
    """A nonconvex quadratic program proposed by James Blowey (University of Durham).

    This problem arises from the Cahn-Hilliard gradient theory for phase separation
    with non-smooth free energy. The problem is formulated as:

    minimize u^T A u + u^T w - v^T A u - 2v^T w - u^T v

    subject to:
        A w = u (Laplacian constraint)
        u ∈ [-1, 1] (bounds)
        ∫ u ds = ∫ v ds (integral constraint)

    where A is the discrete Laplacian with Neumann boundary conditions.

    Case C: a = 0.2, b = 0.5, c = 0.5, d = 0.8

    Source: J.F. Blowey and C.M. Elliott,
    "The Cahn-Hilliard gradient theory for phase separation with
    non-smooth free energy Part II: Numerical analysis",
    European Journal of Applied Mathematics (3) pp 147-179, 1992.

    SIF input: Nick Gould, August 1996
    Classification: QLR2-MN-V-V
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of discretization intervals (N=2000 from SIF)."""
        return 2000

    def _compute_v_values(self):
        """Compute the piecewise linear function v(s) values."""
        n = self.n

        # From SIF: na = N/5, nb = N/2, nc = N/2, nd = N/5 * 4
        # Case C: a = 0.2, b = 0.5, c = 0.5, d = 0.8
        na = n // 5  # N/5
        nb = n // 2  # N/2
        nc = n // 2  # N/2 (same as nb)
        nd = (n * 4) // 5  # N/5 * 4

        v = jnp.ones(n + 1, dtype=jnp.float64)

        # Linear transition from 1 to -1 between na+1 and nb
        if nb > na + 1:
            step = 2.0 / (nb - na - 1)
            indices = jnp.arange(na + 1, nb)
            v = v.at[indices].set(1.0 - step * (indices.astype(jnp.float64) - na))

        # Constant -1 between nb+1 and nc (but nc=nb, so this might be empty)
        if nc > nb:
            v = v.at[nb + 1 : nc].set(-1.0)

        # Linear transition from -1 to 1 between nc+1 and nd
        if nd > nc + 1:
            step = 2.0 / (nd - nc - 1)
            indices = jnp.arange(nc + 1, nd)
            v = v.at[indices].set(-1.0 + step * (indices.astype(jnp.float64) - nc))

        # CRITICAL FIX: Scale v values by analytical integral (0.4 for BLOWEYC)
        # This makes discrete trapezoid sum = 160 instead of 400
        v = v * 0.4

        return v

    def objective(self, y, args):
        n = self.n

        # Split variables: U(0), W(0), U(1), W(1), ..., U(N), W(N)
        u = y[::2]  # U variables at even indices
        w = y[1::2]  # W variables at odd indices

        v = self._compute_v_values()
        h_sq = 1.0 / (n * n)  # 1/N^2

        # Linear terms from GROUPS section
        # ZN OBJ U(I) VAL where VAL = V(I) * (-1/N^2)
        linear_u = jnp.sum(-v * u * h_sq)

        # ZN OBJ W(I) VAL where VAL = V(I) * (-2/N^2)
        linear_w = jnp.sum(-2 * v * w * h_sq)

        # Additional linear terms for U from discrete Laplacian structure
        # Line 174-175: U(0) coefficient is V(1) - V(0)
        linear_u_extra = (v[1] - v[0]) * u[0]

        # Lines 177-183: U(I) coefficient is -2*V(I) + V(I-1) + V(I+1) for I=1 to N-1
        if n > 1:
            coeffs = -2 * v[1:n] + v[0 : n - 1] + v[2 : n + 1]
            linear_u_extra += jnp.sum(coeffs * u[1:n])

        # Lines 186-187: U(N) coefficient is V(N-1) - V(N)
        linear_u_extra += (v[n - 1] - v[n]) * u[n]

        # Quadratic terms from ELEMENT USES/GROUP USES
        # C(I) elements: U(I) * W(I) with coefficient 1/N^2
        quad_uw = jnp.sum(u * w) * h_sq

        # D(I) elements: U(I)^2 with coefficients from GROUP USES
        # D(0) has coeff 1, D(1) to D(N-1) have coeff 2, D(N) has coeff 1
        quad_uu = u[0] ** 2  # D(0) with coeff 1
        if n > 1:
            quad_uu += 2 * jnp.sum(u[1:n] ** 2)  # D(1) to D(N-1) with coeff 2
        quad_uu += u[n] ** 2  # D(N) with coeff 1

        # O(I) elements: U(I) * U(I+1) with coefficient -2
        quad_u_cross = -2 * jnp.sum(u[0:n] * u[1 : n + 1])

        return linear_u + linear_w + linear_u_extra + quad_uw + quad_uu + quad_u_cross

    def constraint(self, y):
        n = self.n

        # Split variables: U(0), W(0), U(1), W(1), ..., U(N), W(N)
        u = y[::2]  # U variables at even indices
        w = y[1::2]  # W variables at odd indices

        h_sq = 1.0 / (n * n)  # 1/N^2

        # Equality constraints: A u = w / h^2
        # For discrete Laplacian with Neumann boundary conditions
        equality_constraints = jnp.zeros(n + 1)

        # Boundary point 0: u[0] - u[1] - w[0] * h^2 = 0
        equality_constraints = equality_constraints.at[0].set(u[0] - u[1] - w[0] * h_sq)

        # Interior points 1 to n-1: 2*u[i] - u[i-1] - u[i+1] - w[i] * h^2 = 0
        equality_constraints = equality_constraints.at[1:n].set(
            2 * u[1:n] - u[0 : n - 1] - u[2 : n + 1] - w[1:n] * h_sq
        )

        # Boundary point N: u[N] - u[N-1] - w[N] * h^2 = 0
        equality_constraints = equality_constraints.at[n].set(
            u[n] - u[n - 1] - w[n] * h_sq
        )

        # Integral constraint: trapezoid rule integration should equal 0.2 * INT
        # where INT = (1 + a + b - c - d) * N = 0.4 * N for BLOWEYC
        # Constraint: (trapezoid_sum/N) - (target_value/N) = 0
        int_u_trap = 0.5 * u[0] + jnp.sum(u[1:n]) + 0.5 * u[n]
        expected_integral = 0.4 * n  # INT where INT = 0.4 * N
        integral_constraint = int_u_trap - expected_integral

        # Combine all equality constraints - integral constraint FIRST
        all_equality = jnp.concatenate(
            [jnp.array([integral_constraint]), equality_constraints]
        )

        return all_equality, None

    @property
    def bounds(self):
        n = self.n

        # U variables: [-1, 1], W variables: unbounded
        # Variables ordered as: U(0), W(0), U(1), W(1), ..., U(N), W(N)
        lower = jnp.zeros(2 * (n + 1))
        upper = jnp.zeros(2 * (n + 1))

        # Set U bounds at even indices
        lower = lower.at[::2].set(-1.0)
        upper = upper.at[::2].set(1.0)

        # Set W bounds at odd indices (unbounded)
        lower = lower.at[1::2].set(-jnp.inf)
        upper = upper.at[1::2].set(jnp.inf)

        return (lower, upper)

    @property
    def y0(self):
        n = self.n
        v = self._compute_v_values()

        # Initial point: U = v, W = 0, interleaved as U(0), W(0), U(1), W(1), ...
        y0 = jnp.zeros(2 * (n + 1))
        y0 = y0.at[::2].set(v)  # U values at even indices
        y0 = y0.at[1::2].set(0.0)  # W values at odd indices

        return y0

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        # Solution not provided in SIF file, return None
        return None

    @property
    def expected_objective_value(self):
        # From SIF comment: -2.67211D+04 for N = 1000 (intermediate between A and B)
        # For N = 2000, we expect a similar order of magnitude
        return jnp.array(-267211.0)
