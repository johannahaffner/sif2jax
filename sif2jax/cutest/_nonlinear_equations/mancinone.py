import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class MANCINONE(AbstractNonlinearEquations):
    """
    Mancino's function with variable dimension.
    This is a nonlinear equation variant of MANCINO

    Source:
    E. Spedicato,
    "Computational experience with quasi-Newton algorithms for
    minimization problems of moderate size",
    Report N-175, CISE, Milano, 1975.

    See also Buckley #51 (p. 72), Schittkowski #391 (for N = 30)

    SIF input: Ph. Toint, Dec 1989.
               correction by Ph. Shott, January, 1995.
               Nick Gould (nonlinear equation version), Jan 2019
               correction by S. Gratton & Ph. Toint, May 2024

    classification NOR2-AN-V-V
    """

    n: int = 100
    alpha: int = 5
    beta: float = 14.0
    gamma: int = 3
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def num_residuals(self) -> int:
        """Number of residuals."""
        return self.n

    def _compute_a(self) -> float:
        """Compute the A parameter."""
        n = self.n
        alpha_p1 = self.alpha + 1.0
        n_minus_1_sq = (n - 1) * (n - 1)
        beta_n = self.beta * n
        beta_n_sq = beta_n * beta_n
        f0 = alpha_p1 * alpha_p1 * n_minus_1_sq
        f1 = -f0
        f2 = beta_n_sq + f1
        f3 = 1.0 / f2
        f4 = beta_n * f3
        return -f4

    def starting_point(self) -> Array:
        """Return the starting point for the problem."""
        n = self.n
        alpha = self.alpha
        gamma = self.gamma
        A = self._compute_a()
        n_half = n / 2.0

        x0 = jnp.zeros(n, dtype=jnp.float64)

        for i in range(1, n + 1):
            h = 0.0

            # Sum over j < i
            for j in range(1, i):
                vij = jnp.sqrt(i / j)
                lij = jnp.log(vij)
                sij = jnp.sin(lij)
                cij = jnp.cos(lij)

                # Compute s^alpha and c^alpha
                sa = sij**alpha
                ca = cij**alpha

                sca = sa + ca
                hij = vij * sca
                h += hij

            # Sum over j > i
            for j in range(i + 1, n + 1):
                vij = jnp.sqrt(i / j)
                lij = jnp.log(vij)
                sij = jnp.sin(lij)
                cij = jnp.cos(lij)

                # Compute s^alpha and c^alpha
                sa = sij**alpha
                ca = cij**alpha

                sca = sa + ca
                hij = vij * sca
                h += hij

            # Compute ci = (i - n/2)^gamma
            i_minus_n_half = i - n_half
            ci = i_minus_n_half**gamma

            # Starting value
            xi0 = (h + ci) * A
            x0 = x0.at[i - 1].set(xi0)

        return x0

    def residual(self, y: Array, args) -> Array:
        """Compute the residual vector."""
        n = self.n
        alpha = self.alpha
        beta_n = self.beta * n
        gamma = self.gamma
        n_half = n / 2.0

        residuals = jnp.zeros(n, dtype=jnp.float64)

        for i in range(1, n + 1):
            # G(i) = beta*n*x(i) + sum of elements

            # Compute ci = (i - n/2)^gamma for the constant term
            i_minus_n_half = i - n_half
            ci = i_minus_n_half**gamma

            res = beta_n * y[i - 1] - ci

            # Add contributions from j < i
            for j in range(1, i):
                x_j = y[j - 1]
                vij = jnp.sqrt(x_j * x_j + i / j)
                lij = jnp.log(vij)
                sij = jnp.sin(lij)
                cij = jnp.cos(lij)

                # Compute s^alpha and c^alpha
                sa = sij**alpha
                ca = cij**alpha

                sumal = sa + ca
                res += vij * sumal

            # Add contributions from j > i
            for j in range(i + 1, n + 1):
                x_j = y[j - 1]
                vij = jnp.sqrt(x_j * x_j + i / j)
                lij = jnp.log(vij)
                sij = jnp.sin(lij)
                cij = jnp.cos(lij)

                # Compute s^alpha and c^alpha
                sa = sij**alpha
                ca = cij**alpha

                sumal = sa + ca
                res += vij * sumal

            residuals = residuals.at[i - 1].set(res)

        return residuals

    @property
    def y0(self) -> Array:
        """Initial guess for the optimization problem."""
        return self.starting_point()

    @property
    def args(self):
        """Additional arguments for the residual function."""
        return None

    def expected_result(self) -> Array:
        """Expected result of the optimization problem."""
        # Not explicitly given, but for nonlinear equations should satisfy F(x*) = 0
        return jnp.zeros(self.n, dtype=jnp.float64)

    def expected_objective_value(self) -> Array:
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)

    def constraint(self, y):
        """Returns the residuals as equality constraints."""
        return self.residual(y, self.args), None

    @property
    def bounds(self) -> tuple[Array, Array] | None:
        """Bounds for variables - free variables."""
        return None
