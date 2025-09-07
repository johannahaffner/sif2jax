from abc import abstractmethod

import jax.numpy as jnp
from jax import config

from ..._problem import AbstractConstrainedMinimisation


config.update("jax_enable_x64", True)


class _AbstractLISWET(AbstractConstrainedMinimisation):
    """Base class for LISWET series - Li and Swetits k-convex approximation problems.

    A k-convex approximation problem posed as a convex quadratic problem.

    minimize 1/2 sum_{i=1}^{n+k} (x_i - c_i)^2

    subject to:
    sum_{i=0}^k C(k,i) (-1)^{k-i} x_{j+i} >= 0  for j=1,...,n

    where c_i = g(t_i) + small perturbation, t_i = (i-1)/(n+k-1)

    Source:
    W. Li and J. Swetits,
    "A Newton method for convex regression, data smoothing and
    quadratic programming with bounded constraints",
    SIAM J. Optimization 3 (3) pp 466-488, 1993.

    Classification: QLR2-AN-V-V
    """

    n: int = 2000
    k: int = 2

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @abstractmethod
    def _g_function(self, t):
        """The specific g(t) function for each LISWET variant."""
        return jnp.array([])

    def _compute_c(self):
        """Compute c_i values for the objective."""
        n_plus_k = self.n + self.k
        t = jnp.arange(n_plus_k) / (n_plus_k - 1)
        g = self._g_function(t)
        # Perturbation: 0.1 * sin(i)
        perturbation = 0.1 * jnp.sin(jnp.arange(1, n_plus_k + 1).astype(float))
        return g + perturbation

    def _compute_constraint_coeffs(self):
        """Compute (-1)^(k-i) * C(k,i) coefficients for constraints."""
        k = self.k
        # Compute binomial coefficients C(k,i)
        binom = jnp.zeros(k + 1)
        binom = binom.at[0].set(1.0)
        for i in range(1, k + 1):
            binom = binom.at[i].set(binom[i - 1] * (k - i + 1) / i)

        # Apply alternating signs
        coeffs = jnp.zeros(k + 1)
        for i in range(k + 1):
            sign = (-1.0) ** (k - i)
            coeffs = coeffs.at[i].set(sign * binom[i])
        return coeffs

    @property
    def y0(self):
        # Default starting point
        return jnp.zeros(self.n + self.k)

    @property
    def args(self):
        return None

    @property
    def bounds(self):
        return None

    def objective(self, y, args):
        del args
        c = self._compute_c()
        # f(x) = 1/2 sum (x_i - c_i)^2
        return 0.5 * jnp.sum((y - c) ** 2)

    def constraint(self, y):
        # Inequality constraints: sum_{i=0}^k C(k,i) (-1)^{k-i} x_{j+i} >= 0
        n = self.n
        k = self.k
        coeffs = self._compute_constraint_coeffs()

        # Vectorized computation using convolution
        # Each constraint j computes sum_{i=0}^k coeffs[i] * y[j+i]
        # This is equivalent to a 1D convolution

        # Pad y with zeros at the end to handle edge cases
        padded_y = jnp.pad(y, (0, k), mode="constant", constant_values=0)

        # Use convolution to compute all constraints at once
        # We need to reverse coeffs for convolution
        constraints = jnp.convolve(padded_y, coeffs[::-1], mode="valid")[:n]

        # Return (equality_constraints, inequality_constraints)
        # Inequality constraints g(x) >= 0
        return None, constraints

    @property
    def expected_result(self):
        # The expected solution is not provided in the SIF file
        return None

    @property
    def expected_objective_value(self):
        # The expected objective value is not provided in the SIF file
        return None
