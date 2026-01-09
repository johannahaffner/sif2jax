from abc import abstractmethod

import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class _AbstractLISWET(AbstractConstrainedMinimisation):
    """Base class for LISWET series - Li and Swetits k-convex approximation problems.

    A k-convex approximation problem posed as a convex quadratic problem.

    minimize 1/2 sum_{i=1}^{n+k} (x_i - c_i)^2

    subject to:
    sum_{i=0}^k C(k,i) (-1)^{k-i} x_{j+i} >= 0  for j=1,...,n

    where c_i = g(t_i) + small perturbation, t_i = (i-1)/(n+k-1)

    Source:
    W. Li and J. Swetits,
    "A Newton method for convex regression, data smoothing and quadratic programming
    with bounded constraints", SIAM J. Optimization 3 (3) pp 466-488, 1993.

    Classification: QLR2-AN-V-V
    """

    # Pre-computed values as tuples (hashable for Equinox)
    _perturbation: tuple
    _t_values: tuple
    _constraint_coeffs: tuple

    n: int = 2000
    k: int = 2

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def __init__(self, n: int = 2000, k: int = 2):
        """Initialize with pre-computed values for performance."""
        self.n = n
        self.k = k
        self.y0_iD = 0
        self.provided_y0s = frozenset({0})

        # Pre-compute perturbation: 0.1 * sin(i)
        n_plus_k = n + k
        perturbation = 0.1 * jnp.sin(jnp.arange(1, n_plus_k + 1).astype(float))
        self._perturbation = tuple(float(x) for x in perturbation)

        # Pre-compute t values for g(t)
        t_values = jnp.arange(n_plus_k) / (n_plus_k - 1)
        self._t_values = tuple(float(x) for x in t_values)

        # Pre-compute constraint coefficients
        # Vectorized computation of binomial coefficients C(k,i)
        # C(k,i) = C(k,i-1) * (k-i+1) / i
        i = jnp.arange(1.0, k + 1.0)
        ratios = (k - i + 1.0) / i
        binom = jnp.concatenate([jnp.array([1.0]), jnp.cumprod(ratios)])

        # Vectorized computation of alternating signs: (-1)^(k-i)
        i = jnp.arange(k + 1.0)
        signs = (-1.0) ** (k - i)
        constraint_coeffs = signs * binom
        self._constraint_coeffs = tuple(float(x) for x in constraint_coeffs)

    @abstractmethod
    def _g_function(self, t) -> jnp.ndarray:
        """The specific g(t) function for each LISWET variant."""
        ...

    def _compute_c(self, dtype):
        """Compute c_i values for the objective using cached perturbation."""
        t_values = jnp.asarray(self._t_values, dtype=dtype)
        perturbation = jnp.asarray(self._perturbation, dtype=dtype)
        g = self._g_function(t_values)
        return g + perturbation

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
        c = self._compute_c(y.dtype)
        # f(x) = 1/2 sum (x_i - c_i)^2
        return 0.5 * jnp.sum((y - c) ** 2)

    def constraint(self, y):
        # Inequality constraints: sum_{i=0}^k C(k,i) (-1)^{k-i} x_{j+i} >= 0
        n = self.n
        k = self.k

        # Convert cached coefficients to array with appropriate dtype
        constraint_coeffs = jnp.asarray(self._constraint_coeffs, dtype=y.dtype)

        # Vectorized computation using convolution with cached coefficients
        # Each constraint j computes sum_{i=0}^k coeffs[i] * y[j+i]
        # This is equivalent to a 1D convolution

        # Pad y with zeros at the end to handle edge cases
        padded_y = jnp.pad(y, (0, k), mode="constant", constant_values=0)

        # Use convolution to compute all constraints at once
        # We need to reverse coeffs for convolution
        constraints = jnp.convolve(padded_y, constraint_coeffs[::-1], mode="valid")[:n]

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
