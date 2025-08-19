import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class NONSCOMP(AbstractBoundedMinimisation):
    """The extended Rosenbrock function (nonseparable version) with bounds.

    The bounds are set such that the strict complementarity condition is
    violated for half of the bounds.

    Source:
    M. Lescrenier,
    "Towards the use of supercomputers for large scale nonlinear
    partially separable optimization",
    PhD Thesis, FUNDP (Namur, B), 1989.

    SIF input: Ph. Toint, May 1990.

    classification SBR2-AN-V-0

    TODO: Human review needed - GROUP TYPE L2 not properly handled
    The SIF file uses GROUP TYPE L2 which affects gradient computation
    Current implementation gives gradient ~16x smaller than pycutest
    """

    n: int = 5000  # Default problem size
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def y0(self):
        """Initial guess - all 3.0."""
        return jnp.full(self.n, 3.0)

    @property
    def args(self):
        return None

    @property
    def bounds(self):
        """Variable bounds.

        Default bounds: [-100, 100]
        Odd-indexed variables have lower bound 1.0
        """
        lower = jnp.full(self.n, -100.0)
        upper = jnp.full(self.n, 100.0)

        # Set lower bounds for odd-indexed variables (0-indexed, so 0, 2, 4...)
        odd_indices = jnp.arange(0, self.n, 2)
        lower = lower.at[odd_indices].set(1.0)

        return lower, upper

    def objective(self, y, args):
        """Extended Rosenbrock objective - vectorized.

        The objective is:
        - (x[0] - 1)^2 + sum_{i=2}^n 0.25 * (x[i] - x[i-1]^2)^2
        """
        del args

        # First term: (x[0] - 1)^2
        obj = (y[0] - 1.0) ** 2

        # Remaining terms: 0.25 * (x[i] - x[i-1]^2)^2 for i=1..n-1
        # Note: SIF uses 1-based indexing, so x[i] in SIF is y[i-1] here
        x_prev = y[:-1]
        x_curr = y[1:]

        # Each term is 0.25 * (x[i] - x[i-1]^2)^2
        terms = 0.25 * (x_curr - x_prev**2) ** 2
        obj += jnp.sum(terms)

        return obj

    @property
    def expected_result(self):
        """Solution: all ones (except those constrained to be >= 1)."""
        return jnp.ones(self.n)

    @property
    def expected_objective_value(self):
        """Expected objective value at solution is 0.0."""
        return jnp.array(0.0)
