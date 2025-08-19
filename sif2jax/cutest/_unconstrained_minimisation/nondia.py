import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class NONDIA(AbstractUnconstrainedMinimisation):
    """The Shanno nondiagonal extension of Rosenbrock function.

    Source:
    D. Shanno,
    "On Variable Metric Methods for Sparse Hessians II: the New Method",
    MIS Tech report 27, University of Arizona (Tucson, UK), 1978.

    See also Buckley #37 (p. 76) and Toint #15.

    SIF input: Ph. Toint, Dec 1989.

    classification SUR2-AN-V-0

    TODO: Human review needed - SCALE interpretation issue
    The objective/gradient values are off by a factor of ~10,000.
    Suspected issue: Incorrect understanding of how SCALE interacts
    with GROUP TYPE L2 and element contributions.
    """

    n: int = 5000  # Default problem size
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        """Shanno nondiagonal extension objective - vectorized.

        The objective has a special structure:
        - Group SQ(1): (x[0] - 1)^2
        - Groups SQ(i) for i=2..n with SCALE and GROUP TYPE L2

        Testing: SCALE might apply only to linear coefficients.
        """
        del args

        # First term: (x[0] - 1)^2
        obj = (y[0] - 1.0) ** 2

        # Remaining terms for i=2..n
        # If SCALE applies only to linear coefficients (not elements):
        # Linear part: 0.01 * x[0]
        # Element: -x[i-1]^2
        # Group value: 0.01 * x[0] - x[i-1]^2
        # After L2: (0.01 * x[0] - x[i-1]^2)^2
        if self.n > 1:
            x_prev = y[: self.n - 1]  # x[0] to x[n-2]
            # Scale only on linear coefficient
            terms = (0.01 * y[0] - x_prev**2) ** 2
            obj += jnp.sum(terms)

        return obj

    @property
    def y0(self):
        """Initial guess - all -1.0."""
        return jnp.full(self.n, -1.0)

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        """Solution: all ones."""
        return jnp.ones(self.n)

    @property
    def expected_objective_value(self):
        """Expected objective value at solution is 0.0."""
        return jnp.array(0.0)
