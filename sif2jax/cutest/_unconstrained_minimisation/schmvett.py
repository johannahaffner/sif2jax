import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class SCHMVETT(AbstractUnconstrainedMinimisation):
    """Schmidt and Vetters problem.

    This problem has N-2 groups, each with 3 nonlinear elements:
    - SCH1: fraction element -1/(1+(v1-v2)²)
    - SCH2: trigonometric element -sin(π*v1 + v2)/2
    - SCH3: exponential element -exp(-((v1+v3)/v2 - 2)²)

    Source:
    J.W. Schmidt and K. Vetters,
    "Ableitungsfreie Verfahren für Nichtlineare Optimierungsprobleme",
    Numerische Mathematik 15:263-282, 1970.

    See also Toint#35 and Buckley#14 (p90)

    SIF input: Ph. Toint, Dec 1989.

    Classification: OUR2-AY-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})
    n: int = 5000

    def __init__(self, n: int = 5000):
        """Initialize SCHMVETT problem.

        Args:
            n: Number of variables (default 5000, must be >= 3)
        """
        if n < 3:
            raise ValueError("n must be >= 3")
        self.n = n

    def objective(self, y, args):
        """Compute objective function.

        Sum of N-2 groups, each containing three element types.
        """
        del args
        n = self.n

        # For i = 1 to n-2, each group G(i) contains:
        # A(i): SCH1 with X(i), X(i+1)
        # B(i): SCH2 with X(i+1), X(i+2)
        # C(i): SCH3 with X(i), X(i+1), X(i+2)

        # Vectorized implementation for all groups
        i_vals = jnp.arange(n - 2)  # i = 0 to n-3 (0-indexed)

        # SCH1 elements: -1/(1+(X(i)-X(i+1))²)
        x_i = y[i_vals]  # X(i)
        x_i1 = y[i_vals + 1]  # X(i+1)
        x_i2 = y[i_vals + 2]  # X(i+2)

        # SCH1: fraction element
        diff = x_i - x_i1
        sch1 = -1.0 / (1.0 + diff * diff)

        # SCH2: trigonometric element -sin((π*X(i+1) + X(i+2))/2)
        # Note: SIF shows U = V1*3.14159265 + V2*1.0 = π*X(i+1) + X(i+2)
        # Then F = -sin(U/2) = -sin((π*X(i+1) + X(i+2))/2)
        u = jnp.pi * x_i1 + x_i2
        sch2 = -jnp.sin(u / 2.0)

        # SCH3: exponential element -exp(-((X(i)+X(i+2))/X(i+1) - 2)²)
        # Handle potential division by zero more carefully
        # When x_i1 is very close to zero, use a different approach
        x_i1_safe = jnp.where(jnp.abs(x_i1) < 1e-8, 1e-8, x_i1)
        ratio = (x_i + x_i2) / x_i1_safe
        diff_sq = (ratio - 2.0) ** 2
        sch3 = -jnp.exp(-diff_sq)

        # Sum all elements for each group
        group_sums = sch1 + sch2 + sch3

        return jnp.sum(group_sums)

    @property
    def bounds(self):
        """All variables are unbounded."""
        lower = jnp.full(self.n, -jnp.inf)
        upper = jnp.full(self.n, jnp.inf)
        return lower, upper

    @property
    def y0(self):
        """Starting point: all variables = 0.5."""
        return jnp.full(self.n, 0.5)

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def expected_result(self):
        """Expected solution not provided in SIF."""
        return None

    @property
    def expected_objective_value(self):
        """Expected minimum values from SIF comments."""
        # Known solutions from SIF file comments
        known_values = {3: -3.0, 10: -24.0, 100: -294.0, 500: -1494.0, 1000: -2994.0}

        if self.n in known_values:
            return jnp.array(known_values[self.n])
        else:
            # For larger problems, no known solution
            return None
