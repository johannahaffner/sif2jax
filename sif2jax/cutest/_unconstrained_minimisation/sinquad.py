import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class SINQUAD(AbstractUnconstrainedMinimisation):
    """A function with nontrivial groups and repetitious elements.

    TODO: Human review needed - Complex SIF group structure interpretation

    CHALLENGES IDENTIFIED:
    1. SIF version confusion:
       - Original SINQUAD.SIF marked as "incorrectly decoded"
       - SINQUAD2.SIF contains corrected formulation (May 2024)
       - pycutest likely serves SINQUAD2 under name "SINQUAD" (see SISSER pattern)

    2. Complex group structure:
       - G1: L4 group type with constant + GVAR
       - G(2) to G(N-1): L2 groups with E(i) - E(1) + S(i) elements
       - G(N): L2 group with E(N) - E(1) element (no sine)
       - Mixed L2/L4 group types with quadratic and trigonometric elements

    3. Test failures:
       - All objective, gradient, and Hessian tests fail
       - Large discrepancies suggest fundamental formulation error
       - Likely issue with group value computation or element interpretation

    SUSPECTED ISSUES:
    - Group USES interpretation: complex interaction between elements and group types
    - L4 vs L2 group type application to combined element values
    - Sine element SINE(V1, V2) with U1 = V1 - V2 interpretation
    - May require detailed SIF group computation rules or reference implementation

    WHAT NEEDS TO BE DONE:
    - Verify pycutest uses SINQUAD2 formulation under SINQUAD name
    - Debug group value computation step-by-step for small test cases
    - Compare element-by-element with pycutest for n=2,3 cases
    - Understand interaction between mixed L2/L4 group types and multiple elements

    Source:
    N. Gould, private communication.

    SIF input: N. Gould, Dec 1989.
    Modified: Formulation corrected May 2024 (SINQUAD2)

    Classification: OUR2-AY-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})
    n: int = 5000

    def __init__(self, n: int = 5000):
        """Initialize SINQUAD problem.

        Args:
            n: Number of variables (default 5000, must be >= 2)
        """
        if n < 2:
            raise ValueError("n must be >= 2")
        self.n = n

    def objective(self, y, args):
        """Compute objective function using SINQUAD2 (corrected) formulation.

        Structure:
        - G1: (1 + X1)⁴  (L4 group with constant 1.0)
        - G(i) for i=2 to N-1: (X(i)² - X1² + sin(X(i) - X(N)))²  (L2 groups)
        - G(N): (X(N)² - X1²)²  (L2 group, no sine term)
        """
        del args
        n = self.n

        # G1: L4 group with constant 1.0 and GVAR X1
        # Group value: 1.0 + X1, then raised to 4th power by L4
        g1 = (1.0 + y[0]) ** 4

        total = g1

        # G(i) for i=2 to N-1: L2 groups with E(i) - E(1) + S(i)
        # E(i) = X(i)², E(1) = X1², S(i) = sin(X(i) - X(N))
        if n > 2:
            # Vectorized computation for groups 2 to N-1
            i_vals = jnp.arange(1, n - 1)  # i=1 to n-2 (0-indexed for i=2 to n-1)
            x_i = y[i_vals]
            x_1 = y[0]
            x_n = y[n - 1]

            # Element contributions: E(i) - E(1) + S(i)
            e_diff = x_i**2 - x_1**2
            s_vals = jnp.sin(x_i - x_n)
            group_vals = e_diff + s_vals

            # L2 groups square the group value
            g_middle = jnp.sum(group_vals**2)
            total = total + g_middle

        # G(N): L2 group with E(N) - E(1) (no S element for i=N)
        # Group value: X(N)² - X1², then squared by L2
        g_n = (y[n - 1] ** 2 - y[0] ** 2) ** 2
        total = total + g_n

        return total

    @property
    def bounds(self):
        """All variables are unbounded."""
        lower = jnp.full(self.n, -jnp.inf)
        upper = jnp.full(self.n, jnp.inf)
        return lower, upper

    @property
    def y0(self):
        """Starting point: all variables = 0.1."""
        return jnp.full(self.n, 0.1)

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
        """Expected minimum value from SIF comments is -3.0."""
        return jnp.array(-3.0)
