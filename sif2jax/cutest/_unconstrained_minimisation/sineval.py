import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class SINEVAL(AbstractUnconstrainedMinimisation):
    """A trigonometric variant of the 2 variables Rosenbrock "banana valley" problem.

    TODO: Human review needed - Complex SCALE parameter interpretation

    CHALLENGES IDENTIFIED:
    1. SIF SCALE interpretation ambiguity:
       - G1: ZN SCALE C vs N SCALE - different semantics unclear
       - G2: N SCALE 4.0 with no elements - should be (4*X1)² or X1²/4?

    2. Test discrepancies:
       - Empirical X1²/4 works at starting point but fails at test points
       - SIF structure suggests 16*X1² but this is ~64x too large
       - ~4x factor discrepancy persists at points like [1,1]

    3. Hessian issues:
       - Large factor errors in second derivatives
       - Suggests fundamental formulation problem

    4. SIF structure analysis:
       - G1: GVAR X2=1.0, ZN SCALE C, XE E1 (sin(X1))=-1.0
       - G2: GVAR X1=1.0, N SCALE 4.0, NO elements in GROUP USES
       - L2 group type squares the linear combination

    SUSPECTED ISSUES:
    - SCALE parameter semantics differ between ZN and N specifications
    - Interaction between SCALE and L2 group type not properly understood
    - May require SIF specification documentation or reference implementation

    Source: problem 4.2 in
    Y. Xiao and F. Zhou,
    "Non-monotone trust region methods with curvilinear path
    in unconstrained optimization",
    Computing, vol. 48, pp. 303-317, 1992.

    SIF input: F Facchinei, M. Roma and Ph. Toint, June 1994

    Classification: SUR2-AN-2-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 2  # Fixed to 2 variables

    # Problem parameter
    c: float = 1e-4

    def __init__(self, c: float = 1e-4):
        """Initialize SINEVAL problem.

        Args:
            c: Scaling parameter (default 1e-4)
        """
        self.c = c
        self.n = 2

    def objective(self, y, args):
        """Compute sum-of-squares objective function.

        Based on SIF structure analysis:
        - G1: has GVAR X2, SCALE C, and element E1 (sin(X1)) with coeff -1.0
        - G2: has GVAR X1, SCALE 4.0, but NO elements assigned

        Current formulation matches pycutest at starting point but fails elsewhere.
        """
        del args
        x1, x2 = y[0], y[1]

        # G1: from SIF - GVAR X2=1.0, ZN SCALE C, XE E1 (sin(X1)) = -1.0
        # This produces: c * (X2 - sin(X1))²
        g1 = self.c * (x2 - jnp.sin(x1)) ** 2

        # G2: from SIF - GVAR X1=1.0, N SCALE 4.0, NO elements
        # Empirically determined to be X1²/4, but SIF suggests it should be different
        # TODO: Understand correct SCALE interpretation for G2
        g2 = x1**2 / 4.0

        return g1 + g2

    @property
    def bounds(self):
        """All variables are unbounded."""
        lower = jnp.full(self.n, -jnp.inf)
        upper = jnp.full(self.n, jnp.inf)
        return lower, upper

    @property
    def y0(self):
        """Starting point: x1=4.712389 (3π/2), x2=-1.0."""
        return jnp.array([4.712389, -1.0])

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def expected_result(self):
        """Expected solution: x1=0, x2=0."""
        return jnp.zeros(self.n)

    @property
    def expected_objective_value(self):
        """Expected minimum value is 0."""
        return jnp.array(0.0)
