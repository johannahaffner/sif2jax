import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


# TODO: Human review needed
# Attempts made: Vectorized the implementation
# Suspected issues: SIF file bug on line 64 - "X(I)" should be "X(N)"
# Resources needed: Fix in the SIF file or confirm intended behavior
class MOREBVNE(AbstractNonlinearEquations):
    """
    The Boundary Value problem.
    This is the nonlinear least-squares version without fixed variables.

    Source:  problem 28 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#17 (p. 75).

    SIF input: Ph. Toint, Dec 1989 and Nick Gould, Oct 1992.
    Modification as a set of nonlinear equations: Nick Gould, Oct 2015.

    classification NOR2-MN-V-V
    """

    n: int = 10
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def __init__(self, n: int = 10):
        self.n = n

    def num_residuals(self) -> int:
        """Number of residuals equals number of variables."""
        return self.n

    def starting_point(self) -> Array:
        """Return the starting point for the problem."""
        n = self.n
        h = 1.0 / (n + 1)

        # x[i] = t[i] * (t[i] - 1) where t[i] = i*h
        i_indices = jnp.arange(1, n + 1, dtype=jnp.float64)
        t = i_indices * h
        x0 = t * (t - 1.0)

        return x0

    def _wcube_element(self, v: Array, b: float) -> Array:
        """WCUBE element: (v + b)^3."""
        vplusb = v + b
        return vplusb**3

    def residual(self, y: Array, args) -> Array:
        """Compute the residual vector."""
        n = self.n
        h = 1.0 / (n + 1)
        h2 = h * h
        halfh2 = 0.5 * h2

        # Create arrays for vectorized computation
        i_indices = jnp.arange(1, n + 1, dtype=jnp.float64)
        ih_plus_1 = i_indices * h + 1.0

        # Compute all E(i) values: (X(i) + (i*h + 1))^3
        e_values = (y + ih_plus_1) ** 3

        # Initialize residuals
        residuals = jnp.zeros(n, dtype=y.dtype)

        # G(1): 2*X(1) - X(2) + halfh2 * E(1)
        residuals = residuals.at[0].set(2.0 * y[0] - y[1] + halfh2 * e_values[0])

        # G(i) for i = 2 to N-1: -X(i-1) + 2*X(i) - X(i+1) + halfh2 * E(i)
        residuals = residuals.at[1 : n - 1].set(
            -y[0 : n - 2] + 2.0 * y[1 : n - 1] - y[2:n] + halfh2 * e_values[1 : n - 1]
        )

        # G(N): -X(N-1) + 2*X(N) + halfh2 * E(N)
        residuals = residuals.at[n - 1].set(
            -y[n - 2] + 2.0 * y[n - 1] + halfh2 * e_values[n - 1]
        )

        return residuals

    @property
    def y0(self) -> Array:
        """Initial guess for the optimization problem."""
        return self.starting_point()

    @property
    def args(self):
        """Additional arguments for the residual function."""
        return None

    @property
    def expected_result(self) -> Array:
        """Expected result of the optimization problem."""
        # Not explicitly given, but for nonlinear equations should satisfy F(x*) = 0
        return jnp.zeros(self.n, dtype=jnp.float64)

    @property
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
