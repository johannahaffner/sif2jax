import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


# TODO: Human review needed
# Attempts made: Vectorized implementation, fixed ordering to match SIF
# Suspected issues: Complex constraint ordering or off-by-one indexing
# Resources needed: Detailed comparison with Fortran implementation
class MODBEALENE(AbstractNonlinearEquations):
    """
    A variation on Beale's problem in 2 variables
    This is a nonlinear equation variant of MODBEALE

    Source: An adaptation by Ph. Toint of Problem 5 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#89.
    SIF input: Ph. Toint, Mar 2003.
               Nick Gould (nonlinear equation version), Jan 2019

    classification NOR2-AN-V-V
    """

    n_half: int = 10000  # N/2 parameter
    alpha: float = 50.0
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def __init__(self, n_half: int = 10000):
        self.n_half = n_half

    @property
    def n(self) -> int:
        """Total number of variables is 2 * N/2."""
        return 2 * self.n_half

    def num_residuals(self) -> int:
        """Number of residuals = 3 * N/2 + (N/2 - 1)."""
        return 4 * self.n_half - 1

    def starting_point(self) -> Array:
        """Return the starting point for the problem."""
        return jnp.ones(self.n, dtype=jnp.float64)

    def _prodb_element(self, v1: Array, v2: Array, power: float) -> Array:
        """Product type element: v1 * (1 - v2^power)."""
        t = 1.0 - v2**power
        return v1 * t

    def residual(self, y: Array, args) -> Array:
        """Compute the residual vector."""
        n_half = self.n_half
        ralphinv = jnp.sqrt(1.0 / self.alpha)

        # Extract odd and even indexed elements
        x_odd = y[0::2]  # X(1), X(3), X(5), ... (0-based: y[0], y[2], y[4], ...)
        x_even = y[1::2]  # X(2), X(4), X(6), ... (0-based: y[1], y[3], y[5], ...)

        # Vectorized computation of BA, BB, BC groups
        # BA(i): X(2i-1) * (1 - X(2i)^1) - 1.5
        ba_residuals = x_odd * (1.0 - x_even) - 1.5

        # BB(i): X(2i-1) * (1 - X(2i)^2) - 2.25
        bb_residuals = x_odd * (1.0 - x_even**2) - 2.25

        # BC(i): X(2i-1) * (1 - X(2i)^3) - 2.625
        bc_residuals = x_odd * (1.0 - x_even**3) - 2.625

        # Vectorized computation of L groups for i = 1 to N/2-1
        # L(i) = ralphinv * (6.0 * X(2i) - X(2i+2))
        # In 0-based: ralphinv * (6.0 * y[2i-1] - y[2i+1])
        # Which is: ralphinv * (6.0 * x_even[i-1] - x_even[i])
        l_residuals = ralphinv * (6.0 * x_even[:-1] - x_even[1:])

        # Build residuals in the correct order
        # For i = 1 to N/2-1: BA(i), BB(i), BC(i), L(i)
        # For i = N/2: BA(N/2), BB(N/2), BC(N/2)

        # Create array for first n_half-1 groups (each with 4 residuals)
        # Stack BA, BB, BC, L for each i into shape (n_half-1, 4)
        first_groups = jnp.stack(
            [ba_residuals[:-1], bb_residuals[:-1], bc_residuals[:-1], l_residuals],
            axis=1,
        ).flatten()

        # Add last group (BA, BB, BC for i=n_half)
        last_group = jnp.array(
            [
                ba_residuals[n_half - 1],
                bb_residuals[n_half - 1],
                bc_residuals[n_half - 1],
            ]
        )

        return jnp.concatenate([first_groups, last_group])

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
