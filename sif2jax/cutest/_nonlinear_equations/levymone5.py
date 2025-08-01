import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class LEVYMONE5(AbstractNonlinearEquations):
    """A global optimization example due to Levy & Montalvo
    This problem is one of the parameterised set LEVYMONT5-LEVYMONT10

    Source:  problem 5 in

    A. V. Levy and A. Montalvo
    "The Tunneling Algorithm for the Global Minimization of Functions"
    SIAM J. Sci. Stat. Comp. 6(1) 1985 15:29
    https://doi.org/10.1137/0906002

    nonlinear equations version

    SIF input: Nick Gould, August 2021

    classification NOR2-AY-2-4
    """

    n: int = 2
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def num_residuals(self) -> int:
        """Number of residuals."""
        return 4  # 2*N where N=2

    def starting_point(self) -> Array:
        """Return the starting point for the problem."""
        # Using the LEVYMONTA starting point
        return jnp.array([-8.0, 8.0])

    def residual(self, y: Array, args) -> Array:
        """Compute the residual vector.

        Args:
            y: Array of shape (2,) containing [x1, x2]
            args: Additional arguments (unused)

        Returns:
            Array of shape (4,) containing the residuals
        """
        x = y
        n = self.n

        # Problem parameters
        a = 1.0
        k = 10.0
        l = 0.25
        c = 0.75

        # Derived parameters
        pi = jnp.pi
        pi_over_n = pi / n
        k_pi_over_n = k * pi_over_n
        sqrt_k_pi_over_n = jnp.sqrt(k_pi_over_n)
        # n_over_pi = n / pi  # Not used with inverted scale
        a_minus_c = a - c

        # Initialize residuals array
        residuals = jnp.zeros(2 * n)

        # Interleave Q and N equations: Q(1), N(1), Q(2), N(2), ...
        # Q(1): pi/n * (l * x[0] - (a - c))
        residuals = residuals.at[0].set(pi_over_n * (l * x[0] - a_minus_c))

        # N(1): sqrt(k*pi/n) * sin(pi*(l*x[0] + c))
        v1 = pi * (l * x[0] + c)
        residuals = residuals.at[1].set(sqrt_k_pi_over_n * jnp.sin(v1))

        # Q(2): pi/n * (l * x[1] - (a - c))
        residuals = residuals.at[2].set(pi_over_n * (l * x[1] - a_minus_c))

        # N(2): sqrt(k*pi/n) * (l*x[0] + c - a) * sin(pi*(l*x[1] + c))
        u2 = l * x[0] + c - a
        v2 = pi * (l * x[1] + c)
        residuals = residuals.at[3].set(sqrt_k_pi_over_n * u2 * jnp.sin(v2))

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
        # The solution should make all residuals zero
        # From the structure, one solution is when sin terms are zero
        # This happens when pi*(l*x + c) = k*pi for integer k
        # So l*x + c = k for integer k
        # With l=0.25, c=0.75, we get x = 4k - 3
        # The closest values in [-10, 10] are x = 1 (k=1) and x = -3 (k=0)
        return jnp.array([1.0, 1.0])

    @property
    def expected_objective_value(self) -> Array:
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)

    def constraint(self, y):
        """Returns the residuals as equality constraints."""
        return self.residual(y, self.args), None

    @property
    def bounds(self) -> tuple[Array, Array]:
        """Return the bounds for the variables."""
        lower = jnp.array([-10.0, -10.0])
        upper = jnp.array([10.0, 10.0])
        return lower, upper
