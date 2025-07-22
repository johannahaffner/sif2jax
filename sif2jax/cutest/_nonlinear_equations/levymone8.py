import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class LEVYMONE8(AbstractNonlinearEquations):
    """A global optimization example due to Levy & Montalvo
    This problem is one of the parameterised set LEVYMONT5-LEVYMONT10

    Source:  problem 8 in

    A. V. Levy and A. Montalvo
    "The Tunneling Algorithm for the Global Minimization of Functions"
    SIAM J. Sci. Stat. Comp. 6(1) 1985 15:29
    https://doi.org/10.1137/0906002

    nonlinear equations version

    SIF input: Nick Gould, August 2021

    classification NOR2-AY-5-10
    """

    n: int = 5
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def num_residuals(self) -> int:
        """Number of residuals."""
        return 10  # 2*N where N=5

    def starting_point(self) -> Array:
        """Return the starting point for the problem."""
        # Using the LEVYMONTA starting point
        x = jnp.ones(self.n) * 8.0
        x = x.at[0].set(-8.0)
        return x

    def residual(self, y: Array, args) -> Array:
        """Compute the residual vector.

        Args:
            y: Array of shape (5,) containing [x1, x2, x3, x4, x5]
            args: Additional arguments (unused)

        Returns:
            Array of shape (10,) containing the residuals
        """
        x = y
        n = self.n

        # Problem parameters
        a = 1.0
        k = 10.0
        l = 1.0
        c = 0.0

        # Derived parameters
        pi = jnp.pi
        pi_over_n = pi / n
        k_pi_over_n = k * pi_over_n
        sqrt_k_pi_over_n = jnp.sqrt(k_pi_over_n)
        n_over_pi = n / pi
        a_minus_c = a - c

        # Initialize residuals array
        residuals = jnp.zeros(2 * n)

        # First set of equations: Q(i)
        for i in range(n):
            residuals = residuals.at[i].set(n_over_pi * (l * x[i] - a_minus_c))

        # Second set of equations: N(i)
        # N(1): sqrt(k*pi/n) * sin(pi*(l*x[0] + c))
        v1 = pi * (l * x[0] + c)
        residuals = residuals.at[n].set(sqrt_k_pi_over_n * jnp.sin(v1))

        # N(i) for i=2 to n: sqrt(k*pi/n) * (l*x[i-1] + c - a) * sin(pi*(l*x[i] + c))
        for i in range(1, n):
            u = l * x[i - 1] + c - a
            v = pi * (l * x[i] + c)
            residuals = residuals.at[n + i].set(sqrt_k_pi_over_n * u * jnp.sin(v))

        return residuals

    @property
    def y0(self) -> Array:
        """Initial guess for the optimization problem."""
        return self.starting_point()

    @property
    def args(self):
        """Additional arguments for the residual function."""
        return None

    def expected_result(self) -> Array:
        """Expected result of the optimization problem."""
        # The solution should make all residuals zero
        # From the structure, one solution is when sin terms are zero
        # This happens when pi*(l*x + c) = k*pi for integer k
        # So l*x + c = k for integer k
        # With l=1.0, c=0.0, we get x = k for integer k
        # The closest values in [-10, 10] are x = 1
        return jnp.ones(self.n)

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
        lower = jnp.full(self.n, -10.0)
        upper = jnp.full(self.n, 10.0)
        return lower, upper
