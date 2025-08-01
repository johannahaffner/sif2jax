import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class LEVYMONE(AbstractNonlinearEquations):
    """A global optimization example due to Levy & Montalvo
    This problem is one of the parameterised set LEVYMONT5-LEVYMONT10

    Source:  A. V. Levy and A. Montalvo
    "The Tunneling Algorithm for the Global Minimization of Functions"
    SIAM J. Sci. Stat. Comp. 6(1) 1985 15:29
    https://doi.org/10.1137/0906002

    nonlinear equations version

    SIF input: Nick Gould, August 2021

    classification NOR2-AY-V-V
    """

    n: int = 100
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def num_residuals(self) -> int:
        """Number of residuals."""
        return 2 * self.n

    def starting_point(self) -> Array:
        """Return the starting point for the problem."""
        # Using the LEVYMONTA starting point
        y0 = 8.0 * jnp.ones(self.n)
        if self.n >= 1:
            y0 = y0.at[0].set(-8.0)
        if self.n >= 2:
            y0 = y0.at[1].set(8.0)
        return y0

    def residual(self, y: Array, args) -> Array:
        """Compute the residual vector.

        Args:
            y: Array of shape (n,) containing the variables
            args: Additional arguments (unused)

        Returns:
            Array of shape (2*n,) containing the residuals
        """
        x = y
        n = self.n

        # Problem parameters (for larger N, L=1.0, C=0.0)
        a = 1.0
        k = 10.0
        l = 1.0
        c = 0.0

        # Derived parameters
        pi = jnp.pi
        pi_over_n = pi / n
        k_pi_over_n = k * pi_over_n
        sqrt_k_pi_over_n = jnp.sqrt(k_pi_over_n)
        a_minus_c = a - c

        # Initialize residuals array
        residuals = jnp.zeros(2 * n)

        # Q equations: pi/n * (l * x[i] - (a - c))
        q_values = pi_over_n * (l * x - a_minus_c)
        residuals = residuals.at[::2].set(q_values)  # Even indices: 0, 2, 4, ...

        # N(1): sqrt(k*pi/n) * sin(pi*(l*x[0] + c))
        v1 = pi * (l * x[0] + c)
        residuals = residuals.at[1].set(sqrt_k_pi_over_n * jnp.sin(v1))

        # N(i) for i >= 2: sqrt(k*pi/n) * (l*x[i-1] + c - a) * sin(pi*(l*x[i] + c))
        if n > 1:
            u_i = l * x[:-1] + c - a  # x[0] to x[n-2]
            v_i = pi * (l * x[1:] + c)  # x[1] to x[n-1]
            n_values = sqrt_k_pi_over_n * u_i * jnp.sin(v_i)
            residuals = residuals.at[3::2].set(
                n_values
            )  # Odd indices starting from 3: 3, 5, 7, ...

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
    def expected_result(self) -> Array | None:
        """Expected result of the optimization problem."""
        return None

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
        lower = -10.0 * jnp.ones(self.n)
        upper = 10.0 * jnp.ones(self.n)
        return lower, upper
