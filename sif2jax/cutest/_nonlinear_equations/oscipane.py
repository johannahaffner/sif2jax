import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class OSCIPANE(AbstractNonlinearEquations):
    """
    An "oscillating path" problem due to Yuri Nesterov
    Nonlinear equations version

    SIF input: Nick Gould, Dec 2006.

    classification NOR2-AN-V-V
    """

    n: int = 10
    rho: float = 500.0  # Florian Jarre's value
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def __init__(self, n: int = 10, rho: float = 500.0):
        self.n = n
        self.rho = rho

    def num_residuals(self) -> int:
        """Number of residuals equals number of variables."""
        return self.n

    def starting_point(self) -> Array:
        """Return the starting point for the problem."""
        x0 = jnp.ones(self.n, dtype=jnp.float64)
        x0 = x0.at[0].set(-1.0)
        return x0

    def residual(self, x: Array, args) -> Array:
        """Compute the residual vector."""
        residuals = jnp.zeros(self.n, dtype=jnp.float64)

        # Q1 = 0.5 * X(1) - 0.5
        residuals = residuals.at[0].set(0.5 * x[0] - 0.5)

        # For i = 2 to N
        # Q(i) = (1/rho) * (X(i) - P(i))
        # where P(i) is the CHEB element: 2.0 * TAU^2 - 1.0 with TAU = X(i-1)
        if self.n > 1:
            # Chebyshev element for indices 2 to N
            tau = x[:-1]  # X(i-1) for i=2..N
            cheb_vals = 2.0 * tau**2 - 1.0

            # Q(i) = (1/rho) * (X(i) - cheb_vals)
            residuals = residuals.at[1:].set((1.0 / self.rho) * (x[1:] - cheb_vals))

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
        # Solution should satisfy F(x*) = 0
        return jnp.zeros(self.n, dtype=jnp.float64)

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
