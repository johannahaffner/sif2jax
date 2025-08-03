import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class OSCIGRNE(AbstractNonlinearEquations):
    """
    The roots of the gradient of Yurii Nesterov's "oscillating path" problem
    Nonlinear equations version

    SIF input: Nick Gould, June 2011.

    classification NOR2-AN-V-V
    """

    n: int = 100000
    rho: float = 500.0  # Florian Jarre's value
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def __init__(self, n: int = 100000, rho: float = 500.0):
        self.n = n
        self.rho = rho

    def num_residuals(self) -> int:
        """Number of residuals equals number of variables."""
        return self.n

    def starting_point(self) -> Array:
        """Return the starting point for the problem."""
        x0 = jnp.ones(self.n, dtype=jnp.float64)
        x0 = x0.at[0].set(-2.0)
        return x0

    def residual(self, x: Array, args) -> Array:
        """Compute the residual vector."""
        residuals = jnp.zeros(self.n, dtype=jnp.float64)

        # G(1) = 0.5 * X(1) - 0.5 + B1
        # B1 element: P * (V - 2.0 * U^2 + 1.0) * U
        # with V = X(2), U = X(1), P = -4*RHO
        b1_p = -4.0 * self.rho
        b1_val = b1_p * (x[1] - 2.0 * x[0] ** 2 + 1.0) * x[0]
        residuals = residuals.at[0].set(0.5 * x[0] - 0.5 + b1_val)

        # For i = 2 to N-1
        # G(i) = A(i) + B(i)
        # A(i): P * (V - 2.0 * U^2 + 1.0) with V = X(i), U = X(i-1), P = 2*RHO
        # B(i): P * (V - 2.0 * U^2 + 1.0) * U with V = X(i+1), U = X(i), P = -4*RHO
        if self.n > 2:
            a_p = 2.0 * self.rho
            b_p = -4.0 * self.rho

            # Vectorized computation for middle elements
            i_mid = jnp.arange(1, self.n - 1)
            a_vals = a_p * (x[i_mid] - 2.0 * x[i_mid - 1] ** 2 + 1.0)
            b_vals = b_p * (x[i_mid + 1] - 2.0 * x[i_mid] ** 2 + 1.0) * x[i_mid]
            residuals = residuals.at[i_mid].set(a_vals + b_vals)

        # G(N) = A(N)
        # A(N): P * (V - 2.0 * U^2 + 1.0) with V = X(N), U = X(N-1), P = 2*RHO
        if self.n > 1:
            a_n_p = 2.0 * self.rho
            a_n_val = a_n_p * (x[-1] - 2.0 * x[-2] ** 2 + 1.0)
            residuals = residuals.at[-1].set(a_n_val)

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
        # Solution should satisfy F(x*) = 0
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
