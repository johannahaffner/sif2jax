import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class NONSCOMPNE(AbstractNonlinearEquations):
    """
    The extended Rosenbrock function (nonseparable version)
    with bounds such that the strict complementarity condition is
    violated for half of the bounds.

    Source:
    M. Lescrenier,
    "Towards the use of supercomputers for large scale nonlinear
    partially separable optimization",
    PhD Thesis, FUNDP (Namur, B), 1989.

    SIF input: Ph. Toint, May 1990.
    Bound-constrained nonlinear equations version: Nick Gould, June 2019.

    classification NOR2-AN-V-V
    """

    n: int = 5000
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def __init__(self, n: int = 5000):
        self.n = n

    def num_residuals(self) -> int:
        """Number of residuals equals number of variables."""
        return self.n

    def starting_point(self) -> Array:
        """Return the starting point for the problem."""
        return jnp.full(self.n, 3.0, dtype=jnp.float64)

    def residual(self, y: Array, args) -> Array:
        """Compute the residual vector."""
        # SQ(1) = X(1) - 1.0
        res1 = y[0] - 1.0

        # SQ(i) = (X(i) - X(i-1)^2) with SCALE 0.5 for i = 2 to N
        # Note: pycutest inverts the SCALE 0.5 to 2.0 for NLE problems
        # Element ELA(i) contributes -X(i-1)^2
        # X(i-1) for i=2..N corresponds to y[0..n-2]
        res_rest = 2.0 * (y[1:] - y[:-1] ** 2)

        # Combine results
        residuals = jnp.concatenate([jnp.array([res1]), res_rest])

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
        # This means x[0] = 1 and x[i] = x[i-1]^2 for all i
        # Which gives x[i] = 1 for all i
        return jnp.ones(self.n, dtype=jnp.float64)

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
        """Bounds for variables.

        Default bounds: -100 <= x[i] <= 100
        For odd i: x[i] >= 1.0
        """
        lower = jnp.full(self.n, -100.0, dtype=jnp.float64)
        upper = jnp.full(self.n, 100.0, dtype=jnp.float64)

        # For odd indices (1, 3, 5, ...), set lower bound to 1.0
        # In 0-based indexing, these are indices 0, 2, 4, ...
        odd_indices = jnp.arange(0, self.n, 2)
        lower = lower.at[odd_indices].set(1.0)

        return lower, upper
