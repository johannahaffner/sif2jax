import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class NONDIANE(AbstractNonlinearEquations):
    """
    The Shanno nondiagonal extension of Rosenbrock function.
    This is a nonlinear equation variant of NONDIA

    Source:
    D. Shanno,
    " On Variable Metric Methods for Sparse Hessians II: the New
    Method",
    MIS Tech report 27, University of Arizona (Tucson, UK), 1978.

    See also Buckley #37 (p. 76) and Toint #15.

    SIF input: Ph. Toint, Dec 1989.
               Nick Gould (nonlinear equation version), Jan 2019

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
        return jnp.full(self.n, -1.0, dtype=jnp.float64)

    def residual(self, y: Array, args) -> Array:
        """Compute the residual vector."""
        gamma = 2.0

        # SQ(1) = X(1) - 1.0
        res1 = y[0] - 1.0

        # SQ(i) = 0.1 * (X(1) - X(i-1)^gamma) for i = 2 to N
        # Element ELA(i) contributes -X(i-1)^gamma
        # X(i-1) for i=2..N corresponds to y[0..n-2]
        res_rest = 0.1 * (y[0] - y[:-1] ** gamma)

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

    def expected_result(self) -> Array:
        """Expected result of the optimization problem."""
        # Solution should satisfy F(x*) = 0
        # This means x[0] = 1 and x[0] = x[i-1]^2 for all i
        # Which gives x[i] = 1 for all i
        return jnp.ones(self.n, dtype=jnp.float64)

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
