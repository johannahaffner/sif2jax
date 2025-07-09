from jax import numpy as jnp
from jaxtyping import Array, Float

from ..._problem import AbstractNonlinearEquations


class HATFLDBNE(AbstractNonlinearEquations):
    """A test problem from the OPTIMA user manual.

    Source:
    "The OPTIMA user manual (issue No.8, p. 12)",
    Numerical Optimization Centre, Hatfield Polytechnic (UK), 1989.

    SIF input: Ph. Toint, May 1990.
    Bound-constrained nonlinear equations version: Nick Gould, June 2019.

    classification NOR2-AN-4-4
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def residual(self, y, args) -> Float[Array, "4"]:
        """Residual function for the nonlinear equations."""
        x = y

        # G(1): X(1) = 1.0
        res1 = x[0] - 1.0

        # G(i) for i=2..4: X(i-1) - sqrt(X(i)) = 0
        residuals = [res1]
        for i in range(1, 4):  # i = 2, 3, 4 in 1-indexed notation
            residuals.append(x[i - 1] - jnp.sqrt(x[i]))

        return jnp.array(residuals)

    def y0(self) -> Float[Array, "4"]:
        """Initial guess for the optimization problem."""
        return jnp.array([0.1, 0.1, 0.1, 0.1])

    def args(self):
        """Additional arguments for the residual function."""
        return None

    def expected_result(self) -> Float[Array, "4"] | None:
        """Expected result of the optimization problem."""
        # The SIF file doesn't provide a solution
        return None

    def expected_objective_value(self) -> Float[Array, ""] | None:
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)

    def bounds(self) -> tuple[Float[Array, "4"], Float[Array, "4"]]:
        """Bounds on variables."""
        # Lower bounds: all 0.0000001
        lower = jnp.array([0.0000001, 0.0000001, 0.0000001, 0.0000001])
        # Upper bounds: only X2 has upper bound of 0.8
        upper = jnp.array([jnp.inf, 0.8, jnp.inf, jnp.inf])
        return lower, upper

    def num_constraints(self) -> tuple[int, int, int]:
        """Returns the number of constraints."""
        # 4 equality constraints (the residuals) + 5 finite bounds (4 lower + 1 upper)
        num_equalities = 4
        num_inequalities = 0
        num_bounds = 5  # 4 finite lower bounds + 1 finite upper bound
        return num_equalities, num_inequalities, num_bounds
