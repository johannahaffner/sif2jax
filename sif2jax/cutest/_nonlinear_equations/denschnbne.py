import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class DENSCHNBNE(AbstractNonlinearEquations):
    """
    Source: an example problem (p. 201) in
    J.E. Dennis and R.B. Schnabel,
    "Numerical Methods for Unconstrained Optimization and Nonlinear
    Equations",
    Prentice-Hall, Englewood Cliffs, 1983.

    SIF input: Ph. Toint, Nov 1990.
    Nonlinear-equations version of DENSCHNB.SIF, Nick Gould, Jan 2020.

    classification NOR2-AN-2-3
    """

    n: int = 2  # Fixed dimension problem
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def residual(self, y: Array, args) -> Array:
        """Compute the residuals of the Dennis-Schnabel nonlinear equations"""
        x1, x2 = y

        # Group A: x1 - 2 = 0
        res1 = x1 - 2.0

        # Group B: (x1 - 2) * x2 = 0
        res2 = (x1 - 2.0) * x2

        # Group C: x2 + 1 = 0
        res3 = x2 + 1.0

        return jnp.array([res1, res2, res3])

    def y0(self) -> Array:
        """Initial guess for the optimization problem."""
        return jnp.array([1.0, 1.0], dtype=jnp.float64)

    def args(self):
        """Additional arguments for the residual function."""
        return None

    def expected_result(self) -> Array | None:
        """Expected result of the optimization problem."""
        # Solution is x = (2, -1)
        return jnp.array([2.0, -1.0], dtype=jnp.float64)

    def expected_objective_value(self) -> Array | None:
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)
