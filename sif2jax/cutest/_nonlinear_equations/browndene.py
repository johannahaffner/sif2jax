import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class BROWNDENE(AbstractNonlinearEquations):
    """
    Brown and Dennis problem in 4 variables.
    This function  is a nonlinear least squares with 20 groups.  Each
    group has 2 nonlinear elements. This is a nonlinear equation version
    of problem BROWNDEN.

    Source: Problem 16 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#30
    SIF input: Ph. Toint, Dec 1989.
    Modification as a set of nonlinear equations: Nick Gould, Oct 2015.

    classification NOR2-AN-4-20
    """

    n: int = 4
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def starting_point(self) -> Array:
        return jnp.array([25.0, 5.0, -5.0, -1.0], dtype=jnp.float64)

    def num_residuals(self) -> int:
        return 20

    def residual(self, y: Array, args) -> Array:
        """Compute the residuals of the Brown and Dennis problem"""
        x1, x2, x3, x4 = y

        residuals = []
        for i in range(1, 21):  # i = 1 to 20
            # Parameters for group i
            t_i = i / 5.0
            exp_t_i = jnp.exp(t_i)
            sin_t_i = jnp.sin(t_i)
            cos_t_i = jnp.cos(t_i)

            # Element A(i): (x1 + t_i * x2 - exp(t_i))^2
            a_i = x1 + t_i * x2 - exp_t_i

            # Element B(i): (x3 + sin(t_i) * x4 - cos(t_i))^2
            b_i = x3 + sin_t_i * x4 - cos_t_i

            # Group G(i) = A(i) + B(i)
            res_i = a_i + b_i
            residuals.append(res_i)

        return jnp.array(residuals)

    def y0(self) -> Array:
        """Initial guess for the optimization problem."""
        return self.starting_point()

    def args(self):
        """Additional arguments for the residual function."""
        return None

    def expected_result(self) -> Array | None:
        """Expected result of the optimization problem."""
        # The SIF file doesn't provide the solution vector
        return None

    def expected_objective_value(self) -> Array | None:
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)
