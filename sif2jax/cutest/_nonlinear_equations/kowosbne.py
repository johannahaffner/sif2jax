import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class KOWOSBNE(AbstractNonlinearEquations):
    """A problem arising in the analysis of kinetic data for an enzyme
    reaction, known under the name of Kowalik and Osborne problem
    in 4 variables. This is a nonlinear equation version
    of problem KOWOSB.

    Source:  Problem 15 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    SIF input: Ph. Toint, Dec 1989.
    Modification as a set of nonlinear equations: Nick Gould, Oct 2015.

    classification NOR2-MN-4-11
    """

    n: int = 4
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def num_residuals(self) -> int:
        """Number of residuals."""
        return 11

    def starting_point(self) -> Array:
        """Return the starting point for the problem."""
        return jnp.array([0.25, 0.39, 0.415, 0.39])

    def residual(self, y: Array, args) -> Array:
        """Compute the residual vector.

        Args:
            y: Array of shape (4,) containing [x1, x2, x3, x4]
            args: Additional arguments (unused)

        Returns:
            Array of shape (11,) containing the residuals
        """
        x1, x2, x3, x4 = y[0], y[1], y[2], y[3]

        # U values for each equation
        u = jnp.array(
            [4.0, 2.0, 1.0, 0.5, 0.25, 0.167, 0.125, 0.1, 0.0833, 0.0714, 0.0624]
        )

        # Target values (y-values)
        y_data = jnp.array(
            [
                0.1957,
                0.1947,
                0.1735,
                0.1600,
                0.0844,
                0.0627,
                0.0456,
                0.0342,
                0.0323,
                0.0235,
                0.0246,
            ]
        )

        # Compute the model: x1 * (u^2 + u*x2) / (u^2 + u*x3 + x4)
        u_sq = u * u
        b1 = u_sq + u * x2
        b2 = u_sq + u * x3 + x4
        model = x1 * b1 / b2

        # Residuals
        residuals = model - y_data

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
        # From the literature, the solution is approximately:
        return jnp.array([0.192807, 0.191282, 0.123057, 0.136062])

    def expected_objective_value(self) -> Array:
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)

    def constraint(self, y):
        """Returns the residuals as equality constraints."""
        return self.residual(y, self.args), None

    @property
    def bounds(self) -> tuple[Array, Array] | None:
        """No bounds for this problem."""
        return None
