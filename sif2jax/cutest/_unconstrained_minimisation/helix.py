import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# TODO: human review required
class HELIX(AbstractUnconstrainedMinimisation):
    """The helix problem.

    This test function involves the computation of the arctangent function
    (theta = arctan(y/x)) which should be in [0, 2π), and represents a path
    around a helix.

    Source: problem 7 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    SIF input: Ph. Toint, Dec 1989.

    Classification: SUR2-AN-3-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        del args

        # Extract variables
        x1, x2, x3 = y

        # Compute theta using arctan2 which handles all quadrants correctly
        # arctan2 returns values in [-π, π], we need [0, 1] normalized by 2π
        raw_theta = jnp.arctan2(x2, x1) / (2.0 * jnp.pi)
        # Shift negative values to positive range [0, 1]
        raw_theta = jnp.where(raw_theta < 0, raw_theta + 1.0, raw_theta)

        # Compute the residuals as in AMPL
        r1 = 10.0 * (x3 - 10.0 * raw_theta)
        r2 = 10.0 * (jnp.sqrt(x1**2 + x2**2) - 1.0)
        r3 = x3

        # Return sum of squared residuals
        return r1**2 + r2**2 + r3**2

    def y0(self):
        # Starting point from the SIF file
        return jnp.array([-1.0, 0.0, 0.0])

    def args(self):
        return None

    def expected_result(self):
        # From the SIF file and the problem description,
        # the solution is (cos(0), sin(0), 0) = (1, 0, 0)
        return jnp.array([1.0, 0.0, 0.0])

    def expected_objective_value(self):
        # At the optimal solution, all residuals should be zero
        return jnp.array(0.0)
