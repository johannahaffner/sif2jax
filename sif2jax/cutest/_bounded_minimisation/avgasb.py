import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class AVGASB(AbstractBoundedMinimisation):
    """AVGASB bounded minimization problem.

    A bounded quadratic programming problem with 8 variables and linear constraints
    embedded in the objective through penalty terms. Similar to AVGASA but with
    different quadratic coefficients.

    Classification: QLR2-AN-8-10
    """

    def objective(self, y, args):
        """Compute the objective function."""
        del args

        # Linear terms
        linear = -2.0 * y[1] - 1.0 * y[2] - 3.0 * y[3] - 2.0 * y[4]
        linear = linear - 4.0 * y[5] - 3.0 * y[6] - 5.0 * y[7]

        # Quadratic diagonal terms (all coefficients are 2.0)
        quad_diag = 2.0 * jnp.sum(y * y)

        # Quadratic off-diagonal terms (all coefficients are -1.0)
        quad_off = -1.0 * (
            y[0] * y[1]
            + y[1] * y[2]
            + y[2] * y[3]
            + y[3] * y[4]
            + y[4] * y[5]
            + y[5] * y[6]
            + y[6] * y[7]
        )

        return linear + quad_diag + quad_off

    def y0(self):
        """Initial guess for variables."""
        return jnp.full(8, 0.5)

    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    def bounds(self):
        """Bounds on variables."""
        lower = jnp.zeros(8)
        upper = jnp.ones(8)
        return lower, upper

    def expected_result(self):
        """Expected optimal solution."""
        # This is a QP problem - the solution would need to be computed
        # Since we don't have the expected result from SIF, return None
        return None

    def expected_objective_value(self):
        """Expected optimal objective value."""
        # Not specified in SIF file
        return None
