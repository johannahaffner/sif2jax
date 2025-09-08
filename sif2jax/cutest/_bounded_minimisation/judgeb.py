import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class JUDGEB(AbstractBoundedMinimisation):
    """SCIPY global optimization benchmark example Judge with box constraints.

    Fit: y = x_1 + a_i * x_2 + b_i^2 * x_2 + e

    This is a 2-variable least squares problem with 20 data points.

    Source: Problem from the SCIPY benchmark set
    https://github.com/scipy/scipy/tree/master/benchmarks/benchmarks/go_benchmark_functions

    SIF input: Nick Gould, July 2021

    Classification: SBR2-MN-2-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return jnp.array(2)

    def objective(self, y, args):
        """Least squares objective function."""
        del args

        x1, x2 = y[0], y[1]

        # Data from SIF file
        a_vals = jnp.array(
            [
                0.286,
                0.973,
                0.384,
                0.276,
                0.973,
                0.543,
                0.957,
                0.948,
                0.543,
                0.797,
                0.936,
                0.889,
                0.006,
                0.828,
                0.399,
                0.617,
                0.939,
                0.784,
                0.072,
                0.889,
            ]
        )

        b_vals = jnp.array(
            [
                0.645,
                0.585,
                0.310,
                0.058,
                0.455,
                0.779,
                0.259,
                0.202,
                0.028,
                0.099,
                0.142,
                0.296,
                0.175,
                0.180,
                0.842,
                0.039,
                0.103,
                0.620,
                0.158,
                0.704,
            ]
        )

        y_vals = jnp.array(
            [
                4.284,
                4.149,
                3.877,
                0.533,
                2.211,
                2.389,
                2.145,
                3.231,
                1.998,
                1.379,
                2.106,
                1.428,
                1.011,
                2.179,
                2.858,
                1.388,
                1.651,
                1.593,
                1.046,
                2.152,
            ]
        )

        # From SIF: F(I) groups have X1 + A(I)*X2 as linear part
        # Element E contributes B(I) * (X2^2) to each F(I) group
        # GROUP TYPE L2 squares each group: (F(I) - Y(I))^2

        # So each group is: F(I) = X1 + A(I)*X2 + B(I)*X2^2
        group_vals = x1 + a_vals * x2 + b_vals * (x2**2)
        residuals = group_vals - y_vals

        return jnp.sum(residuals**2)

    @property
    def y0(self):
        """Initial guess."""
        return jnp.array([1.0, 5.0])

    @property
    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    @property
    def bounds(self):
        """Variable bounds: [-10, 10] for both variables."""
        lower = jnp.array([-10.0, -10.0])
        upper = jnp.array([10.0, 10.0])
        return lower, upper

    @property
    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value: 0.0 (from SIF comments)."""
        return jnp.asarray(0.0)
