import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS70(AbstractConstrainedMinimisation):
    """HS70 problem - Water flow routing problem.

    This problem arises in water flow routing.

    TODO: Human review needed - test failures. The implementation follows the SIF file
    but tests fail. Challenges include:
    - Complex Y1 and Y2 function definitions with multiple terms
    - Y1 uses (1-x3) while Y2 uses x3 in different places
    - The functions involve intricate combinations of power, exponential and other terms
    - Constraint involves a product element (x3*x4)

    Source: problem 70 incorrectly stated in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    SIF input: Nick Gould, August 1991, modified May 2024

    Classification: SQR2-MN-4-1
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return 4

    @property
    def m(self):
        """Number of constraints."""
        return 1  # One inequality constraint

    def objective(self, y, args):
        """Compute the sum of squares objective."""
        del args
        x = y

        # Data points
        c = jnp.array(
            [
                0.1,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
                12.0,
                13.0,
                14.0,
                15.0,
                16.0,
                17.0,
                18.0,
            ]
        )

        y_data = jnp.array(
            [
                0.00189,
                0.1038,
                0.268,
                0.506,
                0.577,
                0.604,
                0.725,
                0.898,
                0.947,
                0.845,
                0.702,
                0.528,
                0.385,
                0.257,
                0.159,
                0.0869,
                0.0453,
                0.01509,
                0.00189,
            ]
        )

        # Compute Y1 and Y2 functions for each data point
        def compute_y1(x1, x2, x3, x4, ci):
            b = x3 + x4 * (1.0 - x3)
            ci_scaled = ci / 7.658
            p0 = 1.0 + 1.0 / (12.0 * x2)
            p1 = 1.0 / p0
            p2 = x3
            p3 = b**x2
            c4 = jnp.sqrt(1.0 / (2.0 * jnp.pi))
            p4 = c4 * jnp.sqrt(x2)
            p5 = (1.0 / ci_scaled) * (ci_scaled**x2)
            p6 = jnp.exp(x2 * (1.0 - ci_scaled * b))
            return p1 * p2 * p3 * p4 * p5 * p6

        def compute_y2(x1, x2, x3, x4, ci):
            b = x3 + x4 * (1.0 - x3)
            ci_scaled = ci / 7.658
            p0 = 1.0 + 1.0 / (12.0 * x1)
            p1 = 1.0 / p0
            p2 = 1.0 - x3  # This is different from Y1!
            p3 = (b / x4) ** x1  # This is also different from Y1!
            c4 = jnp.sqrt(1.0 / (2.0 * jnp.pi))
            p4 = c4 * jnp.sqrt(x1)
            p5 = (1.0 / ci_scaled) * (ci_scaled**x1)
            p6 = jnp.exp(x1 * (1.0 - ci_scaled * b / x4))  # Also adjusted for b/x4
            return p1 * p2 * p3 * p4 * p5 * p6

        # Compute residuals
        residuals = jnp.zeros(19)
        for i in range(19):
            y1_val = compute_y1(x[0], x[1], x[2], x[3], c[i])
            y2_val = compute_y2(x[0], x[1], x[2], x[3], c[i])
            residuals = residuals.at[i].set(y1_val + y2_val - y_data[i])

        # Objective is sum of squares
        return jnp.sum(residuals**2)

    def constraint(self, y):
        """Compute the constraints."""
        x = y

        # Constraint: x3 + x4 + x3*x4 >= 1.0 (written as g(x) >= 0)
        # This comes from the linear part (x3 + x4) plus the PROD element (x3*x4)
        # with coefficient -1
        ineq_constraint = x[2] + x[3] + x[2] * x[3] - 1.0

        # Return as inequality constraint (empty equality constraints)
        return jnp.array([]), jnp.array([ineq_constraint])

    def equality_constraints(self):
        """Specify which constraints are equalities."""
        return jnp.array([])  # No equality constraints

    def y0(self):
        """Initial guess."""
        return jnp.array([2.0, 4.0, 0.04, 2.0])

    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    def bounds(self):
        """Variable bounds."""
        lower = jnp.array([0.00001, 0.00001, 0.00001, 0.00001])
        upper = jnp.array([100.0, 100.0, 1.0, 100.0])
        return lower, upper

    def expected_result(self):
        """Expected optimal solution."""
        return jnp.array([12.27695, 4.631788, 0.3128625, 2.029290])

    def expected_objective_value(self):
        """Expected optimal objective value."""
        return jnp.array(0.007498464)  # From SIF file solution comment
