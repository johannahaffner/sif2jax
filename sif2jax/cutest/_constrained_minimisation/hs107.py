import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS107(AbstractConstrainedMinimisation):
    """Problem 107 from the Hock-Schittkowski test collection.

    A 9-variable static power scheduling problem with equality constraints and bounds.

    f(x) = 3000x₁ + 1000x₁² + 2000x₂ + 666.667x₂³

    Subject to:
        Six equality constraints involving trigonometric functions
        Variable bounds as specified

    Note: This problem involves trigonometric functions and parameters c and d.
    c = (48.4/50.176)sin.25 ≈ 0.965*sin(0.25) ≈ 0.238
    d = (48.4/50.176)cos.25 ≈ 0.965*cos(0.25) ≈ 0.931

    Source: problem 107 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Bartholomew-Biggs [4]

    Classification: PGR-P1-4
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3, x4, x5, x6, x7, x8, x9 = y
        return 3000 * x1 + 1000 * x1**3 + 2000 * x2 + 666.667 * x2**3

    def y0(self):
        return jnp.array(
            [0.8, 0.8, 0.2, 0.2, 1.0454, 1.0454, 1.0454, 0.0, 0.0]
        )  # not feasible according to the problem

    def args(self):
        return None

    def expected_result(self):
        # Solution from PDF
        return jnp.array(
            [
                0.6670095,
                1.022388,
                0.2282879,
                0.1848217,
                1.090900,
                1.090900,
                1.069036,
                0.1066126,
                -0.3387867,
            ]
        )

    def expected_objective_value(self):
        return jnp.array(5055.011803)

    def bounds(self):
        # Bounds from the AMPL formulation
        lower = jnp.array(
            [
                0.0,
                0.0,
                -jnp.inf,
                -jnp.inf,
                0.90909,
                0.90909,
                0.90909,
                -jnp.inf,
                -jnp.inf,
            ]
        )
        upper = jnp.array(
            [
                jnp.inf,
                jnp.inf,
                jnp.inf,
                jnp.inf,
                1.0909,
                1.0909,
                1.0909,
                jnp.inf,
                jnp.inf,
            ]
        )
        return (lower, upper)

    def constraint(self, y):
        x1, x2, x3, x4, x5, x6, x7, x8, x9 = y

        # Trigonometric variables from the PDF
        y1 = jnp.sin(x8)
        y2 = jnp.cos(x8)
        y3 = jnp.sin(x9)
        y4 = jnp.cos(x9)
        y5 = jnp.sin(x8 - x9)
        y6 = jnp.cos(x8 - x9)

        # Constants from the PDF
        c = (48.4 / 50.176) * jnp.sin(0.25)  # ≈ 0.238
        d = (48.4 / 50.176) * jnp.cos(0.25)  # ≈ 0.931

        # Six equality constraints from the PDF
        eq1 = (
            0.4
            - x1
            + 2 * c * x5**2
            - x5 * x6 * (d * y1 + c * y2)
            - x5 * x7 * (d * y3 + c * y4)
        )

        eq2 = (
            0.4
            - x2
            + 2 * c * x6**2
            + x5 * x6 * (d * y1 - c * y2)
            + x6 * x7 * (d * y5 - c * y6)
        )

        eq3 = (
            0.8
            + 2 * c * x7**2
            + x5 * x7 * (d * y3 - c * y4)
            - x6 * x7 * (d * y5 + c * y6)
        )

        eq4 = (
            0.2
            - x3
            + 2 * d * x5**2
            + x5 * x6 * (c * y1 - d * y2)
            + x5 * x7 * (c * y3 - d * y4)
        )

        eq5 = (
            0.2
            - x4
            + 2 * d * x6**2
            - x5 * x6 * (c * y1 + d * y2)
            - x6 * x7 * (c * y5 + d * y6)
        )

        eq6 = (
            -0.337
            + 2 * d * x7**2
            - x5 * x7 * (c * y3 + d * y4)
            + x6 * x7 * (c * y5 - d * y6)
        )

        equality_constraints = jnp.array([eq1, eq2, eq3, eq4, eq5, eq6])
        return equality_constraints, None
