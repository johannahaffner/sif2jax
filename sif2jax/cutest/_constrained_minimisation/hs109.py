import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


# TODO: Human review needed
# Attempts made:
# 1. Initial implementation based on problem description
# 2. Fixed sin(0.25) coefficient from 2 to 2*sin(0.25)
# 3. Fixed constant signs based on SIF file
# 4. Negated all equality constraints to match pycutest sign convention
# 5. Simplified double negations
# Suspected issues:
# - Sign conventions differ between pycutest and our implementation
# - Jacobian signs for x8 and x9 derivatives don't match (difference of 100.352 = 2*a)
# - Constraint values pass some tests but fail at ones vector
# Resources needed:
# - Clarification on pycutest sign conventions for this specific problem
# - Access to pycutest source code or detailed documentation for HS109
class HS109(AbstractConstrainedMinimisation):
    """Problem 109 from the Hock-Schittkowski test collection.

    A 9-variable problem with cubic objective and complex trigonometric constraints.

    f(x) = 3x₁ + 1.E-6x₁³ + 2x₂ + .522074E-6x₂³

    Subject to:
        Four inequality constraints and six equality constraints
        involving trigonometric functions and variable bounds

    Source: problem 109 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Beuneu [9]

    Classification: PGR-P1-5
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3, x4, x5, x6, x7, x8, x9 = y
        return 3 * x1 + 1.0e-6 * x1**3 + 2 * x2 + 0.522074e-6 * x2**3

    @property
    def y0(self):
        return jnp.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )  # not feasible according to the problem

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        # Solution is given in Appendix A according to the problem
        return None  # Would need Appendix A data

    @property
    def expected_objective_value(self):
        return jnp.array(5362.06928)

    @property
    def bounds(self):
        # Bounds from the PDF
        lower = jnp.array([0.0, 0.0, -0.55, -0.55, 196.0, 196.0, 196.0, -400.0, -400.0])
        upper = jnp.array(
            [jnp.inf, jnp.inf, 0.55, 0.55, 252.0, 252.0, 252.0, 800.0, 800.0]
        )
        return (lower, upper)

    def constraint(self, y):
        x1, x2, x3, x4, x5, x6, x7, x8, x9 = y

        # Constants from the PDF
        a = 50.176
        c = jnp.cos(0.25)

        # Four inequality constraints
        ineq1 = x4 - x3 + 0.55
        ineq2 = x3 - x4 + 0.55
        ineq3 = 2250000 - x1**2 - x8**2
        ineq4 = 2250000 - x2**2 - x9**2

        # Six equality constraints with trigonometric functions
        # Note: c = cos(0.25) ≈ 0.9689124217106447
        # From SIF file: 2B = 2*sin(0.25) ≈ 0.49740395925452294
        b = jnp.sin(0.25)

        # Equality constraints
        # Simplified after negation
        eq1 = (
            a * x1
            - x5 * x6 * jnp.sin(-x3 - 0.25)
            - x5 * x7 * jnp.sin(-x4 - 0.25)
            - 2 * b * x5**2
            + 400 * a
        )

        eq2 = (
            a * x2
            - x5 * x6 * jnp.sin(x3 - 0.25)
            - x6 * x7 * jnp.sin(x3 - x4 - 0.25)
            - 2 * b * x6**2
            + 400 * a
        )

        eq3 = (
            -x5 * x7 * jnp.sin(x4 - 0.25)
            - x6 * x7 * jnp.sin(x4 - x3 - 0.25)
            - 2 * b * x7**2
            + 881.779 * a
        )

        eq4 = (
            -a * x8
            - x5 * x6 * jnp.cos(-x3 - 0.25)
            - x5 * x7 * jnp.cos(-x4 - 0.25)
            + 2 * c * x5**2
            - 0.7533e-3 * a * x5**2
            - 200 * a
        )

        eq5 = (
            -a * x9
            - x5 * x6 * jnp.cos(x3 - 0.25)
            - x6 * x7 * jnp.cos(x3 - x4 - 0.25)
            + 2 * c * x6**2
            - 0.7533e-3 * a * x6**2
            - 200 * a
        )

        eq6 = (
            -x5 * x7 * jnp.cos(x4 - 0.25)
            - x6 * x7 * jnp.cos(x4 - x3 - 0.25)
            + 2 * c * x7**2
            - 0.7533e-3 * a * x7**2
            - 22.938 * a
        )

        inequality_constraints = jnp.array([ineq1, ineq2, ineq3, ineq4])
        equality_constraints = jnp.array([eq1, eq2, eq3, eq4, eq5, eq6])

        return equality_constraints, inequality_constraints
