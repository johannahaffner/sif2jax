import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS55(AbstractConstrainedMinimisation):
    """Problem 55 from the Hock-Schittkowski test collection.

    A 6-variable nonlinear objective function with six linear equality constraints
    and bounds on variables.

    f(x) = x₁ + 2x₂ + 4x₅ + exp(x₁x₄)

    Subject to:
        x₁ + 2x₂ + 5x₅ - 6 = 0
        x₁ + x₂ + x₃ - 3 = 0
        x₄ + x₅ + x₆ - 2 = 0
        x₁ + x₄ - 1 = 0
        x₂ + x₅ - 2 = 0
        x₃ + x₆ - 2 = 0
        0 ≤ x₁ ≤ 1
        0 ≤ x₂
        0 ≤ x₃
        0 ≤ x₄ ≤ 1
        0 ≤ x₅
        0 ≤ x₆

    Source: problem 55 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Hsia [33]

    Classification: OLR2-AN-6-6
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3, x4, x5, x6 = y
        return x1 + 2 * x2 + 4 * x5 + jnp.exp(x1 * x4)

    def y0(self):
        return jnp.array([1.0, 2.0, 0.0, 0.0, 0.0, 2.0])

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([0.0, 4.0 / 3.0, 5.0 / 3.0, 1.0, 2.0 / 3.0, 1.0 / 3.0])

    def expected_objective_value(self):
        return jnp.array(19.0 / 3.0)

    def bounds(self):
        lower = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # From SIF file: only X1 and X4 have upper bounds of 1.0
        upper = jnp.array([1.0, jnp.inf, jnp.inf, 1.0, jnp.inf, jnp.inf])
        return lower, upper

    def constraint(self, y):
        x1, x2, x3, x4, x5, x6 = y
        # Equality constraints
        eq1 = x1 + 2 * x2 + 5 * x5 - 6
        eq2 = x1 + x2 + x3 - 3
        eq3 = x4 + x5 + x6 - 2
        eq4 = x1 + x4 - 1
        eq5 = x2 + x5 - 2
        eq6 = x3 + x6 - 2
        equality_constraints = jnp.array([eq1, eq2, eq3, eq4, eq5, eq6])
        return equality_constraints, None
