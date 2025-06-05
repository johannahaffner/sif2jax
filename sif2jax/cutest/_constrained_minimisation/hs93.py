import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS93(AbstractConstrainedMinimisation):
    """Problem 93 from the Hock-Schittkowski test collection.

    A 6-variable nonlinear objective function with two equality constraints.

    f(x) = 0.0204*x₁*x₄*(x₁ + x₂ + x₃) + 0.0187*x₂*x₃*(x₁ + 1.57*x₂ + x₄)
           + 0.0607*x₁*x₄*x₅²*(x₁ + x₂ + x₃) + 0.0437*x₂*x₃*x₆²*(x₁ + 1.57*x₂ + x₄)

    Subject to:
        x₁ + x₂ + x₃ + x₄ + x₅ + x₆ = 6
        x₁*x₄ + x₂*x₃ + x₅² + x₆² = 4
        0 ≤ xᵢ, i=1,...,6

    Source: problem 93 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Classification: PGR-P1-5
    """

    def objective(self, y, args):
        x1, x2, x3, x4, x5, x6 = y
        return (
            0.0204 * x1 * x4 * (x1 + x2 + x3)
            + 0.0187 * x2 * x3 * (x1 + 1.57 * x2 + x4)
            + 0.0607 * x1 * x4 * x5**2 * (x1 + x2 + x3)
            + 0.0437 * x2 * x3 * x6**2 * (x1 + 1.57 * x2 + x4)
        )

    def y0(self):
        return jnp.array([5.54, 4.4, 12.02, 11.82, 0.702, 0.852])

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([5.332666, 4.656744, 10.43299, 12.08230, 0.7526074, 0.8420251])

    def expected_objective_value(self):
        return jnp.array(135.075961)

    def bounds(self):
        return [(0.0, None) for _ in range(6)]

    def constraint(self, y):
        x1, x2, x3, x4, x5, x6 = y
        # Equality constraints
        eq1 = x1 + x2 + x3 + x4 + x5 + x6 - 6
        eq2 = x1 * x4 + x2 * x3 + x5**2 + x6**2 - 4
        equality_constraints = jnp.array([eq1, eq2])
        return equality_constraints, None
