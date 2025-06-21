import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS63(AbstractConstrainedMinimisation):
    """Problem 63 from the Hock-Schittkowski test collection.

    A 3-variable nonlinear objective function with two equality constraints and bounds.

    f(x) = 1000 - x₁² - 2*x₂² - x₃² - x₁*x₂ - x₁*x₃

    Subject to:
        8*x₁ + 14*x₂ + 7*x₃ - 56 = 0
        x₁² + x₂² + x₃² - 25 = 0
        0 ≤ xᵢ, i=1,2,3

    Source: problem 63 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Himmelblau [29], Paviani [48], Sheela [57]

    Classification: QQR-P1-2
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3 = y
        return 1000 - x1**2 - 2 * x2**2 - x3**2 - x1 * x2 - x1 * x3

    def y0(self):
        return jnp.array([2.0, 2.0, 2.0])  # not feasible according to the problem

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([3.51211841, 0.216988174, 3.55217403])

    def expected_objective_value(self):
        return jnp.array(961.7151721)

    def bounds(self):
        return (jnp.array([0.0, 0.0, 0.0]), jnp.array([jnp.inf, jnp.inf, jnp.inf]))

    def constraint(self, y):
        x1, x2, x3 = y
        # Equality constraints
        eq1 = 8 * x1 + 14 * x2 + 7 * x3 - 56
        eq2 = x1**2 + x2**2 + x3**2 - 25
        equality_constraints = jnp.array([eq1, eq2])
        return equality_constraints, None
