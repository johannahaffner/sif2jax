import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS35(AbstractConstrainedMinimisation):
    """Problem 35 (Beale's problem) from the Hock-Schittkowski test collection.

    A 3-variable quadratic function with one inequality constraint and bounds.

    f(x) = 9 - 8x₁ - 6x₂ - 4x₃ + 2x₁² + 2x₂² + x₃² + 2x₁x₂ + 2x₁x₃

    Subject to: 3 - x₁ - x₂ - 2x₃ ≥ 0
                0 ≤ xᵢ, i = 1,2,3

    Source: problem 35 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Asaadi [1], Charalambous [18], Dimitru [23], Sheela [57]

    Classification: QLR-T1-3
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3 = y
        return (
            9.0
            - 8.0 * x1
            - 6.0 * x2
            - 4.0 * x3
            + 2.0 * x1**2
            + 2.0 * x2**2
            + x3**2
            + 2.0 * x1 * x2
            + 2.0 * x1 * x3
        )

    def y0(self):
        return jnp.array([0.5, 0.5, 0.5])

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([4.0 / 3.0, 7.0 / 9.0, 4.0 / 9.0])

    def expected_objective_value(self):
        return jnp.array(1.0 / 9.0)

    def bounds(self):
        return (jnp.array([0.0, 0.0, 0.0]), jnp.array([jnp.inf, jnp.inf, jnp.inf]))

    def constraint(self, y):
        x1, x2, x3 = y
        # Inequality constraint: 3 - x₁ - x₂ - 2x₃ ≥ 0
        inequality_constraint = jnp.array([3.0 - x1 - x2 - 2.0 * x3])
        return None, inequality_constraint
