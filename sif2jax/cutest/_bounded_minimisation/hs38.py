import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class HS38(AbstractBoundedMinimisation):
    """Problem 38 from the Hock-Schittkowski test collection (Colville No.4).

    A 4-variable nonlinear function with bounds on all variables.

    f(x) = 100(x₂ - x₁²)² + (1 - x₁)² + 90(x₄ - x₃²)² + (1-x₃)²
           + 10.1((x₂ - 1)² + (x₄ - 1)²) + 19.8(x₂ - 1)(x₄ - 1)

    Subject to: -10 ≤ xᵢ ≤ 10, i=1,...,4

    Source: problem 38 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Colville [20], Himmelblau [29]

    Classification: PBR-T1-4
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3, x4 = y
        return (
            100 * (x2 - x1**2) ** 2
            + (1 - x1) ** 2
            + 90 * (x4 - x3**2) ** 2
            + (1 - x3) ** 2
            + 10.1 * ((x2 - 1) ** 2 + (x4 - 1) ** 2)
            + 19.8 * (x2 - 1) * (x4 - 1)
        )

    def y0(self):
        return jnp.array([-3.0, -1.0, -3.0, -1.0])

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([1.0, 1.0, 1.0, 1.0])

    def expected_objective_value(self):
        return jnp.array(0.0)

    def bounds(self):
        # -10 ≤ xi ≤ 10 for i = 1,2,3,4
        lower = jnp.array([-10.0, -10.0, -10.0, -10.0])
        upper = jnp.array([10.0, 10.0, 10.0, 10.0])
        return lower, upper
