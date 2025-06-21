import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class HS45(AbstractBoundedMinimisation):
    """Problem 45 from the Hock-Schittkowski test collection.

    A 5-variable nonlinear function with bounds.

    f(x) = 2 - (1/120) * x₁ * x₂ * x₃ * x₄ * x₅

    Subject to: 0 ≤ xᵢ ≤ i for i = 1, 2, 3, 4, 5

    Source: problem 45 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Betts [8], Miele e.al. [42]

    Classification: PBR-T1-5
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3, x4, x5 = y
        return 2 - (1.0 / 120.0) * x1 * x2 * x3 * x4 * x5

    def y0(self):
        return jnp.array([2.0, 2.0, 2.0, 2.0, 2.0])

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

    def expected_objective_value(self):
        return jnp.array(1.0)

    def bounds(self):
        # 0 ≤ xi ≤ i for i = 1, 2, 3, 4, 5
        lower = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
        upper = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        return lower, upper
