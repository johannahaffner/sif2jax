import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class HS4(AbstractBoundedMinimisation):
    """Problem 4 from the Hock-Schittkowski test collection.

    A 2-variable nonlinear function with bounds on both variables.

    f(x) = (1/3)(x₁ + 1)³ + x₂

    Subject to: 1 ≤ x₁, 0 ≤ x₂

    Source: problem 4 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Asaadi [1]

    Classification: PBR-T1-3
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2 = y
        return (1.0 / 3.0) * (x1 + 1) ** 3 + x2

    def y0(self):
        return jnp.array([1.125, 0.125])

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([1.0, 0.0])

    def expected_objective_value(self):
        return jnp.array(8.0 / 3.0)

    def bounds(self):
        # x1 >= 1, x2 >= 0
        lower = jnp.array([1.0, 0.0])
        upper = jnp.array([jnp.inf, jnp.inf])
        return lower, upper
