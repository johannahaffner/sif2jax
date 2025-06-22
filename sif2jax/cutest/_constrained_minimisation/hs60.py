import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS60(AbstractConstrainedMinimisation):
    """Problem 60 from the Hock-Schittkowski test collection.

    A 3-variable nonlinear objective function with one equality constraint and bounds.

    f(x) = (x₁ - 1)² + (x₁ - x₂)² + (x₂ - x₃)⁴

    Subject to:
        x₁(1 + x₂²) + x₃⁴ - 4 - 3√2 = 0
        -10 ≤ xᵢ ≤ 10, i=1,2,3

    Source: problem 60 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Betts [8], Miele e.al. [42,44]

    Classification: PPR-P1-1
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3 = y
        return (x1 - 1) ** 2 + (x1 - x2) ** 2 + (x2 - x3) ** 4

    def y0(self):
        return jnp.array([2.0, 2.0, 2.0])  # not feasible according to the problem

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([1.10485902, 1.19667419, 1.53526226])

    def expected_objective_value(self):
        return jnp.array(0.03256820025)

    def bounds(self):
        return (jnp.array([-10.0, -10.0, -10.0]), jnp.array([10.0, 10.0, 10.0]))

    def constraint(self, y):
        x1, x2, x3 = y
        # Equality constraint
        eq1 = x1 * (1 + x2**2) + x3**4 - 4 - 3 * jnp.sqrt(2)
        equality_constraints = jnp.array([eq1])
        return equality_constraints, None
