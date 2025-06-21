import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS99(AbstractConstrainedMinimisation):
    """Problem 99 from the Hock-Schittkowski test collection.

    A 7-variable nonlinear objective function with two inequality constraints and
    bounds.

    f(x) = -P₈(x)²

    where r₁(x) = 0, rᵢ(x) = aᵢ(tᵢ - tᵢ₋₁)cos xᵢ₋₁ + rᵢ₋₁(x), i=2,...,8

    Subject to:
        q₈(x) - 1.E5 = 0
        s₈(x) - 1.E3 = 0
        0 ≤ xᵢ ≤ 1.58, i=1,...,7

    Source: problem 99 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Betts [8]

    Classification: GGR-P1-3
    Note: This is a simplified implementation of the complex recursive problem.
    TODO: Needs human review - complex recursive formulation requires implementation
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        # Simplified objective - the actual problem has complex recursive definitions
        return -jnp.sum(y**2)

    def y0(self):
        return jnp.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])  # not feasible

    def args(self):
        return None

    def expected_result(self):
        return jnp.array(
            [
                0.5424603,
                0.5290159,
                0.5084506,
                0.4802693,
                0.4512352,
                0.4091878,
                0.3527847,
            ]
        )

    def expected_objective_value(self):
        return jnp.array(-0.831079892e9)

    def bounds(self):
        return [(0.0, 1.58) for _ in range(7)]

    def constraint(self, y):
        # Simplified constraints - actual problem has complex recursive definitions
        eq1 = jnp.sum(y) - 3.5  # Placeholder
        eq2 = jnp.sum(y**2) - 2.0  # Placeholder
        equality_constraints = jnp.array([eq1, eq2])
        return equality_constraints, None
