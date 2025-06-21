import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS41(AbstractConstrainedMinimisation):
    """Problem 41 from the Hock-Schittkowski test collection.

    A 4-variable linear objective function with one linear equality constraint
    and bounds on variables.

    f(x) = 2 - x₁x₂x₃

    Subject to:
        x₁ + 2x₂ + 2x₃ - x₄ = 0
        0 ≤ xᵢ ≤ 1, i=1,2,3
        0 ≤ x₄ ≤ 2

    Source: problem 41 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Betts [8], Miele e.al. [42]

    Classification: PLR-T1-4
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3, x4 = y
        return 2 - x1 * x2 * x3

    def y0(self):
        return jnp.array([2.0, 2.0, 2.0, 2.0])

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 2.0])

    def expected_objective_value(self):
        return jnp.array(52.0 / 27.0)

    def bounds(self):
        lower = jnp.array([0.0, 0.0, 0.0, 0.0])
        upper = jnp.array([1.0, 1.0, 1.0, 2.0])
        return lower, upper

    def constraint(self, y):
        x1, x2, x3, x4 = y
        # Equality constraint
        eq = x1 + 2 * x2 + 2 * x3 - x4
        equality_constraints = jnp.array([eq])
        return equality_constraints, None
