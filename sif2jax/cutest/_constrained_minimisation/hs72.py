import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS72(AbstractConstrainedMinimisation):
    """Problem 72 from the Hock-Schittkowski test collection (optimal sample size).

    A 4-variable linear objective function with two inequality constraints and bounds.

    f(x) = 1 + x₁ + x₂ + x₃ + x₄

    Subject to:
        0.0401 - 4/x₁ - 2.25/x₂ - 1/x₃ - 0.25/x₄ ≥ 0
        0.010085 - 0.16/x₁ - 0.36/x₂ - 0.64/x₃ - 0.64/x₄ ≥ 0
        0.001 ≤ xᵢ ≤ (5 - i)E5, i=1,...,4

    Source: problem 72 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Bracken, McCormick [13]

    Classification: LPR-P1-1
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3, x4 = y
        return 1 + x1 + x2 + x3 + x4

    def y0(self):
        return jnp.array([1.0, 1.0, 1.0, 1.0])  # not feasible according to the problem

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([193.4071, 179.5475, 185.0186, 168.7062])

    def expected_objective_value(self):
        return jnp.array(727.67937)

    def bounds(self):
        return (
            jnp.array([0.001, 0.001, 0.001, 0.001]),
            jnp.array([4e5, 3e5, 2e5, 1e5]),
        )

    def constraint(self, y):
        x1, x2, x3, x4 = y
        # Inequality constraints (g(x) ≥ 0)
        ineq1 = 0.0401 - 4 / x1 - 2.25 / x2 - 1 / x3 - 0.25 / x4
        ineq2 = 0.010085 - 0.16 / x1 - 0.36 / x2 - 0.64 / x3 - 0.64 / x4
        inequality_constraints = jnp.array([ineq1, ineq2])
        return None, inequality_constraints
