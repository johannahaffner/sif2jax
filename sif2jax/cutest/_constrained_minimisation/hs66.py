import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS66(AbstractConstrainedMinimisation):
    """Problem 66 from the Hock-Schittkowski test collection.

    A 3-variable linear objective function with two inequality constraints and bounds.

    f(x) = 0.2*x₃ - 0.8*x₁

    Subject to:
        x₂ - exp(x₁) ≥ 0
        x₃ - exp(x₂) ≥ 0
        0 ≤ x₁ ≤ 100
        0 ≤ x₂ ≤ 100
        0 ≤ x₃ ≤ 10

    Source: problem 66 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Eckhardt [24]

    Classification: LGR-P1-1
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3 = y
        return 0.2 * x3 - 0.8 * x1

    def y0(self):
        return jnp.array([0.0, 1.05, 2.9])  # feasible according to the problem

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([0.1841264879, 1.202167873, 3.327322322])

    def expected_objective_value(self):
        return jnp.array(0.5181632741)

    def bounds(self):
        return (jnp.array([0.0, 0.0, 0.0]), jnp.array([100.0, 100.0, 10.0]))

    def constraint(self, y):
        x1, x2, x3 = y
        # Inequality constraints (g(x) ≥ 0)
        ineq1 = x2 - jnp.exp(x1)
        ineq2 = x3 - jnp.exp(x2)
        inequality_constraints = jnp.array([ineq1, ineq2])
        return None, inequality_constraints
