import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS73(AbstractConstrainedMinimisation):
    """Problem 73 from the Hock-Schittkowski test collection (cattle-feed).

    A 4-variable linear objective function with one equality constraint, two inequality
    constraints and bounds.

    f(x) = 24.55*x₁ + 26.75*x₂ + 39*x₃ + 40.50*x₄

    Subject to:
        2.3*x₁ + 5.6*x₂ + 11.1*x₃ + 1.3*x₄ - 5 ≥ 0
        12*x₁ + 11.9*x₂ + 41.8*x₃ + 52.1*x₄ - 21
        - 1.645*(0.28*x₁² + 0.19*x₂² + 20.5*x₃² + 0.62*x₄²)^(1/2) ≥ 0
        x₁ + x₂ + x₃ + x₄ - 1 = 0
        0 ≤ xᵢ, i=1,...,4

    Source: problem 73 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Biggs [10], Bracken, McCormick [13]

    Classification: LGI-P1-1
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3, x4 = y
        return 24.55 * x1 + 26.75 * x2 + 39 * x3 + 40.50 * x4

    def y0(self):
        return jnp.array([1.0, 1.0, 1.0, 1.0])  # not feasible according to the problem

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([0.6355216, -0.12e-11, 0.3127019, 0.05177655])

    def expected_objective_value(self):
        return jnp.array(29.894378)

    def bounds(self):
        return (
            jnp.array([0.0, 0.0, 0.0, 0.0]),
            jnp.array([jnp.inf, jnp.inf, jnp.inf, jnp.inf]),
        )

    def constraint(self, y):
        x1, x2, x3, x4 = y
        # Inequality constraints (g(x) ≥ 0)
        ineq1 = 2.3 * x1 + 5.6 * x2 + 11.1 * x3 + 1.3 * x4 - 5
        ineq2 = (
            12 * x1
            + 11.9 * x2
            + 41.8 * x3
            + 52.1 * x4
            - 21
            - 1.645
            * jnp.sqrt(0.28 * x1**2 + 0.19 * x2**2 + 20.5 * x3**2 + 0.62 * x4**2)
        )

        # Equality constraint
        eq1 = x1 + x2 + x3 + x4 - 1

        equality_constraints = jnp.array([eq1])
        inequality_constraints = jnp.array([ineq1, ineq2])
        return equality_constraints, inequality_constraints
