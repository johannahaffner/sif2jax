import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS64(AbstractConstrainedMinimisation):
    """Problem 64 from the Hock-Schittkowski test collection.

    A 3-variable nonlinear objective function with one inequality constraint and bounds.

    f(x) = 5*x₁ + 50000/x₁ + 20*x₂ + 72000/x₂ + 10*x₃ + 144000/x₃

    Subject to:
        4/x₁ + 32/x₂ + 120/x₃ ≤ 1
        1.E-5 ≤ xᵢ, i=1,2,3

    Source: problem 64 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Best [7]

    Classification: OOR2-AN-3-1
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3 = y
        return 5 * x1 + 50000 / x1 + 20 * x2 + 72000 / x2 + 10 * x3 + 144000 / x3

    def y0(self):
        return jnp.array([1.0, 1.0, 1.0])  # not feasible according to the problem

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([108.7347175, 85.12613942, 204.3247078])

    def expected_objective_value(self):
        return jnp.array(6299.842428)

    def bounds(self):
        return (
            jnp.array([1.0e-5, 1.0e-5, 1.0e-5]),
            jnp.array([jnp.inf, jnp.inf, jnp.inf]),
        )

    def constraint(self, y):
        x1, x2, x3 = y
        # Inequality constraint from SIF file
        # L type: 4/x1 + 32/x2 + 120/x3 ≤ 1
        # pycutest returns LHS - RHS
        ineq1 = 4 / x1 + 32 / x2 + 120 / x3 - 1
        inequality_constraints = jnp.array([ineq1])
        return None, inequality_constraints
