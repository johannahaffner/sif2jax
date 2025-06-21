import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS62(AbstractConstrainedMinimisation):
    """Problem 62 from the Hock-Schittkowski test collection.

    A 3-variable nonlinear objective function with one equality constraint and bounds.

    f(x) = -8204.37*ln((x₁+x₂+x₃+0.03)/(0.09*x₁+x₂+x₃+0.03))
           -9008.72*ln((x₂+x₃+0.03)/(0.07*x₂+x₃+0.03))
           -9330.46*ln((x₃+0.03)/(0.13*x₃+0.03))

    Subject to:
        x₁ + x₂ + x₃ - 1 = 0
        0 ≤ xᵢ ≤ 1, i=1,2,3

    Source: problem 62 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Betts [8], Picket [50]

    Classification: GLR-P1-1
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3 = y
        # Based on SIF file: elements are combinations of variables, not products
        return (
            -8204.37 * jnp.log((x1 + x2 + x3 + 0.03) / (0.09 * x1 + x2 + x3 + 0.03))
            - 9008.72 * jnp.log((x2 + x3 + 0.03) / (0.07 * x2 + x3 + 0.03))
            - 9330.46 * jnp.log((x3 + 0.03) / (0.13 * x3 + 0.03))
        )

    def y0(self):
        return jnp.array([0.7, 0.2, 0.1])  # feasible according to the problem

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([0.617812691, 0.328202223, 0.053985086])

    def expected_objective_value(self):
        return jnp.array(-26272.51448)

    def bounds(self):
        return (jnp.array([0.0, 0.0, 0.0]), jnp.array([1.0, 1.0, 1.0]))

    def constraint(self, y):
        x1, x2, x3 = y
        # Equality constraint
        eq1 = x1 + x2 + x3 - 1
        equality_constraints = jnp.array([eq1])
        return equality_constraints, None
