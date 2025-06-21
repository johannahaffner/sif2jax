import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS56(AbstractConstrainedMinimisation):
    """Problem 56 from the Hock-Schittkowski test collection.

    A 7-variable linear objective function with four nonlinear equality constraints.

    f(x) = -x₁x₂x₃

    Subject to:
        x₁ - 4.2sin²x₄ = 0
        x₂ - 4.2sin²x₅ = 0
        x₃ - 4.2sin²x₆ = 0
        x₁ + 2x₂ + 2x₃ - 7.2sin²x₇ = 0

    Source: problem 56 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Brusch [15]

    Classification: PGR-T1-2
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3, x4, x5, x6, x7 = y
        return -x1 * x2 * x3

    def y0(self):
        # Starting point from SIF file
        # Note: There's a typo mentioned in the SIF file about decimal points
        # The values below match what pycutest uses
        return jnp.array(
            [1.0, 1.0, 1.0, 0.50973968, 0.50973968, 0.50973968, 0.98511078]
        )

    def args(self):
        return None

    def expected_result(self):
        # Solution involves arcsin functions
        # a = arcsin(√(1/4.2))
        # b = arcsin(√(5/7.2))
        # c = arcsin(√(4/7))
        # d = arcsin(√(2/7))
        # x* = (2.4, 1.2, 1.2, ±c+jπ, ±d+kπ, ±d+lπ, (π+0.5)π)
        # where j,k,l,r = 0,±1,±2,...
        # We provide one feasible solution
        c = jnp.arcsin(jnp.sqrt(4.0 / 7.0))
        d = jnp.arcsin(jnp.sqrt(2.0 / 7.0))
        r_val = 0.5  # choosing r=0.5 for (r+0.5)π
        return jnp.array([2.4, 1.2, 1.2, c, d, d, (r_val + 0.5) * jnp.pi])

    def expected_objective_value(self):
        return jnp.array(-3.456)

    def bounds(self):
        return None

    def constraint(self, y):
        x1, x2, x3, x4, x5, x6, x7 = y
        # Precompute sin values and their squares
        sin_vals = jnp.sin(y[3:])  # sin(x4), sin(x5), sin(x6), sin(x7)
        sin_sq = sin_vals**2

        # Equality constraints
        eq1 = x1 - 4.2 * sin_sq[0]
        eq2 = x2 - 4.2 * sin_sq[1]
        eq3 = x3 - 4.2 * sin_sq[2]
        eq4 = x1 + 2 * x2 + 2 * x3 - 7.2 * sin_sq[3]
        equality_constraints = jnp.array([eq1, eq2, eq3, eq4])
        return equality_constraints, None
