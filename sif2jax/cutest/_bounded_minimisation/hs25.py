import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class HS25(AbstractBoundedMinimisation):
    """Problem 25 from the Hock-Schittkowski test collection.

    A 3-variable nonlinear function with bounds constraints.

    f(x) = Σᵢ₌₁⁹⁹ (fᵢ(x))²

    where fᵢ(x) = -0.01i + exp(-1/x₁(uᵢ - x₂)^x₃)
    and uᵢ = 25 + (-50 ln(0.01i))^(2/3)
    for i = 1, ..., 99

    Subject to: 0.1 ≤ x₁ ≤ 100
                0 ≤ x₂ ≤ 25.6
                0 ≤ x₃ ≤ 5

    Source: problem 25 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Holzmann [32], Himmelblau [29]

    Classification: SBR-T1-1
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3 = y

        # Compute sum of squared residuals
        sum_val = 0.0
        for i in range(1, 100):  # i = 1, ..., 99
            # uᵢ = 25 + (-50 ln(0.01i))^(2/3)
            ui = 25.0 + (-50.0 * jnp.log(0.01 * i)) ** (2.0 / 3.0)

            # fᵢ(x) = -0.01i + exp(-1/x₁(uᵢ - x₂)^x₃)
            fi = -0.01 * i + jnp.exp(-(1.0 / x1) * (ui - x2) ** x3)

            sum_val += fi**2

        return jnp.array(sum_val)

    def y0(self):
        return jnp.array([100.0, 12.5, 3.0])

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([50.0, 25.0, 1.5])

    def expected_objective_value(self):
        return jnp.array(0.0)

    def bounds(self):
        return (jnp.array([0.1, 0.0, 0.0]), jnp.array([100.0, 25.6, 5.0]))

    def constraint(self, y):
        return None, None
