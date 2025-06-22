import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS83(AbstractConstrainedMinimisation):
    """Problem 83 from the Hock-Schittkowski test collection (Colville No.3).

    A 5-variable nonlinear objective function with inequality constraints and bounds.

    f(x) = 5.3578547*x₃² + 0.8356891*x₁*x₅ + 37.293239*x₁ - 40792.141

    Subject to:
        92 ≥ a₁ + a₂*x₂*x₅ + a₃*x₁*x₄ - a₄*x₄*x₅ ≥ 0
        20 ≥ a₅ + a₆*x₂*x₅ + a₇*x₁*x₂ + a₈*x₃² - 90 ≥ 0
        5 ≥ a₉ + a₁₀*x₃*x₅ + a₁₁*x₁*x₃ + a₁₂*x₃*x₄ - 20 ≥ 0
        78 ≤ x₁ ≤ 102, 33 ≤ x₂ ≤ 45, 27 ≤ xᵢ ≤ 45, i=3,4,5

    Source: problem 83 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Classification: QQR-P1-4
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3, x4, x5 = y
        return 5.3578547 * x3**2 + 0.8356891 * x1 * x5 + 37.293239 * x1 - 40792.141

    def y0(self):
        return jnp.array([78.0, 33.0, 27.0, 27.0, 27.0])  # not feasible

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([78.0, 33.0, 29.99526, 45.0, 36.77581])

    def expected_objective_value(self):
        return jnp.array(-30665.53867)

    def bounds(self):
        return (
            jnp.array([78.0, 33.0, 27.0, 27.0, 27.0]),
            jnp.array([102.0, 45.0, 45.0, 45.0, 45.0]),
        )

    def constraint(self, y):
        x1, x2, x3, x4, x5 = y

        # Parameters from SIF file
        a1 = 85.334407
        a2 = 0.0056858
        a3 = 0.0006262
        a4 = 0.0022053
        a5 = 80.51249
        a6 = 0.0071317
        a7 = 0.0029955
        a8 = 0.0021813
        a9 = 9.300961
        a10 = 0.0047026
        a11 = 0.0012547
        a12 = 0.0019085

        # Three ranged constraints from SIF file
        # pycutest returns ranged constraints shifted by negative of constants
        # C1: 0 <= -a1 + a2*x2*x5 + a3*x1*x4 - a4*x3*x5 <= 92
        # pycutest returns: (raw expression) - (-a1) = raw + a1
        c1 = a2 * x2 * x5 + a3 * x1 * x4 - a4 * x3 * x5 + a1

        # C2: 0 <= -(a5-90) + a6*x2*x5 + a7*x1*x2 + a8*x3^2 <= 20
        # pycutest returns: raw - (-(a5-90)) = raw + (a5-90)
        c2 = a6 * x2 * x5 + a7 * x1 * x2 + a8 * x3**2 + (a5 - 90)

        # C3: 0 <= -(a9-20) + a10*x3*x5 + a11*x1*x3 + a12*x3*x4 <= 5
        # pycutest returns: raw - (-(a9-20)) = raw + (a9-20)
        c3 = a10 * x3 * x5 + a11 * x1 * x3 + a12 * x3 * x4 + (a9 - 20)

        inequality_constraints = jnp.array([c1, c2, c3])
        return None, inequality_constraints
