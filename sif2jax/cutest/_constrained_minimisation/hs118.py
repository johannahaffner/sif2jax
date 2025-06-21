import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS118(AbstractConstrainedMinimisation):
    """Problem 118 from the Hock-Schittkowski test collection.

    A 15-variable quadratic problem with many inequality constraints.

    f(x) = ∑(2.3x₃ₖ₊₁ + 0.0001x₃ₖ₊₁² + 1.7x₃ₖ₊₂ + 0.0001x₃ₖ₊₂²
           k=0 to 4
           + 2.2x₃ₖ₊₃ + 0.00015x₃ₖ₊₃²)

    Subject to:
        Seventeen inequality constraints (12 ranged, 5 one-sided) and variable bounds

    Source: problem 118 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Bartholomew-Biggs [4]

    Classification: QLR-P1-2
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        objective_sum = 0.0

        for k in range(5):  # k = 0, 1, 2, 3, 4
            x3k1 = y[3 * k]  # x₃ₖ₊₁: indices 0, 3, 6, 9, 12
            x3k2 = y[3 * k + 1]  # x₃ₖ₊₂: indices 1, 4, 7, 10, 13
            x3k3 = y[3 * k + 2]  # x₃ₖ₊₃: indices 2, 5, 8, 11, 14

            objective_sum += (
                2.3 * x3k1
                + 0.0001 * x3k1**2
                + 1.7 * x3k2
                + 0.0001 * x3k2**2
                + 2.2 * x3k3
                + 0.00015 * x3k3**2
            )

        return jnp.array(objective_sum)

    def y0(self):
        return jnp.array(
            [
                20.0,
                55.0,
                15.0,
                20.0,
                60.0,
                20.0,
                20.0,
                60.0,
                20.0,
                20.0,
                60.0,
                20.0,
                20.0,
                60.0,
                20.0,
            ]
        )  # feasible according to the problem

    def args(self):
        return None

    def expected_result(self):
        # Solution from PDF
        return jnp.array(
            [
                8.0,
                49.0,
                3.0,
                1.0,
                56.0,
                0.0,
                1.0,
                63.0,
                6.0,
                3.0,
                70.0,
                12.0,
                5.0,
                77.0,
                18.0,
            ]
        )

    def expected_objective_value(self):
        return jnp.array(664.8204500)

    def bounds(self):
        # Variable bounds from the PDF
        lower = jnp.array(
            [8.0, 43.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        upper = jnp.array(
            [
                21.0,
                57.0,
                16.0,
                90.0,
                120.0,
                60.0,
                90.0,
                120.0,
                60.0,
                90.0,
                120.0,
                60.0,
                90.0,
                120.0,
                60.0,
            ]
        )
        return (lower, upper)

    def constraint(self, y):
        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15 = y

        # Constraints from the SIF file
        # Note: pycutest returns ranged constraints shifted by their constants
        constraints = []

        # The SIF file defines constraints in groups, appearing in this order:
        # A(K), B(K), C(K) for K=1 to 4, alternating in the loop
        # Constraints: -7 ≤ value ≤ upper_bound
        for k in range(1, 5):  # k = 1, 2, 3, 4
            # A(K): x(3k+1) - x(3k-2), bounds: -7 ≤ A(K) ≤ 6
            idx1 = 3 * k  # x₃ₖ₊₁ (k=1: x4, k=2: x7, k=3: x10, k=4: x13)
            idx2 = 3 * k - 3  # x₃ₖ₋₂ (k=1: x1, k=2: x4, k=3: x7, k=4: x10)
            constraints.append(y[idx1] - y[idx2] + 7)  # A(K) shifted by constant

            # C(K): x(3k+2) - x(3k-1), bounds: -7 ≤ C(K) ≤ 7
            idx5 = 3 * k + 1  # x₃ₖ₊₂ (k=1: x5, k=2: x8, k=3: x11, k=4: x14)
            idx6 = 3 * k - 2  # x₃ₖ₋₁ (k=1: x2, k=2: x5, k=3: x8, k=4: x11)
            constraints.append(y[idx5] - y[idx6] + 7)  # C(K) shifted

            # B(K): x(3k+3) - x(3k), bounds: -7 ≤ B(K) ≤ 6
            idx3 = 3 * k + 2  # x₃ₖ₊₃ (k=1: x6, k=2: x9, k=3: x12, k=4: x15)
            idx4 = 3 * k - 1  # x₃ₖ (k=1: x3, k=2: x6, k=3: x9, k=4: x12)
            constraints.append(y[idx3] - y[idx4] + 7)  # B(K) shifted

        # D constraints: sum constraints (one-sided, G type)
        # These are already in pycutest convention
        constraints.append(x1 + x2 + x3 - 60)  # D1: x₁ + x₂ + x₃ - 60
        constraints.append(x4 + x5 + x6 - 50)  # D2: x₄ + x₅ + x₆ - 50
        constraints.append(x7 + x8 + x9 - 70)  # D3: x₇ + x₈ + x₉ - 70
        constraints.append(x10 + x11 + x12 - 85)  # D4: x₁₀ + x₁₁ + x₁₂ - 85
        constraints.append(x13 + x14 + x15 - 100)  # D5: x₁₃ + x₁₄ + x₁₅ - 100

        inequality_constraints = jnp.array(constraints)
        return None, inequality_constraints
