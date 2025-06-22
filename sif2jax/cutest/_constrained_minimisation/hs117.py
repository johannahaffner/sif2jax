import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractConstrainedMinimisation


class HS117(AbstractConstrainedMinimisation):
    """Problem 117 from the Hock-Schittkowski test collection.

    A 15-variable optimization problem (Colville No.2, Shell Dual) with complex
    objective.

    f(x) = -∑bⱼxⱼ + ∑∑cᵢⱼx₁₀₊ᵢx₁₀₊ⱼ + 2∑dⱼx₁₀₊ⱼ³
           j=1 to 10  i=1 j=1 to 5              j=1 to 5

    Subject to:
        Five inequality constraints
        0 ≤ xᵢ, i=1,...,15

    Source: problem 117 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Colville [20], Himmelblau [29]

    Classification: PQR-P1-1
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        # Coefficients from AMPL formulation
        b = inexact_asarray(jnp.array([-40, -2, -0.25, -4, -4, -1, -40, -60, 5, 1]))

        # c matrix (5x5) from AMPL
        c = inexact_asarray(
            jnp.array(
                [
                    [30, -20, -10, 32, -10],
                    [-20, 39, -6, -31, 32],
                    [-10, -6, 10, -6, -10],
                    [32, -31, -6, 39, -20],
                    [-10, 32, -10, -20, 30],
                ]
            )
        )

        d = inexact_asarray(jnp.array([4, 8, 10, 6, 2]))

        # First 10 variables
        x_first10 = y[:10]
        # Last 5 variables (x11 to x15, indices 10-14)
        x_last5 = y[10:15]

        # Calculate objective components
        term1 = -jnp.sum(b * x_first10)

        # Quadratic term
        term2 = 0.0
        for i in range(5):
            for j in range(5):
                term2 += c[i, j] * x_last5[i] * x_last5[j]

        # Cubic term
        term3 = 2 * jnp.sum(d * x_last5**3)

        return term1 + term2 + term3

    def y0(self):
        return jnp.array(
            [
                0.001,
                0.001,
                0.001,
                0.001,
                0.001,
                0.001,
                60.0,
                0.001,
                0.001,
                0.001,
                0.001,
                0.001,
                0.001,
                0.001,
                0.001,
            ]
        )  # from AMPL formulation

    def args(self):
        return None

    def expected_result(self):
        # Solution from PDF
        return jnp.array(
            [
                0.0,
                0.0,
                5.174136,
                0.0,
                3.061093,
                11.83968,
                0.0,
                0.0,
                0.1039071,
                0.0,
                0.2999929,
                0.3334709,
                0.3999910,
                0.4283145,
                0.2239607,
            ]
        )

    def expected_objective_value(self):
        return jnp.array(32.348679)

    def bounds(self):
        # Bounds: 0 ≤ xᵢ for all i
        lower = jnp.array([0.0] * 15)
        upper = jnp.array([jnp.inf] * 15)
        return (lower, upper)

    def constraint(self, y):
        # Coefficients from AMPL formulation
        a = inexact_asarray(
            jnp.array(
                [
                    [-16, 2, 0, 1, 0],  # a[1,:]
                    [0, -2, 0, 4, 2],  # a[2,:]
                    [-3.5, 0, 2, 0, 0],  # a[3,:]
                    [0, -2, 0, -4, -1],  # a[4,:]
                    [0, -9, -2, 1, -2.8],  # a[5,:]
                    [2, 0, -4, 0, 0],  # a[6,:]
                    [-1, -1, -1, -1, -1],  # a[7,:]
                    [-1, -2, -3, -2, -1],  # a[8,:]
                    [1, 2, 3, 4, 5],  # a[9,:]
                    [1, 1, 1, 1, 1],  # a[10,:]
                ]
            )
        )

        # c matrix (5x5) from AMPL
        c = inexact_asarray(
            jnp.array(
                [
                    [30, -20, -10, 32, -10],
                    [-20, 39, -6, -31, 32],
                    [-10, -6, 10, -6, -10],
                    [32, -31, -6, 39, -20],
                    [-10, 32, -10, -20, 30],
                ]
            )
        )

        d = inexact_asarray(jnp.array([4, 8, 10, 6, 2]))
        e = inexact_asarray(jnp.array([-15, -27, -36, -18, -12]))

        x_first10 = y[:10]
        x_last5 = y[10:15]

        # Five inequality constraints from AMPL formulation (vectorized)
        # 2*c[k,j]*x[10+k] + 3*d[j]*x[10+j]^2 + e[j] - sum{k in 1..10} a[k,j]*x[k] >= 0

        # term1: 2*sum(c[k,j]*x[10+k]) for each j
        term1 = 2 * jnp.sum(c * x_last5[:, None], axis=0)  # shape: (5,)

        # term2: 3*d[j]*x[10+j]^2 for each j
        term2 = 3 * d * x_last5**2  # shape: (5,)

        # term3: e[j] for each j
        term3 = e  # shape: (5,)

        # term4: -sum{k in 1..10} a[k,j]*x[k] for each j
        term4 = -jnp.sum(a * x_first10[:, None], axis=0)  # shape: (5,)

        inequality_constraints = term1 + term2 + term3 + term4
        return None, inequality_constraints
