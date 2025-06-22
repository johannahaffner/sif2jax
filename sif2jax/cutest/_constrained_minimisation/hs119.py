import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS119(AbstractConstrainedMinimisation):
    """Problem 119 from the Hock-Schittkowski test collection.

    A 16-variable optimization problem (Colville No.7) with quadratic objective.

    f(x) = ∑∑aᵢⱼ(xᵢ² + xᵢ + 1)(xⱼ² + xⱼ + 1)
           i=1 j=1 to 16

    Subject to:
        Eight equality constraints
        0 ≤ xᵢ ≤ 5, i=1,...,16

    Source: problem 119 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Colville [20], Himmelblau [29]

    Classification: PLR-P1-2
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        # a matrix from AMPL formulation (16x16 sparse matrix)
        a = jnp.zeros((16, 16))

        # Fill in the non-zero entries from AMPL data
        sparse_entries = [
            (0, 0, 1),
            (0, 3, 1),
            (0, 6, 1),
            (0, 7, 1),
            (0, 15, 1),
            (1, 1, 1),
            (1, 2, 1),
            (1, 6, 1),
            (1, 9, 1),
            (2, 2, 1),
            (2, 6, 1),
            (2, 8, 1),
            (2, 9, 1),
            (2, 13, 1),
            (3, 3, 1),
            (3, 6, 1),
            (3, 10, 1),
            (3, 14, 1),
            (4, 4, 1),
            (4, 5, 1),
            (4, 9, 1),
            (4, 11, 1),
            (4, 15, 1),
            (5, 5, 1),
            (5, 7, 1),
            (5, 14, 1),
            (6, 6, 1),
            (6, 10, 1),
            (6, 12, 1),
            (7, 7, 1),
            (7, 9, 1),
            (7, 14, 1),
            (8, 8, 1),
            (8, 11, 1),
            (8, 15, 1),
            (9, 9, 1),
            (9, 13, 1),
            (10, 10, 1),
            (10, 12, 1),
            (11, 11, 1),
            (11, 13, 1),
            (12, 12, 1),
            (12, 13, 1),
            (13, 13, 1),
            (14, 14, 1),
            (15, 15, 1),
        ]

        # Set the sparse entries
        for i, j, val in sparse_entries:
            a = a.at[i, j].set(val)

        objective_sum = 0.0
        for i in range(16):
            for j in range(16):
                term_i = y[i] ** 2 + y[i] + 1
                term_j = y[j] ** 2 + y[j] + 1
                objective_sum += a[i, j] * term_i * term_j

        return jnp.array(objective_sum)

    def y0(self):
        return jnp.array([10.0] * 16)  # not feasible according to the problem

    def args(self):
        return None

    def expected_result(self):
        # Solution from PDF
        return jnp.array(
            [
                0.03984735,
                0.7919832,
                0.2028703,
                0.8443579,
                1.126991,
                0.9347387,
                1.681962,
                0.1553009,
                1.567870,
                0.0,
                0.0,
                0.0,
                0.6602041,
                0.0,
                0.6742559,
                0.0,
            ]
        )

    def expected_objective_value(self):
        return jnp.array(244.899698)

    def bounds(self):
        # Bounds: 0 ≤ xᵢ ≤ 5 for all i
        lower = jnp.array([0.0] * 16)
        upper = jnp.array([5.0] * 16)
        return (lower, upper)

    def constraint(self, y):
        # b matrix from AMPL formulation (8x16 sparse matrix)
        b = jnp.zeros((8, 16))

        # Fill in the non-zero entries from AMPL data
        b_entries = [
            # Row 1
            (0, 0, 0.22),
            (0, 1, 0.20),
            (0, 2, 0.19),
            (0, 3, 0.25),
            (0, 4, 0.15),
            (0, 5, 0.11),
            (0, 6, 0.12),
            (0, 7, 0.13),
            (0, 8, 1),
            # Row 2
            (1, 0, -1.46),
            (1, 2, -1.30),
            (1, 3, 1.82),
            (1, 4, -1.15),
            (1, 6, 0.80),
            (1, 9, 1),
            # Row 3
            (2, 0, 1.29),
            (2, 1, -0.89),
            (2, 4, -1.16),
            (2, 5, -0.96),
            (2, 7, -0.49),
            (2, 10, 1),
            # Row 4
            (3, 0, -1.10),
            (3, 1, -1.06),
            (3, 2, 0.95),
            (3, 3, -0.54),
            (3, 5, -1.78),
            (3, 6, -0.41),
            (3, 11, 1),
            # Row 5
            (4, 3, -1.43),
            (4, 4, 1.51),
            (4, 5, 0.59),
            (4, 6, -0.33),
            (4, 7, -0.43),
            (4, 12, 1),
            # Row 6
            (5, 1, -1.72),
            (5, 2, -0.33),
            (5, 4, 1.62),
            (5, 5, 1.24),
            (5, 6, 0.21),
            (5, 7, -0.26),
            (5, 13, 1),
            # Row 7
            (6, 0, 1.12),
            (6, 3, 0.31),
            (6, 6, 1.12),
            (6, 8, -0.36),
            (6, 14, 1),
            # Row 8
            (7, 1, 0.45),
            (7, 2, 0.26),
            (7, 3, -1.10),
            (7, 4, 0.58),
            (7, 6, -1.03),
            (7, 7, 0.10),
            (7, 15, 1),
        ]

        # Set the sparse entries
        for i, j, val in b_entries:
            b = b.at[i, j].set(val)

        # c vector from AMPL formulation
        c = jnp.array([2.5, 1.1, -3.1, -3.5, 1.3, 2.1, 2.3, -1.5])

        # Eight equality constraints: ∑bᵢⱼxⱼ - cᵢ = 0 for i=1,...,8
        equality_constraints = []
        for i in range(8):
            constraint_val = jnp.sum(b[i, :] * y) - c[i]
            equality_constraints.append(constraint_val)

        equality_constraints = jnp.array(equality_constraints)
        return equality_constraints, None
