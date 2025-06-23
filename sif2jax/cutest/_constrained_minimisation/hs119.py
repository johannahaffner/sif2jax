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
        # Build sparse matrix a using COO format indices and values
        row_indices = jnp.array(
            [
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                2,
                3,
                3,
                3,
                3,
                4,
                4,
                4,
                4,
                4,
                5,
                5,
                5,
                6,
                6,
                6,
                7,
                7,
                7,
                8,
                8,
                8,
                9,
                9,
                10,
                10,
                11,
                11,
                12,
                12,
                13,
                14,
                15,
            ]
        )
        col_indices = jnp.array(
            [
                0,
                3,
                6,
                7,
                15,
                1,
                2,
                6,
                9,
                2,
                6,
                8,
                9,
                13,
                3,
                6,
                10,
                14,
                4,
                5,
                9,
                11,
                15,
                5,
                7,
                14,
                6,
                10,
                12,
                7,
                9,
                14,
                8,
                11,
                15,
                9,
                13,
                10,
                12,
                11,
                13,
                12,
                13,
                13,
                14,
                15,
            ]
        )

        # Compute terms: y² + y + 1 for all variables once
        terms = y**2 + y + 1

        # Vectorized computation using advanced indexing
        term_i = terms[row_indices]
        term_j = terms[col_indices]

        # Sum all contributions (all values in sparse matrix are 1)
        objective_sum = jnp.sum(term_i * term_j)

        return objective_sum

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
        # Build sparse matrix b using COO format
        b_row_indices = jnp.array(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                2,
                2,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                4,
                4,
                4,
                4,
                4,
                4,
                5,
                5,
                5,
                5,
                5,
                5,
                5,
                6,
                6,
                6,
                6,
                6,
                7,
                7,
                7,
                7,
                7,
                7,
                7,
            ]
        )
        b_col_indices = jnp.array(
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                0,
                2,
                3,
                4,
                6,
                9,
                0,
                1,
                4,
                5,
                7,
                10,
                0,
                1,
                2,
                3,
                5,
                6,
                11,
                3,
                4,
                5,
                6,
                7,
                12,
                1,
                2,
                4,
                5,
                6,
                7,
                13,
                0,
                3,
                6,
                8,
                14,
                1,
                2,
                3,
                4,
                6,
                7,
                15,
            ]
        )
        b_values = jnp.array(
            [
                0.22,
                0.20,
                0.19,
                0.25,
                0.15,
                0.11,
                0.12,
                0.13,
                1,
                -1.46,
                -1.30,
                1.82,
                -1.15,
                0.80,
                1,
                1.29,
                -0.89,
                -1.16,
                -0.96,
                -0.49,
                1,
                -1.10,
                -1.06,
                0.95,
                -0.54,
                -1.78,
                -0.41,
                1,
                -1.43,
                1.51,
                0.59,
                -0.33,
                -0.43,
                1,
                -1.72,
                -0.33,
                1.62,
                1.24,
                0.21,
                -0.26,
                1,
                1.12,
                0.31,
                1.12,
                -0.36,
                1,
                0.45,
                0.26,
                -1.10,
                0.58,
                -1.03,
                0.10,
                1,
            ]
        )

        # c vector from AMPL formulation
        c = jnp.array([2.5, 1.1, -3.1, -3.5, 1.3, 2.1, 2.3, -1.5])

        # Vectorized constraint computation using scatter_add
        # For each constraint i, compute: sum(b[i,j] * y[j]) - c[i]
        b_y_products = b_values * y[b_col_indices]
        equality_constraints = jnp.zeros(8).at[b_row_indices].add(b_y_products) - c

        return equality_constraints, None
