import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class AVGASB(AbstractConstrainedMinimisation):
    """AVGASB constrained optimization problem.

    LP problem avgas, variation b.

    Note: The original problem has INTEGER variables, but we treat them
    as continuous for compatibility with optimization algorithms.

    SIF input: S. Leyffer, May 1998.

    Classification: QLR2-AN-8-10
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return 8

    @property
    def m(self):
        """Number of constraints."""
        return 10

    def objective(self, y, args):
        """Compute the objective function."""
        del args

        x = y  # Use standard indexing (0-based)

        # Linear terms from groups (no X1 term)
        linear = -2.0 * x[1] - 1.0 * x[2] - 3.0 * x[3]
        linear = linear - 2.0 * x[4] - 4.0 * x[5] - 3.0 * x[6] - 5.0 * x[7]

        # Quadratic diagonal terms (all coefficients are 2.0)
        quad_diag = (
            2.0 * x[0] * x[0]
            + 2.0 * x[1] * x[1]
            + 2.0 * x[2] * x[2]
            + 2.0 * x[3] * x[3]
            + 2.0 * x[4] * x[4]
            + 2.0 * x[5] * x[5]
            + 2.0 * x[6] * x[6]
            + 2.0 * x[7] * x[7]
        )

        # Quadratic off-diagonal terms (all coefficients are -1.0)
        quad_off = (
            -1.0 * x[0] * x[1]
            - 1.0 * x[1] * x[2]
            - 1.0 * x[2] * x[3]
            - 1.0 * x[3] * x[4]
            - 1.0 * x[4] * x[5]
            - 1.0 * x[5] * x[6]
            - 1.0 * x[6] * x[7]
        )

        return linear + quad_diag + quad_off

    def constraint(self, y):
        """Compute the constraints."""
        x = y  # Use standard indexing (0-based)

        # All constraints are >= type
        constraints = []

        # CON1: -X1 - X2 >= -1
        constraints.append(-x[0] - x[1] + 1.0)

        # CON2: -X3 - X4 >= -1
        constraints.append(-x[2] - x[3] + 1.0)

        # CON3: -X5 - X6 >= -1
        constraints.append(-x[4] - x[5] + 1.0)

        # CON4: -X7 - X8 >= -1
        constraints.append(-x[6] - x[7] + 1.0)

        # CON5: -X1 - X3 - X5 - X7 >= -2
        constraints.append(-x[0] - x[2] - x[4] - x[6] + 2.0)

        # CON6: -X2 - X4 - X6 - X8 >= -2
        constraints.append(-x[1] - x[3] - x[5] - x[7] + 2.0)

        # CON7: 2*X1 + X3 - X7 >= 0
        constraints.append(2.0 * x[0] + x[2] - x[6])

        # CON8: 5*X1 + 3*X3 - 3*X5 - X7 >= 0
        constraints.append(5.0 * x[0] + 3.0 * x[2] - 3.0 * x[4] - x[6])

        # CON9: X2 - X4 - 3*X6 - 5*X8 >= 0
        constraints.append(x[1] - x[3] - 3.0 * x[5] - 5.0 * x[7])

        # CON10: X2 - 3*X6 - 2*X8 >= 0
        constraints.append(x[1] - 3.0 * x[5] - 2.0 * x[7])

        return None, jnp.array(constraints)

    def equality_constraints(self):
        """All constraints are inequalities."""
        return jnp.zeros(10, dtype=bool)

    def y0(self):
        """Initial guess for variables."""
        return jnp.full(8, 0.5)

    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    def bounds(self):
        """Bounds on variables."""
        # All variables bounded between 0 and 1
        lower = jnp.zeros(8)
        upper = jnp.ones(8)
        return lower, upper

    def expected_result(self):
        """Expected optimal solution."""
        return None

    def expected_objective_value(self):
        """Expected optimal objective value."""
        return None
