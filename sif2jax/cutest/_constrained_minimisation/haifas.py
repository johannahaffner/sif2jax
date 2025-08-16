import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HAIFAS(AbstractConstrainedMinimisation):
    """Truss Topology Design problem HAIFAS (t6-9).

    A truss topology design optimization problem with 13 variables and 9 constraints.
    This is a quadratic minimization problem with inequality constraints derived
    from structural engineering applications.

    Variables:
    - z: objective variable to be minimized
    - x(1), ..., x(12): design variables

    Objective: minimize z

    Constraints: 9 inequality constraints of the form G(i) - z - x(10) ≤ 0,
    where each G(i) contains quadratic terms in the design variables.

    Source: M. Tsibulevsky, Optimization Laboratory,
    Faculty of Industrial Engineering, Technion,
    Haifa, 32000, Israel.

    SIF input: Conn, Gould and Toint, May, 1992
    minor correction by Ph. Shott, Jan 1995.

    Classification: LQR2-AN-13-9
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        # Minimize z (the first variable)
        return y[0]

    @property
    def y0(self):
        # Starting point: all variables initialized to 0
        return jnp.zeros(13)

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        # Solution not provided in SIF file
        return None

    @property
    def expected_objective_value(self):
        # Lower bound is 0.0 according to SIF file
        return None

    @property
    def bounds(self):
        # All variables are real with no explicit bounds
        return None

    def constraint(self, y):
        z = y[0]
        x = y[1:13]  # x[0] corresponds to X(1) in SIF, etc.

        # Extract individual variables (0-indexed)
        x4, x5, x6 = x[3], x[4], x[5]  # X(4), X(5), X(6)
        x10, x11, x12 = x[9], x[10], x[11]  # X(10), X(11), X(12)

        # Constraint functions G(i) based on the SIF analysis
        g1 = 5.0 * x4**2

        g2 = 3.2 * x5**2 + 3.2 * x5 * x11 + 0.8 * x11**2

        g3 = 20.0 * x10**2 - 40.0 * x10 * x11 + 20.0 * x11**2

        g4 = 3.2 * x4**2 - 3.2 * x4 * x10 + 0.8 * x10**2

        g5 = 5.0 * x5**2

        g6 = 3.2 * x6**2 + 3.2 * x6 * x12 + 0.8 * x12**2

        g7 = 20.0 * x11**2 - 40.0 * x11 * x12 + 20.0 * x12**2

        g8 = 3.2 * x5**2 - 3.2 * x5 * x11 + 0.8 * x11**2

        g9 = 5.0 * x6**2

        # Each constraint: G(i) - z - x(10) ≤ 0
        # Rearranged as: G(i) - z - x(10)
        inequality_constraints = jnp.array(
            [
                g1 - z - x10,
                g2 - z - x10,
                g3 - z - x10,
                g4 - z - x10,
                g5 - z - x10,
                g6 - z - x10,
                g7 - z - x10,
                g8 - z - x10,
                g9 - z - x10,
            ]
        )

        return None, inequality_constraints
