import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS108(AbstractConstrainedMinimisation):
    """Problem 108 from the Hock-Schittkowski test collection.

    A 9-variable quadratic optimization problem with many inequality constraints.

    f(x) = -0.5(x₁x₄ - x₂x₃ + x₃x₉ - x₅x₉ + x₅x₈ - x₆x₇)

    Subject to:
        Thirteen inequality constraints involving quadratic terms
        One positivity constraint on x₉

    Source: problem 108 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Himmelblau [29], Pearson [49]

    Classification: QQR-P1-6
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2, x3, x4, x5, x6, x7, x8, x9 = y
        return -0.5 * (x1 * x4 - x2 * x3 + x3 * x9 - x5 * x9 + x5 * x8 - x6 * x7)

    @property
    def y0(self):
        return jnp.array(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        )  # not feasible according to the problem

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        # Solution from PDF
        return jnp.array(
            [
                0.8841292,
                0.4672425,
                0.03742076,
                0.9992996,
                0.8841292,
                0.4672424,
                0.03742076,
                0.9992996,
                2.6e-19,
            ]
        )

    @property
    def expected_objective_value(self):
        return jnp.array(-0.866025403)

    @property
    def bounds(self):
        # No explicit bounds except x₉ ≥ 0
        lower = jnp.array(
            [
                -jnp.inf,
                -jnp.inf,
                -jnp.inf,
                -jnp.inf,
                -jnp.inf,
                -jnp.inf,
                -jnp.inf,
                -jnp.inf,
                0.0,
            ]
        )
        upper = jnp.array(
            [
                jnp.inf,
                jnp.inf,
                jnp.inf,
                jnp.inf,
                jnp.inf,
                jnp.inf,
                jnp.inf,
                jnp.inf,
                jnp.inf,
            ]
        )
        return (lower, upper)

    def constraint(self, y):
        x1, x2, x3, x4, x5, x6, x7, x8, x9 = y

        # Thirteen inequality constraints as defined in SIF file
        # C1: CE1 + CE2 where CE1=x3^2, CE2=x4^2 (L type, RHS=1)
        c1 = -(1 - x3**2 - x4**2)  # Convert to <= 0 form

        # C2: CE3 + CE4 where CE3=x5^2, CE4=x6^2 (L type, RHS=1)
        c2 = -(1 - x5**2 - x6**2)  # Convert to <= 0 form

        # C3: CE5 where CE5=x9^2 (L type, RHS=1)
        c3 = -(1 - x9**2)  # Convert to <= 0 form

        # C4: CE6 + CE7 where CE6=x1^2, CE7=(x2-x9)^2 (L type, RHS=1)
        c4 = -(1 - x1**2 - (x2 - x9) ** 2)  # Convert to <= 0 form

        # C5: CE8 + CE9 where CE8=(x1-x5)^2, CE9=(x2-x6)^2 (L type, RHS=1)
        c5 = -(1 - (x1 - x5) ** 2 - (x2 - x6) ** 2)  # Convert to <= 0 form

        # C6: CE10 + CE11 where CE10=(x1-x7)^2, CE11=(x2-x8)^2 (L type, RHS=1)
        c6 = -(1 - (x1 - x7) ** 2 - (x2 - x8) ** 2)  # Convert to <= 0 form

        # C7: CE12 + CE13 where CE12=(x3-x5)^2, CE13=(x4-x6)^2 (L type, RHS=1)
        c7 = -(1 - (x3 - x5) ** 2 - (x4 - x6) ** 2)  # Convert to <= 0 form

        # C8: CE16 + CE17 where CE16=(x3-x7)^2, CE17=(x4-x8)^2 (L type, RHS=1)
        c8 = -(1 - (x3 - x7) ** 2 - (x4 - x8) ** 2)  # Convert to <= 0 form

        # C9: CE18 + CE20 where CE18=x7^2, CE20=(x8-x9)^2 (L type, RHS=1)
        c9 = -(1 - x7**2 - (x8 - x9) ** 2)  # Convert to <= 0 form

        # C10: CE26 where CE26=x3*x9 (G type, RHS=0)
        c10 = x3 * x9  # Already in >= 0 form

        # C11: CE21 - CE22 where CE21=x5*x8, CE22=x6*x7 (G type, RHS=0)
        c11 = x5 * x8 - x6 * x7  # Already in >= 0 form

        # C12: CE23 - CE24 where CE23=x1*x4, CE24=x2*x3 (G type, RHS=0)
        c12 = x1 * x4 - x2 * x3  # Already in >= 0 form

        # C13: CE25 where CE25=x5*x9 (L type, RHS=0)
        c13 = x5 * x9  # L type with RHS=0: x5*x9 <= 0, already in correct form

        inequality_constraints = jnp.array(
            [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13]
        )
        return None, inequality_constraints
