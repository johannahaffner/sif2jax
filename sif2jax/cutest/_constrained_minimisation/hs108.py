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

    def y0(self):
        return jnp.array(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        )  # not feasible according to the problem

    def args(self):
        return None

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

    def expected_objective_value(self):
        return jnp.array(-0.866025403)

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

        # Thirteen inequality constraints ordered as in AMPL file
        # (c1-c13, c14 is handled as bound)
        # Note: The SIF file formulation differs from AMPL/PDF. Through empirical
        # testing, pycutest expects a specific sign pattern that doesn't match
        # the documented conventions. These signs were determined by comparing
        # with pycutest output.
        c1 = -(1 - x3**2 - x4**2)  # Negated to match pycutest
        c2 = -(1 - x5**2 - x6**2)  # Negated to match pycutest
        c3 = 1 - x9**2  # Not negated
        c4 = 1 - x1**2 - (x2 - x9) ** 2  # Not negated
        c5 = (
            1 - (x1 - x5) ** 2 - (x2 - x6) ** 2
        )  # Negated to match pycutest (next line)
        c5 = -c5
        c6 = -(1 - (x1 - x7) ** 2 - (x2 - x8) ** 2)  # Negated to match pycutest
        c7 = -(1 - (x3 - x7) ** 2 - (x4 - x8) ** 2)  # Negated to match pycutest
        c8 = -(1 - (x3 - x5) ** 2 - (x4 - x6) ** 2)  # Negated to match pycutest
        c9 = 1 - x7**2 - (x8 - x9) ** 2  # Not negated
        # Note: The SIF file has a different ordering for c10-c13 than the AMPL/PDF
        c10 = x3 * x9  # SIF: CE26
        c11 = x5 * x8 - x6 * x7  # SIF: CE21 - CE22
        c12 = x1 * x4 - x2 * x3  # SIF: CE23 - CE24
        c13 = x5 * x9  # SIF: CE25

        inequality_constraints = jnp.array(
            [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13]
        )
        return None, inequality_constraints
