import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# TODO: human review required
class HIMMELBCLS(AbstractUnconstrainedMinimisation):
    """Himmelblau's nonlinear least-squares problem.

    A 2-variable problem with 2 residuals.

    Source: problem 201 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lecture Notes in Economics and Mathematical Systems 187,
    Springer Verlag, Berlin, 1981.

    SIF input: Ph. Toint, Dec 1989.
    Classification: SUR2-AN-2-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        del args
        x1, x2 = y

        # From SIF file:
        # G1 = X2 + X1² - 11.0
        # G2 = X1 + X2² - 7.0
        # Objective = G1² + G2²
        g1 = x2 + x1**2 - 11.0
        g2 = x1 + x2**2 - 7.0

        # Return the sum of squares
        return g1**2 + g2**2

    def y0(self):
        return jnp.array([1.0, 1.0])

    def args(self):
        return None

    def expected_result(self):
        # The exact solution is (±√7, ±√11)
        # We'll return one of the four solutions
        return jnp.array([jnp.sqrt(7.0), jnp.sqrt(11.0)])

    def expected_objective_value(self):
        # At the optimal solution, both residuals are 0
        return jnp.array(0.0)
