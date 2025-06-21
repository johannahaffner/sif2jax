import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# TODO: human review required
class DJTL(AbstractUnconstrainedMinimisation):
    """DJTL optimization problem.

    This is a 2-dimensional nonlinear optimization problem derived from
    a modification of problem 19 in the Hock and Schittkowski collection.
    It is meant to simulate a Lagrangian barrier objective function
    for particular values of shifts and multipliers.

    The problem includes cubic and quadratic terms with logarithmic barrier functions,
    making it a challenging optimization problem.

    Source: modified version of problem 19 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    SIF input: A.R. Conn August 1993

    Classification: OUR2-AN-2-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 2  # Number of variables

    def objective(self, y, args):
        del args
        x1, x2 = y

        # Base cubic terms
        f = (x1 - 10.0) ** 3 + (x2 - 20.0) ** 3

        # Barrier/penalty terms from AMPL model
        # Each term has the form: if g(x)+1 <= 0 then 1e10*g(x)^2 else -log(g(x)+1)

        # Term 1: -(x1-5)^2-(x2-5)^2+200
        g1 = -((x1 - 5.0) ** 2) - (x2 - 5.0) ** 2 + 200.0
        f = f + jnp.where(g1 + 1.0 <= 0.0, 1e10 * g1**2, -jnp.log(g1 + 1.0))

        # Term 2: (x1-5)^2+(x2-5)^2-100
        g2 = (x1 - 5.0) ** 2 + (x2 - 5.0) ** 2 - 100.0
        f = f + jnp.where(g2 + 1.0 <= 0.0, 1e10 * g2**2, -jnp.log(g2 + 1.0))

        # Term 3: (x2-5)^2+(x1-6)^2
        g3 = (x2 - 5.0) ** 2 + (x1 - 6.0) ** 2
        f = f + jnp.where(g3 + 1.0 <= 0.0, 1e10 * g3**2, -jnp.log(g3 + 1.0))

        # Term 4: -(x2-5)^2-(x1-6)^2+82.81
        g4 = -((x2 - 5.0) ** 2) - (x1 - 6.0) ** 2 + 82.81
        f = f + jnp.where(g4 + 1.0 <= 0.0, 1e10 * g4**2, -jnp.log(g4 + 1.0))

        # Term 5: 100-x1
        g5 = 100.0 - x1
        f = f + jnp.where(g5 + 1.0 <= 0.0, 1e10 * g5**2, -jnp.log(g5 + 1.0))

        # Term 6: x1-13
        g6 = x1 - 13.0
        f = f + jnp.where(g6 + 1.0 <= 0.0, 1e10 * g6**2, -jnp.log(g6 + 1.0))

        # Term 7: 100-x2
        g7 = 100.0 - x2
        f = f + jnp.where(g7 + 1.0 <= 0.0, 1e10 * g7**2, -jnp.log(g7 + 1.0))

        # Term 8: x2
        g8 = x2
        f = f + jnp.where(g8 + 1.0 <= 0.0, 1e10 * g8**2, -jnp.log(g8 + 1.0))

        return f

    def y0(self):
        # Initial values from SIF file
        return jnp.array([15.0, 6.0])

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution is not explicitly given in the SIF file
        return None

    def expected_objective_value(self):
        # According to the SIF file, the solution value is approximately -8951.54472
        # But this needs human verification
        return None
