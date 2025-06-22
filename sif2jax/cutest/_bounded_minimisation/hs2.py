import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class HS2(AbstractBoundedMinimisation):
    """Problem 2 from the Hock-Schittkowski test collection.

    The 2-variable Rosenbrock "banana valley" problem with a different bound.

    f(x) = 100(x₂ - x₁²)² + (1 - x₁)²

    Subject to: x₂ ≥ 1.5

    Source: problem 2 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    SIF input: A.R. Conn, March 1990.

    Classification: PBR-T1-2
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        x1, x2 = y
        return 100 * (x2 - x1**2) ** 2 + (1 - x1) ** 2

    def y0(self):
        return jnp.array([-2.0, 1.0])

    def args(self):
        return None

    def expected_result(self):
        # From the PDF: x* = (2a cos(1/3 arccos(1/b)), 1.5)
        # where a = (598/1200)^(1/2), b = 400 a³
        a = jnp.sqrt(598.0 / 1200.0)
        return jnp.array([2 * a * jnp.cos(jnp.arccos(1.0 / (400 * a**3)) / 3), 1.5])

    def expected_objective_value(self):
        return jnp.array(0.05042618790)

    def bounds(self):
        # Only x2 has a lower bound of 1.5, x1 is unbounded
        lower = jnp.array([-jnp.inf, 1.5])
        upper = jnp.array([jnp.inf, jnp.inf])
        return lower, upper
