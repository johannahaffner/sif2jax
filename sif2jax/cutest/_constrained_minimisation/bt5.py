import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class BT5(AbstractConstrainedMinimisation):
    """BT5 - Boggs-Tolle test problem 5.

    n = 3, m = 2.
    f(x) = 1000 - x₁² - 2x₂² - x₃² - x₁x₂ - x₁x₃.
    g₁(x) = x₁² + x₂² + x₃² - 25.
    g₂ = 8x₁ + 14x₂ + 7x₃ - 56.

    Start 1: xᵢ = 2, i = 1, 2, 3.
    Start 2: xᵢ = 20, i = 1, 2, 3.
    Start 3: xᵢ = 80, i = 1, 2, 3.
    Solution: x* = (3.5121, 0.21699, 3.5522).

    Source: Boggs, P.T. and Tolle, J.W.,
    "A strategy for global convergence in a sequential
    quadratic programming algorithm",
    SIAM J. Numer. Anal. 26(3), pp. 600-623, 1989.

    Note: The problem as stated in the paper seems to contain a typo.
    The sign of the x₃² term in the first constraint has been
    changed from negative to positive to ensure that the problem is
    bounded below and the optimal point stated can be recovered.

    Classification: QQR2-AN-3-2
    Note: The SIF file has -1000 as the constant, but AMPL and pycutest implement +1000.
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    y0_id: int = 0
    provided_y0s: frozenset = frozenset({0, 1, 2})

    def objective(self, y, args):
        del args
        x1, x2, x3 = y
        return 1000 - x1**2 - 2 * x2**2 - x3**2 - x1 * x2 - x1 * x3

    def y0(self):
        if self.y0_id == 0:
            return jnp.array([2.0, 2.0, 2.0])
        elif self.y0_id == 1:
            return jnp.array([20.0, 20.0, 20.0])
        elif self.y0_id == 2:
            return jnp.array([80.0, 80.0, 80.0])

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([3.5121, 0.21699, 3.5522])

    def expected_objective_value(self):
        return None  # Not explicitly given

    def bounds(self):
        return None

    def constraint(self, y):
        x1, x2, x3 = y
        # Equality constraints
        g1 = x1**2 + x2**2 + x3**2 - 25  # Note: typo fixed in SIF - positive x3^2
        g2 = 8 * x1 + 14 * x2 + 7 * x3 - 56
        equality_constraints = jnp.array([g1, g2])
        return equality_constraints, None
