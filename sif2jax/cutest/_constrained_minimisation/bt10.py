import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class BT10(AbstractConstrainedMinimisation):
    """BT10 - Boggs-Tolle test problem 10.

    n = 2, m = 2.
    f(x) = -x₁.
    g₁(x) = x₂ - x₁³.
    g₂(x) = x₁² - x₂.

    Start 1: xᵢ = 2, i = 1, 2.
    Start 2: xᵢ = 20, i = 1, 2.
    Start 3: xᵢ = 50, i = 1, 2.
    Solution: x* = (1.0, 1.0).

    Source: Boggs, P.T. and Tolle, J.W.,
    "A strategy for global convergence in a sequential
    quadratic programming algorithm",
    SIAM J. Numer. Anal. 26(3), pp. 600-623, 1989.

    Note: The problem as stated in the paper seems to contain a typo.
    In order to make the second constraint feasible at the proposed
    solution, the sign of x₂ in the second constraint has been
    changed from + to -.

    Classification: LOR2-AN-2-2
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    y0_id: int = 0
    provided_y0s: frozenset = frozenset({0, 1, 2})

    def objective(self, y, args):
        del args
        x1, x2 = y
        return -x1

    def y0(self):
        if self.y0_id == 0:
            return jnp.array([2.0, 2.0])
        elif self.y0_id == 1:
            return jnp.array([20.0, 20.0])
        elif self.y0_id == 2:
            return jnp.array([50.0, 50.0])

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([1.0, 1.0])

    def expected_objective_value(self):
        return jnp.array(-1.0)

    def bounds(self):
        return None

    def constraint(self, y):
        x1, x2 = y
        # Equality constraints
        g1 = x2 - x1**3
        g2 = x1**2 - x2  # Note: sign of x2 changed from + to - per SIF file
        equality_constraints = jnp.array([g1, g2])
        return equality_constraints, None
