import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class BT1(AbstractConstrainedMinimisation):
    """BT1 - Boggs-Tolle test problem 1.

    n = 2, m = 1.
    f(x) = -x₁ + 10(x₁² + x₂² - 1).
    g(x) = x₁² + x₂² - 1.

    Start: x₁ = 0.08, x₂ = 0.06.
    Solution: x* = (1.0, 0.0).

    Source: Boggs, P.T. and Tolle, J.W.,
    "A strategy for global convergence in a sequential
    quadratic programming algorithm",
    SIAM J. Numer. Anal. 26(3), pp. 600-623, 1989.

    Classification: SQR2-AN-2-1
    """

    def objective(self, y, args):
        del args
        x1, x2 = y
        return -x1 + 10 * (x1**2 + x2**2 - 1)

    def y0(self):
        return jnp.array([0.08, 0.06])

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([1.0, 0.0])

    def expected_objective_value(self):
        return jnp.array(-1.0)

    def bounds(self):
        return None

    def constraint(self, y):
        x1, x2 = y
        # Equality constraint: x₁² + x₂² - 1 = 0
        equality_constraint = x1**2 + x2**2 - 1
        return equality_constraint, None


class BT2(AbstractConstrainedMinimisation):
    """BT2 - Boggs-Tolle test problem 2.

    n = 3, m = 1.
    f(x) = (x₁ - 1)² + (x₁ - x₂)² + (x₂ - x₃)⁴.
    g(x) = x₁(1 + x₂²) + x₃⁴ - 4 - 3√2.

    Start 1: xᵢ = 1, i = 1, 2, 3.
    Start 2: xᵢ = 10, i = 1, 2, 3.
    Start 3: xᵢ = 100, i = 1, 2, 3.
    Solution: x* = (1.1049, 1.1967, 1.5353).

    Source: Boggs, P.T. and Tolle, J.W.,
    "A strategy for global convergence in a sequential
    quadratic programming algorithm",
    SIAM J. Numer. Anal. 26(3), pp. 600-623, 1989.

    Classification: SQR2-AN-3-1
    """

    y0_id: int = 0
    provided_y0s: frozenset = frozenset({0, 1, 2})

    def objective(self, y, args):
        del args
        x1, x2, x3 = y
        return (x1 - 1) ** 2 + (x1 - x2) ** 2 + (x2 - x3) ** 4

    def y0(self):
        if self.y0_id == 0:
            return jnp.array([1.0, 1.0, 1.0])
        elif self.y0_id == 1:
            return jnp.array([10.0, 10.0, 10.0])
        elif self.y0_id == 2:
            return jnp.array([100.0, 100.0, 100.0])

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([1.1049, 1.1967, 1.5353])

    def expected_objective_value(self):
        return None  # Not explicitly given

    def bounds(self):
        return None

    def constraint(self, y):
        x1, x2, x3 = y
        # Equality constraint: x₁(1 + x₂²) + x₃⁴ - 4 - 3√2 = 0
        equality_constraint = x1 * (1 + x2**2) + x3**4 - 4 - 3 * jnp.sqrt(2)
        return equality_constraint, None


class BT3(AbstractConstrainedMinimisation):
    """BT3 - Boggs-Tolle test problem 3.

    n = 5, m = 3.
    f(x) = (x₁ - x₂)² + (x₂ + x₃ - 2)² + (x₄ - 1)² + (x₅ - 1)².
    g₁(x) = x₁ + 3x₂.
    g₂(x) = x₃ + x₄ - 2x₅.
    g₃(x) = x₂ - x₅.

    Start 1: xᵢ = 2, i = 1, ..., 5.
    Start 2: xᵢ = 20, i = 1, ..., 5.
    Start 3: xᵢ = 200, i = 1, ..., 5.
    Solution: x* = (-0.76744, 0.25581, 0.62791, -0.11628, 0.25581).

    Source: Boggs, P.T. and Tolle, J.W.,
    "A strategy for global convergence in a sequential
    quadratic programming algorithm",
    SIAM J. Numer. Anal. 26(3), pp. 600-623, 1989.

    Classification: SQR2-AN-5-3
    """

    y0_id: int = 0
    provided_y0s: frozenset = frozenset({0, 1, 2})

    def objective(self, y, args):
        del args
        x1, x2, x3, x4, x5 = y
        return (x1 - x2) ** 2 + (x2 + x3 - 2) ** 2 + (x4 - 1) ** 2 + (x5 - 1) ** 2

    def y0(self):
        if self.y0_id == 0:
            return jnp.array([2.0, 2.0, 2.0, 2.0, 2.0])
        elif self.y0_id == 1:
            return jnp.array([20.0, 20.0, 20.0, 20.0, 20.0])
        elif self.y0_id == 2:
            return jnp.array([200.0, 200.0, 200.0, 200.0, 200.0])

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([-0.76744, 0.25581, 0.62791, -0.11628, 0.25581])

    def expected_objective_value(self):
        return None  # Not explicitly given

    def bounds(self):
        return None

    def constraint(self, y):
        x1, x2, x3, x4, x5 = y
        # Equality constraints
        g1 = x1 + 3 * x2
        g2 = x3 + x4 - 2 * x5
        g3 = x2 - x5
        equality_constraints = jnp.array([g1, g2, g3])
        return equality_constraints, None


class BT4(AbstractConstrainedMinimisation):
    """BT4 - Boggs-Tolle test problem 4.

    n = 3, m = 2.
    f(x) = x₁ - x₂ + x₃².
    g₁(x) = x₁² + x₂² + x₃² - 25.
    g₂(x) = x₁ + x₂ + x₃ - 1.

    Start 1: x₁ = 3.1494, x₂ = 1.4523, x₃ = -3.6017.
    Start 2: x₁ = 3.122, x₂ = 1.489, x₃ = -3.611.
    Start 3: x₁ = -0.94562, x₂ = -2.35984, x₃ = 4.30546.
    Solution: x* = (4.0382, -2.9470, -0.09115).

    Source: Boggs, P.T. and Tolle, J.W.,
    "A strategy for global convergence in a sequential
    quadratic programming algorithm",
    SIAM J. Numer. Anal. 26(3), pp. 600-623, 1989.

    Classification: SQR2-AN-3-2
    """

    y0_id: int = 0
    provided_y0s: frozenset = frozenset({0, 1, 2})

    def objective(self, y, args):
        del args
        x1, x2, x3 = y
        return x1 - x2 + x3**2

    def y0(self):
        if self.y0_id == 0:
            return jnp.array([3.1494, 1.4523, -3.6017])
        elif self.y0_id == 1:
            return jnp.array([3.122, 1.489, -3.611])
        elif self.y0_id == 2:
            return jnp.array([-0.94562, -2.35984, 4.30546])

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([4.0382, -2.9470, -0.09115])

    def expected_objective_value(self):
        return None  # Not explicitly given

    def bounds(self):
        return None

    def constraint(self, y):
        x1, x2, x3 = y
        # Equality constraints
        g1 = x1**2 + x2**2 + x3**2 - 25
        g2 = x1 + x2 + x3 - 1
        equality_constraints = jnp.array([g1, g2])
        return equality_constraints, None


class BT5(AbstractConstrainedMinimisation):
    """BT5 - Boggs-Tolle test problem 5.

    n = 3, m = 2.
    f(x) = 1000 - x₁² - 2x₃² - x₂² - x₁x₂ - x₁x₃.
    g₁(x) = x₁² + x₂² - x₃² - 25.
    g₂ = 8x₁ + 14x₂ + 7x₃ - 56.

    Start 1: xᵢ = 2, i = 1, 2, 3.
    Start 2: xᵢ = 20, i = 1, 2, 3.
    Start 3: xᵢ = 80, i = 1, 2, 3.
    Solution: x* = (3.5121, 0.21699, 3.5522).

    Source: Boggs, P.T. and Tolle, J.W.,
    "A strategy for global convergence in a sequential
    quadratic programming algorithm",
    SIAM J. Numer. Anal. 26(3), pp. 600-623, 1989.

    Classification: SQR2-AN-3-2
    """

    y0_id: int = 0
    provided_y0s: frozenset = frozenset({0, 1, 2})

    def objective(self, y, args):
        del args
        x1, x2, x3 = y
        return 1000 - x1**2 - 2 * x3**2 - x2**2 - x1 * x2 - x1 * x3

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
        g1 = x1**2 + x2**2 - x3**2 - 25
        g2 = 8 * x1 + 14 * x2 + 7 * x3 - 56
        equality_constraints = jnp.array([g1, g2])
        return equality_constraints, None
