import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class MADSEN(AbstractConstrainedMinimisation):
    """MADSEN - A nonlinear minmax problem.

    n = 3, m = 6.

    Minimize U subject to:
    U ≥ x₁² + x₂² + x₁x₂
    U ≥ -(x₁² + x₂² + x₁x₂)
    U ≥ sin(x₁)
    U ≥ -sin(x₁)
    U ≥ cos(x₂)
    U ≥ -cos(x₂)

    Start: x₁ = 3.0, x₂ = 1.0, U = 1.0.

    Source: K. Madsen
    "An algorithm for minmax solution of overdetermined systems of non-linear
    equations"
    JIMA, vol.16, pp. 321-328, 1975.

    SIF input: Ph. Toint, April 1992.

    Classification: LOR2-AN-3-6
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        del args
        x1, x2, u = y
        return u

    @property
    def y0(self):
        return jnp.array([3.0, 1.0, 1.0])

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        return None

    @property
    def expected_objective_value(self):
        return None

    @property
    def bounds(self):
        return None

    def constraint(self, y):
        x1, x2, u = y

        # All constraints are inequality: g(x) ≥ 0
        # Convert minmax constraints U ≥ f(x) to f(x) - U ≤ 0, or -(f(x) - U) ≥ 0

        quad_expr = x1**2 + x2**2 + x1 * x2
        sin_x1 = jnp.sin(x1)
        cos_x2 = jnp.cos(x2)

        inequality_constraints = jnp.array(
            [
                u - quad_expr,  # U ≥ x₁² + x₂² + x₁x₂
                u + quad_expr,  # U ≥ -(x₁² + x₂² + x₁x₂)
                u - sin_x1,  # U ≥ sin(x₁)
                u + sin_x1,  # U ≥ -sin(x₁)
                u - cos_x2,  # U ≥ cos(x₂)
                u + cos_x2,  # U ≥ -cos(x₂)
            ]
        )

        return None, inequality_constraints
