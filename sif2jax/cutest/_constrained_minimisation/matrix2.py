import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class MATRIX2(AbstractConstrainedMinimisation):
    """MATRIX2 - Find closest pair of 2x2 symmetric matrices.

    Find the pair of 2x2 symmetric matrices X and Y that are closest in
    Frobenius norm, subject to X being positive semidefinite and
    Y being negative semidefinite.

    Variables: X11, X12, X22, Y11, Y12, Y22 (6 total)

    Objective: minimize ||X - Y||_F^2 = (X11-Y11)² + 2*(X12-Y12)² + (X22-Y22)²

    Constraints:
    - X positive semidefinite: X11*X22 - X12² ≥ 0, X11 ≥ 0, X22 ≥ 0
    - Y negative semidefinite: Y11*Y22 - Y12² ≥ 0, Y11 ≤ 0, Y22 ≤ 0

    Start: all variables = 1.0
    Solution: 0.0 (matrices can be made identical at the zero matrix)

    Source: A problem made up for the SIF/LANCELOT user manual.
    SIF input: Ph. Toint, Jan 91.

    Classification: QOR2-AY-6-2
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        del args
        x11, x12, x22, y11, y12, y22 = y

        # Frobenius norm squared: ||X - Y||_F^2
        # = (X11-Y11)² + 2*(X12-Y12)² + (X22-Y22)²
        diff11 = x11 - y11
        diff12 = x12 - y12
        diff22 = x22 - y22

        return diff11**2 + 2 * diff12**2 + diff22**2

    @property
    def y0(self):
        # Starting point: all variables = 1.0
        return jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        # Solution: matrices can be made identical at zero matrix
        return jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    @property
    def expected_objective_value(self):
        # Solution value: 0.0
        return jnp.array(0.0)

    @property
    def bounds(self):
        # Bounds from SIF file:
        # XR (free): X12, Y12
        # XM (≤ 0): Y11, Y22
        # Implicit: X11, X22 ≥ 0 (from positive semidefinite constraint)

        lower_bounds = jnp.array([0.0, -jnp.inf, 0.0, -jnp.inf, -jnp.inf, -jnp.inf])
        upper_bounds = jnp.array([jnp.inf, jnp.inf, jnp.inf, 0.0, jnp.inf, 0.0])

        return (lower_bounds, upper_bounds)

    def constraint(self, y):
        x11, x12, x22, y11, y12, y22 = y

        # Constraint that X is positive semidefinite: X11*X22 - X12² ≥ 0
        x_posdef = x11 * x22 - x12**2

        # Constraint that Y is negative semidefinite: Y11*Y22 - Y12² ≥ 0
        y_negdef = y11 * y22 - y12**2

        inequality_constraints = jnp.array([x_posdef, y_negdef])

        return None, inequality_constraints
