import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class FLETCHER(AbstractConstrainedMinimisation):
    """FLETCHER - Minimal area right-angled triangle containing unit circle.

    Find the least area of a right-angled triangle which contains a circle
    of unit radius. This is a classic geometric optimization problem from
    Fletcher's textbook.

    Variables: X1, X2, X3, X4
    Objective: minimize X1 * X2 (area of triangle)
    Constraints:
        C1: SPEC(X1,X2,X3,X4) - X3² - X4² = -1 (unit circle containment)
        C2: X1 - X3 ≥ 1 (geometric constraint)
        C3: X2 - X4 ≥ 1 (geometric constraint)
        C4: X3 - X4 ≥ 0 (geometric constraint)
    Bounds: X4 ≥ 1, others free

    Where SPEC(X1,X2,X3,X4) = (X1*X3 + X2*X4)² / (X1² + X2²)

    Source: R. Fletcher, "Practical Methods of Optimization",
    second edition, Wiley, 1987.

    SIF input: Ph. Toint, March 1994.

    Classification: QOR2-AN-4-4
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def y0(self):
        # Starting point from SIF: all variables = 1.0
        return jnp.array([1.0, 1.0, 1.0, 1.0])

    @property
    def args(self):
        return None

    def objective(self, y, args):
        del args
        x1, x2, x3, x4 = y

        # Objective: minimize X1 * X2 (area of right-angled triangle)
        # From SIF: OBJ uses 2PR element type with XX=X1, YY=X2
        return x1 * x2

    def constraint(self, y):
        x1, x2, x3, x4 = y

        # Helper function for SPEC element
        def spec_element(xx, yy, vv, ww):
            """SPEC element: (XX*VV + YY*WW)² / (XX² + YY²)"""
            numerator = xx * vv + yy * ww
            denominator = xx**2 + yy**2
            return (numerator**2) / denominator

        # Handle SPEC element with division by zero protection
        def safe_spec_element(xx, yy, vv, ww):
            numerator = xx * vv + yy * ww
            denominator = xx**2 + yy**2
            # Add small epsilon to prevent division by zero
            eps = 1e-12
            return (numerator**2) / (denominator + eps)

        # C1: SPEC(X1,X2,X3,X4) - X3² - X4² = -1 (equality)
        spec_value = safe_spec_element(x1, x2, x3, x4)
        c1 = spec_value - x3**2 - x4**2 + 1.0  # +1.0 because RHS is -1.0

        eq_constraints = jnp.array([c1])

        # From SIF analysis:
        # C2: X1 - X3 - 1 = 0 → X1 - X3 - 1 ≥ 0
        # C3: X2 - X4 - 1 = 0 → X2 - X4 - 1 ≥ 0
        # C4: X3 - X4 = 0 → X3 - X4 ≥ 0
        ineq_constraints = jnp.array(
            [
                x1 - x3 - 1.0,  # C2: X1 - X3 - 1 ≥ 0
                x2 - x4 - 1.0,  # C3: X2 - X4 - 1 ≥ 0
                x3 - x4,  # C4: X3 - X4 ≥ 0
            ]
        )

        return eq_constraints, ineq_constraints

    @property
    def bounds(self):
        # From SIF: X4 ≥ 1.0, others free
        lower = jnp.array([-jnp.inf, -jnp.inf, -jnp.inf, 1.0])
        upper = jnp.array([jnp.inf, jnp.inf, jnp.inf, jnp.inf])
        return lower, upper

    @property
    def expected_result(self):
        # Analytical solution for minimal area triangle containing unit circle
        # The optimal triangle has area = 2 + 2*sqrt(2) ≈ 4.828
        # This occurs when the triangle sides are tangent to the circle
        sqrt2 = jnp.sqrt(2.0)
        opt_val = 1.0 + sqrt2  # ≈ 2.414
        return jnp.array([opt_val, opt_val, sqrt2, sqrt2])

    @property
    def expected_objective_value(self):
        # Minimal area = (1 + √2)² = 1 + 2√2 + 2 = 3 + 2√2 ≈ 5.828
        sqrt2 = jnp.sqrt(2.0)
        return jnp.array(3.0 + 2.0 * sqrt2)
