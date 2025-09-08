"""DNIEPER problem implementation."""

import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractConstrainedMinimisation


# TODO: Human review needed - complex constraint formulation needs detailed SIF analysis
# Issues:
# 1. Objective function discrepancy: pycutest=-891.019272, sif2jax=-666.091272
# 2. Constraint values differ significantly (max diff ~11.22)
# 3. SIF structure is complex with multiple WJ/WK nonlinear elements
# 4. Variable indexing and element function mapping needs verification
# 5. Constraint constant terms and linear coefficients need review
class DNIEPER(AbstractConstrainedMinimisation):
    """DNIEPER - Water resource planning in the Dnieper river basin.

    This problem models the planning of systematic use of water resources
    in the basin of the river Dnieper.

    Source: p. 139sq in
    B.N. Pshenichnyj
    "The Linearization Method for Constrained Optimization",
    Springer Verlag, SCM Series 22, Heidelberg, 1994

    SIF input: Ph. Toint, December 1994.

    Classification: QOR2-MN-61-24
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n_var(self) -> int:
        """Number of variables: 56 X(I) + 4 fixed variables + 1 AC = 61."""
        return jnp.array(61)

    @property
    def n_con(self) -> int:
        """Number of constraints."""
        return jnp.array(24)

    def objective(self, y: Array, args) -> Array:
        """Objective function with quadratic, cubic and product terms."""
        del args

        # Extract variables: X(1)..X(56), X0F, X24F, X12F, X36F, AC
        x = y[:56]  # X(1) to X(56)
        # x0f = y[56]  # X0F - unused
        # x24f = y[57]  # X24F - unused
        # x12f = y[58]  # X12F - unused
        # x36f = y[59]  # X36F - unused
        ac = y[60]  # AC

        obj = -112.464  # Constant term from CONSTANTS section

        # Linear terms from GROUPS section
        # DO I 1 12: I0=I+12, I1=I+24, I2=I+36
        for i in range(12):
            i_idx = i  # X(I) - 0-based indexing
            i0_idx = i + 12  # X(I0) = X(I+12) - indices 12-23
            i1_idx = i + 24  # X(I1) = X(I+24) - indices 24-35
            i2_idx = i + 36  # X(I2) = X(I+36) - indices 36-47

            # From line 96-97: XN OBJ X(I1) 19.95 X(I) 0.07656
            obj += 19.95 * x[i1_idx] + 0.07656 * x[i_idx]
            # XN OBJ X(I2) -24.89 X(I0) -0.7135
            obj += -24.89 * x[i2_idx] + -0.7135 * x[i0_idx]

        # Nonlinear elements from ELEMENT USES and GROUP USES
        # E(I) elements: 2PR type (product of X(I1) * X(I2))
        for i in range(12):
            i1_idx = i + 12  # X(I+12) from ELEMENT USES
            i2_idx = i + 36  # X(I+36) from ELEMENT USES
            obj += 2.155 * x[i1_idx] * x[i2_idx]

        # ACSQ element: SQ type (AC squared)
        obj += -2000.0 * ac * ac

        # Scale by -1 for maximization (line 102)
        return -obj

    def constraint(self, y: Array) -> tuple[Array, None]:
        """Constraint functions with nonlinear elements."""
        # Extract variables
        x = y[:56]  # X(1) to X(56)
        x0f = y[56]  # X0F
        x24f = y[57]  # X24F
        x12f = y[58]  # X12F
        x36f = y[59]  # X36F
        ac = y[60]  # AC

        # Coefficients from problem data (C1 to C24)
        c = jnp.array(
            [
                5.61,
                4.68,
                1.62,
                1.8,
                2.13,
                2.1,
                1.99,
                2.02,
                2.14,
                2.15,
                2.36,
                2.63,
                -0.02,
                -0.01,
                -0.16,
                -0.47,
                -0.75,
                -0.94,
                -0.93,
                -0.99,
                -0.42,
                -0.07,
                0.04,
                -0.06,
            ]
        )

        constraints = jnp.zeros(24)

        # CC(I) constraints: linear terms + W1(I) - W2(I) + constants
        # From CONSTANTS: CC(I) gets constant -C(I)
        for i in range(24):
            constraint_val = -c[i]  # Constant from CONSTANTS section

            # Linear terms from GROUPS section
            if i < 4:  # CC(1) to CC(4): DO I 1 4, I0 = I + 24
                i0_idx = i + 24  # I0 = I + 24
                constraint_val += -2.68 * x[i0_idx]

            elif i < 8:  # CC(5) to CC(8): DO I 5 8
                i0_idx = i + 24  # I0 = I + 24
                i1_idx = i + 44  # I1 = I + 44
                constraint_val += -2.68 * x[i0_idx] + -2.68 * x[i1_idx]

            elif i < 12:  # CC(9) to CC(12): DO I 9 12
                i0_idx = i + 24  # I0 = I + 24
                constraint_val += -2.68 * x[i0_idx]

            elif i < 16:  # CC(13) to CC(16): DO I 13 16
                i0_idx = i + 12  # I0 = I + 12
                i1_idx = i + 24  # I1 = I + 24
                constraint_val += -2.68 * x[i0_idx] + -2.68 * x[i1_idx] + -1.0 * ac

            elif i < 20:  # CC(17) to CC(20): DO I 17 20
                i0_idx = i + 12  # I0 = I + 12
                i1_idx = i + 24  # I1 = I + 24
                i2_idx = i + 36  # I2 = I + 36
                constraint_val += (
                    -2.68 * x[i0_idx]
                    + -2.68 * x[i1_idx]
                    + -2.68 * x[i2_idx]
                    + -1.0 * ac
                )

            else:  # CC(21) to CC(24): DO I 21 24
                i0_idx = i + 12  # I0 = I + 12
                i1_idx = i + 24  # I1 = I + 24
                constraint_val += -2.68 * x[i0_idx] + -2.68 * x[i1_idx] + -1.0 * ac

            # Nonlinear elements W1(I) and W2(I)
            # From GROUP USES: CC(I) = linear + W1(I) - W2(I)

            # W1(I) elements (I = 1 to 24)
            if i < 12:  # W1(1) to W1(12): WJ elements
                w1_val = self._wj_element(x[i], x[i + 24])
            else:  # W1(13) to W1(24): WK elements
                w1_val = self._wk_element(x[i], x[i + 24])

            # W2(I) elements (I = 1 to 24)
            if i == 0:  # W2(1): WJ element with X0F, X24F
                w2_val = self._wj_element(x0f, x24f)
            elif i < 12:  # W2(2) to W2(12): WJ elements
                w2_val = self._wj_element(x[i - 1], x[i + 23])
            elif i == 12:  # W2(13): WK element with X12F, X36F
                w2_val = self._wk_element(x12f, x36f)
            else:  # W2(14) to W2(24): WK elements
                w2_val = self._wk_element(x[i - 1], x[i + 23])

            constraint_val += w1_val - w2_val
            constraints = constraints.at[i].set(constraint_val)

        return constraints, None

    def _wj_element(self, x_val: Array, y_val: Array) -> Array:
        """WJ element function with cubic polynomial."""
        # Coefficients from ELEMENTS section
        a1 = 34.547
        a2 = -0.55878
        a3 = 8.05339
        a4 = -0.02252
        a5 = -0.29316
        a6 = -0.013521
        a7 = 0.00042
        a8 = 0.00267
        a9 = 0.000281
        a10 = 0.0000032

        result = (
            a1
            + a2 * x_val
            + a3 * y_val
            + a4 * x_val**2
            + a5 * x_val * y_val
            + a6 * y_val**2
            + a7 * x_val**3
            + a8 * x_val**2 * y_val
            + a9 * x_val * y_val**2
            + a10 * y_val**3
        )

        return result

    def _wk_element(self, x_val: Array, y_val: Array) -> Array:
        """WK element function with cubic polynomial."""
        # Coefficients from ELEMENTS section
        a1 = 20.923
        a2 = -4.22088
        a3 = 1.42061
        a4 = -0.41040
        a5 = -0.15082
        a7 = -0.00826
        a8 = 0.00404
        a9 = 0.000168
        a10 = -0.000038

        result = (
            a1
            + a2 * x_val
            + a3 * y_val
            + a4 * x_val**2
            + a5 * x_val * y_val
            + a7 * x_val**3
            + a8 * x_val**2 * y_val
            + a9 * x_val * y_val**2
            + a10 * y_val**3
        )

        return result

    @property
    def y0(self) -> Array:
        """Starting point from START POINT section."""
        y0 = jnp.zeros(61)

        # X(1) to X(12): 51.35
        y0 = y0.at[:12].set(51.35)

        # X(13) to X(24): 15.5
        y0 = y0.at[12:24].set(15.5)

        # X(25) to X(36): 2.5
        y0 = y0.at[24:36].set(2.5)

        # X(37) to X(48): 2.6
        y0 = y0.at[36:48].set(2.6)

        # X(49) to X(56): 0.3
        y0 = y0.at[48:56].set(0.3)

        # Fixed variables
        y0 = y0.at[56].set(50.82)  # X0F
        y0 = y0.at[57].set(2.0)  # X24F
        y0 = y0.at[58].set(15.5)  # X12F
        y0 = y0.at[59].set(2.3)  # X36F
        y0 = y0.at[60].set(0.0)  # AC (initial guess)

        return y0

    @property
    def bounds(self) -> tuple[Array, Array]:
        """Variable bounds from BOUNDS section."""
        lower = jnp.full(61, -jnp.inf)
        upper = jnp.full(61, jnp.inf)

        # X(1) to X(12): [51.2, 51.4]
        lower = lower.at[:12].set(51.2)
        upper = upper.at[:12].set(51.4)

        # X(13) to X(24): [15.0, 16.1]
        lower = lower.at[12:24].set(15.0)
        upper = upper.at[12:24].set(16.1)

        # X(25) to X(36): [0.4, 4.6]
        lower = lower.at[24:36].set(0.4)
        upper = upper.at[24:36].set(4.6)

        # X(37) to X(48): [0.5, 4.8]
        lower = lower.at[36:48].set(0.5)
        upper = upper.at[36:48].set(4.8)

        # X(49) to X(56): [0.0, 0.7]
        lower = lower.at[48:56].set(0.0)
        upper = upper.at[48:56].set(0.7)

        # Fixed variables
        lower = lower.at[56].set(50.82)  # X0F = 50.82
        upper = upper.at[56].set(50.82)

        lower = lower.at[57].set(2.0)  # X24F = 2.0
        upper = upper.at[57].set(2.0)

        lower = lower.at[58].set(15.5)  # X12F = 15.5
        upper = upper.at[58].set(15.5)

        lower = lower.at[59].set(2.3)  # X36F = 2.3
        upper = upper.at[59].set(2.3)

        # AC is free (FR)
        # lower[60] and upper[60] remain -inf and +inf

        return lower, upper

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def expected_result(self) -> Array:
        """Expected solution from SIF file comments."""
        solution = jnp.array(
            [
                51.4,
                51.2,
                51.4,
                51.2,
                51.4,
                51.4,
                51.4,
                51.4,
                51.4,
                51.2,
                51.4,
                51.2,
                15.3142,
                15.1879,
                15.1846,
                15.165,
                15.1615,
                15.1495,
                15.138,
                15.1238,
                15.0864,
                15.069,
                15.0389,
                15.0,
                2.18087,
                1.7078,
                0.618689,
                0.639532,
                0.4,
                0.4,
                0.4,
                0.4,
                0.804561,
                0.76947,
                0.915035,
                0.949886,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.423624,
                0.383582,
                0.342537,
                0.353731,
                0.0,
                0.0,
                0.0,
                0.0,
                50.82,
                2.0,
                15.5,
                2.3,
                -3.08411,
            ]
        )
        return solution

    @property
    def expected_objective_value(self) -> Array:
        """Expected objective value from SIF file comments."""
        return jnp.array(18743.9)  # From OBJECT BOUND comment

    def num_constraints(self) -> tuple[int, int, int]:
        """Returns the number of constraints and bounds."""
        # Count finite bounds
        lower, upper = self.bounds
        num_finite_lower = jnp.sum(jnp.isfinite(lower))
        num_finite_upper = jnp.sum(jnp.isfinite(upper))
        num_finite_bounds = int(num_finite_lower + num_finite_upper)

        # All 24 constraints are equalities
        return self.n_con, 0, num_finite_bounds
