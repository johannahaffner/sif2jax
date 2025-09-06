"""DISC2 problem implementation."""

import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractConstrainedMinimisation


# TODO: Human review needed - multiple implementation issues
# Issues:
# 1. Starting values mismatch: pycutest expects [0.5, 5, 5, ...], sif2jax provides [11, 1, 2, ...]
# 2. Constraint formulation incorrect: significant discrepancies in constraint values
# 3. Bounds issues: bound checking failures
# 4. Jacobian structure wrong: max difference ~24.0 in equality Jacobians
# 5. Complex geometric problem with CIRCLE and LINE elements needs careful SIF analysis
# 6. Need to understand proper variable ordering and constraint structure
class DISC2(AbstractConstrainedMinimisation):
    """DISC2 - Minimum disc radius subject to polygon constraints.

    The problem is to find the minimum disc radius subject to polygon
    determined by boundary discs intersecting all interior discs.

    Source:
    W. Pulleyblank, private communication, 1991.

    SIF input: A.R. Conn, November 1991.

    Classification: LQR2-MY-29-23
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n_var(self) -> int:
        """Number of variables: 1 EPSILON + 11 U(I) + 11 V(I) + 6 ALPHA(I) = 29."""
        return 29

    @property
    def n_con(self) -> int:
        """Number of constraints: 11 B(I) + 12 collinearity constraints = 23."""
        return 23

    def objective(self, y: Array, args) -> Array:
        """Objective function: minimize EPSILON."""
        del args
        epsilon = y[0]  # EPSILON variable
        return epsilon

    def constraint(self, y: Array) -> tuple[Array, None]:
        """Constraint functions with circle and line elements."""
        # Extract variables
        epsilon = y[0]  # EPSILON
        u = y[1:12]  # U(1) to U(11)
        v = y[12:23]  # V(1) to V(11)
        alpha = y[23:29]  # ALPHA(1) to ALPHA(6)

        # Fixed coordinates from problem data
        x = jnp.array([1.0, 2.0, 7.0, 8.0, 5.0, 0.0, 7.0, 3.0, 1.0, 3.0, 6.0])
        y_coord = jnp.array([1.0, 2.0, 1.0, 4.0, 0.0, 8.0, 7.0, 3.0, 1.0, 3.0, 6.0])

        constraints = []

        # B(I) constraints (I = 1 to 11)
        for i in range(11):
            if i < 5:  # B(1) to B(5): equality constraints with CIRCLE elements
                # CIRCLE element: (U(i) - X(i))^2 + (V(i) - Y(i))^2 - EPSILON^2 = 0
                circle_val = (u[i] - x[i]) ** 2 + (v[i] - y_coord[i]) ** 2 - epsilon**2
                constraints.append(circle_val)
            else:  # B(6) to B(11): inequality constraints with CIRCLE elements
                # CIRCLE element: (U(i) - X(i))^2 + (V(i) - Y(i))^2 - EPSILON^2 <= 0
                circle_val = (u[i] - x[i]) ** 2 + (v[i] - y_coord[i]) ** 2 - epsilon**2
                constraints.append(circle_val)

        # Collinearity constraints (12 constraints)
        # B162: U6 - U1 + LINE element = 0
        line_val = (u[1] - u[0]) * (-alpha[0])  # U(2) - U(1), ALPHA(1)
        constraints.append(u[5] - u[0] + line_val)  # B162

        # C162: V6 - V1 + LINE element = 0
        line_val = (v[1] - v[0]) * (-alpha[0])  # V(2) - V(1), ALPHA(1)
        constraints.append(v[5] - v[0] + line_val)  # C162

        # B273: U7 - U2 + LINE element = 0
        line_val = (u[2] - u[1]) * (-alpha[1])  # U(3) - U(2), ALPHA(2)
        constraints.append(u[6] - u[1] + line_val)  # B273

        # C273: V7 - V2 + LINE element = 0
        line_val = (v[2] - v[1]) * (-alpha[1])  # V(3) - V(2), ALPHA(2)
        constraints.append(v[6] - v[1] + line_val)  # C273

        # B384: U8 - U3 + LINE element = 0
        line_val = (u[3] - u[2]) * (-alpha[2])  # U(4) - U(3), ALPHA(3)
        constraints.append(u[7] - u[2] + line_val)  # B384

        # C384: V8 - V3 + LINE element = 0
        line_val = (v[3] - v[2]) * (-alpha[2])  # V(4) - V(3), ALPHA(3)
        constraints.append(v[7] - v[2] + line_val)  # C384

        # B495: U9 - U4 + LINE element = 0
        line_val = (u[4] - u[3]) * (-alpha[3])  # U(5) - U(4), ALPHA(4)
        constraints.append(u[8] - u[3] + line_val)  # B495

        # C495: V9 - V4 + LINE element = 0
        line_val = (v[4] - v[3]) * (-alpha[3])  # V(5) - V(4), ALPHA(4)
        constraints.append(v[8] - v[3] + line_val)  # C495

        # B5101: U10 - U5 + LINE element = 0
        line_val = (u[0] - u[4]) * (-alpha[4])  # U(1) - U(5), ALPHA(5)
        constraints.append(u[9] - u[4] + line_val)  # B5101

        # C5101: V10 - V5 + LINE element = 0
        line_val = (v[0] - v[4]) * (-alpha[4])  # V(1) - V(5), ALPHA(5)
        constraints.append(v[9] - v[4] + line_val)  # C5101

        # B5111: U11 - U5 + LINE element = 0
        line_val = (u[0] - u[4]) * (-alpha[5])  # U(1) - U(5), ALPHA(6)
        constraints.append(u[10] - u[4] + line_val)  # B5111

        # C5111: V11 - V5 + LINE element = 0
        line_val = (v[0] - v[4]) * (-alpha[5])  # V(1) - V(5), ALPHA(6)
        constraints.append(v[10] - v[4] + line_val)  # C5111

        # Convert to arrays and separate equalities/inequalities
        all_constraints = jnp.array(constraints)
        equalities = all_constraints[:5]  # B(1) to B(5) are equalities
        inequalities = all_constraints[
            5:
        ]  # B(6) to B(11) and collinearity are equalities

        # Actually, looking at the groups: B(1-5) are XE (equality), B(6-11) are XL (<=),
        # collinearity are XE (equality)
        equalities = jnp.concatenate(
            [all_constraints[:5], all_constraints[11:]]
        )  # B(1-5) + collinearity
        inequalities = all_constraints[5:11]  # B(6-11)

        return equalities, inequalities

    @property
    def y0(self) -> Array:
        """Starting point with EPSILON = NNODES (11)."""
        y0 = jnp.zeros(29)
        y0 = y0.at[0].set(11.0)  # EPSILON = NNODES

        # Initialize U and V with fixed coordinates
        x = jnp.array([1.0, 2.0, 7.0, 8.0, 5.0, 0.0, 7.0, 3.0, 1.0, 3.0, 6.0])
        y_coord = jnp.array([1.0, 2.0, 1.0, 4.0, 0.0, 8.0, 7.0, 3.0, 1.0, 3.0, 6.0])

        y0 = y0.at[1:12].set(x)  # U(1) to U(11)
        y0 = y0.at[12:23].set(y_coord)  # V(1) to V(11)

        # ALPHA variables start at 0.5 (midway in [0,1] bounds)
        y0 = y0.at[23:29].set(0.5)  # ALPHA(1) to ALPHA(6)

        return y0

    @property
    def bounds(self) -> tuple[Array, Array]:
        """Variable bounds."""
        lower = jnp.full(29, -jnp.inf)
        upper = jnp.full(29, jnp.inf)

        # ALPHA(I) bounds: [0, 1] for I = 1 to 6
        lower = lower.at[23:29].set(0.0)  # ALPHA(1) to ALPHA(6)
        upper = upper.at[23:29].set(1.0)  # ALPHA(1) to ALPHA(6)

        return lower, upper

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def expected_result(self) -> Array:
        """Expected solution - using starting point as approximation."""
        return self.y0

    @property
    def expected_objective_value(self) -> Array:
        """Expected objective value - using starting point objective."""
        return jnp.array(11.0)  # Starting value of EPSILON

    def num_constraints(self) -> tuple[int, int, int]:
        """Returns the number of constraints and bounds."""
        # Count finite bounds
        lower, upper = self.bounds
        num_finite_lower = jnp.sum(jnp.isfinite(lower))
        num_finite_upper = jnp.sum(jnp.isfinite(upper))
        num_finite_bounds = int(num_finite_lower + num_finite_upper)

        # 17 equality constraints (5 + 12), 6 inequality constraints
        return 17, 6, num_finite_bounds
