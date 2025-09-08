"""DIXCHLNG problem implementation."""

import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractConstrainedMinimisation


class DIXCHLNG(AbstractConstrainedMinimisation):
    """DIXCHLNG - A constrained problem set as a challenge for SQP methods.

    A constrained problem set as a challenge for SQP methods
    by L.C.W. Dixon at the APMOD91 Conference.

    Source:
    L.C.W. Dixon, personal communication, Jan 1991.

    SIF input: Ph. Toint, Feb 1991.

    Classification: SOR2-AN-10-5
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n_var(self) -> int:
        """Number of variables: 10."""
        return 10

    @property
    def n_con(self) -> int:
        """Number of constraints: 5 (P(2), P(4), P(6), P(8), P(10))."""
        return 5

    def objective(self, y: Array, args) -> Array:
        """Objective function with squared and product terms."""
        del args

        # Constants from SIF
        scale_90 = 1.0 / 90.0
        scale_10_1 = 1.0 / 10.1
        scale_19_8 = 1.0 / 19.8

        obj = jnp.array(0.0)

        # Process 7 groups (i = 1 to 7, using 0-based indexing)
        for i in range(7):
            # A(i): X(i+1) - XSQ(i)
            # Using 0.01 scale factor from 'SCALE'
            a_val = 0.01 * (y[i + 1] - y[i] ** 2)
            obj += a_val**2  # L2 norm

            # B(i): X(i) with constant 1.0
            b_val = y[i] - 1.0
            obj += b_val**2  # L2 norm

            # C(i): X(i+3) - XSQ(i+2) with scale 1/90.0
            c_val = scale_90 * (y[i + 3] - y[i + 2] ** 2)
            obj += c_val**2  # L2 norm

            # D(i): X(i+2) with constant 1.0
            d_val = y[i + 2] - 1.0
            obj += d_val**2  # L2 norm

            # E(i): X(i+1) with scale 1/10.1 and constant 1.0
            e_val = scale_10_1 * (y[i + 1] - 1.0)
            obj += e_val**2  # L2 norm

            # F(i): X(i+3) with scale 1/10.1 and constant 1.0
            f_val = scale_10_1 * (y[i + 3] - 1.0)
            obj += f_val**2  # L2 norm

            # G(i): PR(i) = (X(i+1) - 1) * (X(i+3) - 1) with scale 1/19.8
            # S2PR element type
            g_val = scale_19_8 * ((y[i + 1] - 1.0) * (y[i + 3] - 1.0))
            obj += g_val**2  # L2 norm

        return obj

    def constraint(self, y: Array) -> tuple[Array, None]:
        """Constraint functions: products of variables."""
        constraints = []

        # P(2): Product of X(1) and X(2) = 1.0
        constraints.append(y[0] * y[1] - 1.0)

        # P(4): Product of X(1), X(2), X(3), X(4) = 1.0
        constraints.append(y[0] * y[1] * y[2] * y[3] - 1.0)

        # P(6): Product of X(1) through X(6) = 1.0
        prod6 = jnp.prod(y[:6])
        constraints.append(prod6 - 1.0)

        # P(8): Product of X(1) through X(8) = 1.0
        prod8 = jnp.prod(y[:8])
        constraints.append(prod8 - 1.0)

        # P(10): Product of all X(1) through X(10) = 1.0
        prod10 = jnp.prod(y)
        constraints.append(prod10 - 1.0)

        # All constraints are equalities
        return jnp.array(constraints), None

    @property
    def y0(self) -> Array:
        """Starting point from SIF file."""
        x0 = jnp.zeros(10)

        # Starting values follow pattern: X0A = 2.0, X0M = -1.0
        # For odd i: X(i) = X0A * X0M, X(i+1) = 1/X(i)
        # X0A increases by 1 each iteration, X0M multiplies by -1

        x0a = 2.0
        x0m = -1.0

        for i in range(0, 10, 2):
            x0_val = x0a * x0m
            x0 = x0.at[i].set(x0_val)
            x0 = x0.at[i + 1].set(1.0 / x0_val)
            x0a += 1.0
            x0m *= -1.0

        return x0

    @property
    def bounds(self) -> tuple[Array, Array]:
        """Variable bounds: all free variables."""
        lower = jnp.full(10, -jnp.inf)
        upper = jnp.full(10, jnp.inf)
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
        """Expected objective value from SIF file comments."""
        return jnp.array(0.0)  # From OBJECT BOUND section (SOLTN = 0.0)

    def num_constraints(self) -> tuple[int, int, int]:
        """Returns the number of constraints and bounds."""
        # All 5 constraints are equalities, no bounds
        return 5, 0, 0
