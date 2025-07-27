import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class PALMER8ANE(AbstractNonlinearEquations):
    """A nonlinear least squares problem with bounds arising from chemical kinetics.

    model: H-N=C=Se TZVP + MP2
    fitting Y to A0 + A2 X**2 + A4 X**4 + A6 X**6
                + B / ( C + X**2 ), B, C nonnegative.

    Source:
    M. Palmer, Edinburgh, private communication.

    SIF input: Nick Gould, 1992.
    Bound-constrained nonlinear equations version: Nick Gould, June 2019.

    classification NOR2-RN-6-12
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Data points
    M: int = 12

    # X values (radians) - indices 12-23
    X_data = jnp.array(
        [
            0.000000,
            0.174533,
            0.314159,
            0.436332,
            0.514504,
            0.610865,
            0.785398,
            0.959931,
            1.134464,
            1.308997,
            1.483530,
            1.570796,
        ]
    )

    # Y values (KJmol-1)
    Y_data = jnp.array(
        [
            4.757534,
            3.121416,
            1.207606,
            0.131916,
            0.000000,
            0.258514,
            3.380161,
            10.762813,
            23.745996,
            44.471864,
            76.541947,
            97.874528,
        ]
    )

    @property
    def n(self):
        """Number of variables."""
        return 6  # A0, A2, A4, A6, B, C

    @property
    def M_residuals(self):
        """Number of residual functions."""
        return 12

    def residual(self, y, args):
        """Compute the residual functions."""
        del args

        # Extract variables
        A0, A2, A4, A6, B, C = y

        # Precompute powers of X
        X_sqr = self.X_data * self.X_data
        X_4 = X_sqr * X_sqr
        X_6 = X_sqr * X_4

        # Model predictions
        predictions = A0 + A2 * X_sqr + A4 * X_4 + A6 * X_6 + B / (C + X_sqr)

        # Residuals (prediction - observation)
        residuals = predictions - self.Y_data

        return residuals

    @property
    def y0(self):
        """Initial guess."""
        return jnp.ones(6)

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return jnp.zeros(6)  # Placeholder

    @property
    def expected_objective_value(self):
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)

    def constraint(self, y):
        """Returns the residuals as equality constraints."""
        return self.residual(y, self.args), None

    @property
    def bounds(self):
        """Returns the bounds on the variable y."""
        # A0, A2, A4, A6 are free, B and C have lower bounds
        lower = jnp.array([-jnp.inf, -jnp.inf, -jnp.inf, -jnp.inf, 0.00001, 0.00001])
        upper = jnp.full(6, jnp.inf)
        return lower, upper
