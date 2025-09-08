import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class PALMER3NE(AbstractNonlinearEquations):
    """A nonlinear least squares problem with bounds arising from chemical kinetics.

    model: H-N=C=S TZVP + MP2
    fitting Y to A X**2 + B / ( C + X**2 / D ), B, C, D nonnegative.

    Source:
    M. Palmer, Edinburgh, private communication.

    SIF input: Nick Gould, 1990.
    Bound-constrained nonlinear equations version: Nick Gould, June 2019.

    classification NOR2-RN-4-23
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Data points
    M: int = 23

    # X values (radians)
    X_data = jnp.array(
        [
            -1.658063,
            -1.570796,
            -1.396263,
            -1.221730,
            -1.047198,
            -0.872665,
            -0.766531,
            -0.698132,
            -0.523599,
            -0.349066,
            -0.174533,
            0.0,
            0.174533,
            0.349066,
            0.523599,
            0.698132,
            0.766531,
            0.872665,
            1.047198,
            1.221730,
            1.396263,
            1.570796,
            1.658063,
        ]
    )

    # Y values (KJmol-1)
    Y_data = jnp.array(
        [
            64.87939,
            50.46046,
            28.2034,
            13.4575,
            4.6547,
            0.59447,
            0.0000,
            0.2177,
            2.3029,
            5.5191,
            8.5519,
            9.8919,
            8.5519,
            5.5191,
            2.3029,
            0.2177,
            0.0000,
            0.59447,
            4.6547,
            13.4575,
            28.2034,
            50.46046,
            64.87939,
        ]
    )

    @property
    def n(self):
        """Number of variables."""
        return 4  # A, B, C, D

    @property
    def M_residuals(self):
        """Number of residual functions."""
        return jnp.array(23)

    def residual(self, y, args):
        """Compute the residual functions."""
        del args

        # Extract variables
        A, B, C, D = y

        # Precompute powers of X
        X_sqr = self.X_data * self.X_data

        # Model predictions: A*X² + B/(C + X²/D)
        predictions = A * X_sqr + B / (C + X_sqr / D)

        # Residuals (prediction - observation)
        residuals = predictions - self.Y_data

        return residuals

    @property
    def y0(self):
        """Initial guess."""
        return jnp.ones(4)

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return jnp.zeros(4)  # Placeholder

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
        # A is free, B, C, D have lower bounds
        lower = jnp.array([-jnp.inf, 0.00001, 0.00001, 0.00001])
        upper = jnp.full(4, jnp.inf)
        return lower, upper
