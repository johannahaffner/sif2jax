import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class PALMER2(AbstractBoundedMinimisation):
    """A nonlinear least squares problem with bounds arising from chemical kinetics.

    model: H-N=C=O TZVP + MP2
    fitting Y to A X**2 + B / ( C + X**2 / D ), B, C, D nonnegative.

    Source:
    M. Palmer, Edinburgh, private communication.

    SIF input: Nick Gould, 1990.

    classification: SBR2-RN-4-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 4  # 4 variables: A, B, C, D
    m: int = 23  # 23 data points

    @property
    def y0(self):
        # All variables start at 1.0
        return jnp.ones(self.n)

    @property
    def args(self):
        # X data values (radians)
        x_data = jnp.array(
            [
                -1.745329,
                -1.570796,
                -1.396263,
                -1.221730,
                -1.047198,
                -0.937187,
                -0.872665,
                -0.698132,
                -0.523599,
                -0.349066,
                -0.174533,
                0.0,
                0.174533,
                0.349066,
                0.523599,
                0.698132,
                0.872665,
                0.937187,
                1.047198,
                1.221730,
                1.396263,
                1.570796,
                1.745329,
            ]
        )

        # Y data values (KJmol-1)
        y_data = jnp.array(
            [
                72.676767,
                40.149455,
                18.8548,
                6.4762,
                0.8596,
                0.00000,
                0.2730,
                3.2043,
                8.1080,
                13.4291,
                17.7149,
                19.4529,
                17.7149,
                13.4291,
                8.1080,
                3.2053,
                0.2730,
                0.00000,
                0.8596,
                6.4762,
                18.8548,
                40.149455,
                72.676767,
            ]
        )

        return (x_data, y_data)

    def objective(self, y, args):
        """Compute the sum of squared residuals."""
        a, b, c, d = y
        x_data, y_data = args

        # Model: Y = A * X^2 + B / (C + X^2 / D)
        x_sqr = x_data**2

        predicted = a * x_sqr + b / (c + x_sqr / d)

        # Compute sum of squared residuals
        residuals = predicted - y_data
        return jnp.sum(residuals**2)

    @property
    def bounds(self):
        # A is free (unbounded)
        # B, C, D have lower bound 0.00001
        lower = jnp.array([-jnp.inf, 0.00001, 0.00001, 0.00001])
        upper = jnp.array([jnp.inf, jnp.inf, jnp.inf, jnp.inf])
        return lower, upper

    @property
    def expected_result(self):
        # The optimal solution is not explicitly given in the SIF file
        return None

    @property
    def expected_objective_value(self):
        # From the SIF file comment
        return jnp.array(3651.097532)
