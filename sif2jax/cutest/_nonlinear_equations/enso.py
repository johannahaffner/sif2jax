import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractNonlinearEquations


class ENSO(AbstractNonlinearEquations):
    """NIST Data fitting problem ENSO as inconsistent nonlinear equations.

    Fit: y = b1 + b2*cos( 2*pi*x/12 ) + b3*sin( 2*pi*x/12 )
                    + b5*cos( 2*pi*x/b4 ) + b6*sin( 2*pi*x/b4 )
                    + b8*cos( 2*pi*x/b7 ) + b9*sin( 2*pi*x/b7 ) + e

    Source:  Problem from the NIST nonlinear regression test set
        http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml

    Reference: Kahaner, D., C. Moler, and S. Nash, (1989).
        Numerical Methods and Software.
        Englewood Cliffs, NJ: Prentice Hall, pp. 441-445.

    SIF input: Nick Gould and Tyrone Rees, Oct 2015

    Classification: NOR2-MN-9-168
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Problem parameters
    M: int = 168  # Number of data points
    N: int = 9  # Number of variables

    @property
    def n(self):
        """Number of variables."""
        return self.N

    def num_residuals(self):
        """Number of residual equations."""
        return self.M

    def residual(self, y, args):
        """Compute the residuals for the ENSO fitting problem."""
        del args

        # Extract parameters
        b1, b2, b3, b4, b5, b6, b7, b8, b9 = (
            y[0],
            y[1],
            y[2],
            y[3],
            y[4],
            y[5],
            y[6],
            y[7],
            y[8],
        )

        # Data points
        x_data = jnp.arange(1, self.M + 1, dtype=y.dtype)

        # Target values (from SIF file)
        y_data = jnp.array(
            [
                12.9,
                11.3,
                10.6,
                11.2,
                10.9,
                7.5,
                7.7,
                11.7,
                12.9,
                14.3,
                10.9,
                13.7,
                17.1,
                14.0,
                15.3,
                8.5,
                5.7,
                5.5,
                7.6,
                8.6,
                7.3,
                7.6,
                12.7,
                11.0,
                12.7,
                12.9,
                13.0,
                10.9,
                10.4,
                10.2,
                8.0,
                10.9,
                13.6,
                10.5,
                9.2,
                12.4,
                12.7,
                13.3,
                10.1,
                7.8,
                4.8,
                3.0,
                2.5,
                6.3,
                9.7,
                11.6,
                8.6,
                12.4,
                10.5,
                13.3,
                10.4,
                8.1,
                3.7,
                10.7,
                5.1,
                10.4,
                10.9,
                11.7,
                11.4,
                13.7,
                14.1,
                14.0,
                12.5,
                6.3,
                9.6,
                11.7,
                5.0,
                10.8,
                12.7,
                10.8,
                11.8,
                12.6,
                15.7,
                12.6,
                14.8,
                7.8,
                7.1,
                11.2,
                8.1,
                6.4,
                5.2,
                12.0,
                10.2,
                12.7,
                10.2,
                14.7,
                12.2,
                7.1,
                5.7,
                6.7,
                3.9,
                8.5,
                8.3,
                10.8,
                16.7,
                12.6,
                12.5,
                12.5,
                9.8,
                7.2,
                4.1,
                10.6,
                10.1,
                10.1,
                11.9,
                13.6,
                16.3,
                17.6,
                15.5,
                16.0,
                15.2,
                11.2,
                14.3,
                14.5,
                8.5,
                12.0,
                12.7,
                11.3,
                14.5,
                15.1,
                10.4,
                11.5,
                13.4,
                7.5,
                0.6,
                0.3,
                5.5,
                5.0,
                4.6,
                8.2,
                9.9,
                9.2,
                12.5,
                10.9,
                9.9,
                8.9,
                7.6,
                9.5,
                8.4,
                10.7,
                13.6,
                13.7,
                13.7,
                16.5,
                16.8,
                17.1,
                15.4,
                9.5,
                6.1,
                10.1,
                9.3,
                5.3,
                11.2,
                16.6,
                15.6,
                12.0,
                11.5,
                8.6,
                13.8,
                8.7,
                8.6,
                8.6,
                8.7,
                12.8,
                13.2,
                14.0,
                13.4,
                14.8,
            ],
            dtype=y.dtype,
        )

        # Compute model predictions
        two_pi = 2.0 * jnp.pi
        two_pi_by_12 = two_pi / 12.0

        # Fixed period terms (period = 12)
        arg_12 = two_pi_by_12 * x_data
        cos_12 = jnp.cos(arg_12)
        sin_12 = jnp.sin(arg_12)

        # Variable period terms (period = b4)
        two_pi_x = two_pi * x_data
        arg_b4 = two_pi_x / b4
        cos_b4 = jnp.cos(arg_b4)
        sin_b4 = jnp.sin(arg_b4)

        # Variable period terms (period = b7)
        arg_b7 = two_pi_x / b7
        cos_b7 = jnp.cos(arg_b7)
        sin_b7 = jnp.sin(arg_b7)

        # Model predictions
        y_pred = (
            b1
            + b2 * cos_12
            + b3 * sin_12
            + b5 * cos_b4
            + b6 * sin_b4
            + b8 * cos_b7
            + b9 * sin_b7
        )

        # Residuals
        residuals = y_pred - y_data

        return residuals

    @property
    def y0(self):
        """Initial guess (START1 from SIF file)."""
        return inexact_asarray(
            jnp.array(
                [
                    11.0,  # b1
                    3.0,  # b2
                    0.5,  # b3
                    40.0,  # b4
                    -0.7,  # b5
                    -1.3,  # b6
                    25.0,  # b7
                    -0.3,  # b8
                    1.4,  # b9
                ]
            )
        )

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def expected_result(self):
        """Expected optimal solution (from SIF file)."""
        return inexact_asarray(
            jnp.array(
                [
                    10.051074919,
                    3.0762128085,
                    0.5328013823,
                    44.311088700,
                    -1.623142859,
                    0.5255449376,
                    26.887614440,
                    0.2123228849,
                    1.4966870418,
                ]
            )
        )

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        # For nonlinear equations, objective should be close to 0
        return jnp.array(0.0)

    def constraint(self, y):
        """Returns the residuals as equality constraints."""
        return self.residual(y, self.args), None

    @property
    def bounds(self):
        """No bounds for this problem."""
        return None
