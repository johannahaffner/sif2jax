import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class VIBRBEAM(AbstractUnconstrainedMinimisation):
    """Vibrating beam least-squares problem from laser-Doppler measurements.

    A nonlinear least-squares problem arising from laser-Doppler
    measurements of a vibrating beam. The data correspond to a simulated
    experiment where two laser-Doppler velocimeters take measurements
    at random points along the centreline of the beam. These measurements
    consist of a position (x), an incident angle (p) and the magnitude
    of the velocity along the line of sight (v).

    The problem is to fit:
    v = (c0 + c1*x + c2*x^2 + c3*x^3) * cos(d0 + d1*x + d2*x^2 + d3*x^3 - p)

    Source:
    A modification of an exercise for L. Watson course on LANCELOT in
    the Spring 1993. Compared to the original proposal, the unnecessary
    elements were removed as well as an unnecessary constraint on the phase.

    SIF input: Ph. L. Toint, May 1993, based on a proposal by
    D. E. Montgomery, Virginia Tech., April 1993.

    Classification: SUR2-MN-8-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 8  # Number of variables
    m: int = 30  # Number of measurements

    @property
    def y0(self):
        """Initial guess."""
        return jnp.array([-3.5, 1.0, 0.0, 0.0, 1.7, 0.0, 0.0, 0.0])

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def bounds(self):
        """All variables are unbounded."""
        return None

    def objective(self, y, args=None):
        """Compute the least squares objective."""
        del args

        # Extract coefficients
        c0, c1, c2, c3 = y[0], y[1], y[2], y[3]
        d0, d1, d2, d3 = y[4], y[5], y[6], y[7]

        # Data: positions
        x_data = jnp.array(
            [
                39.1722,
                53.9707,
                47.9829,
                12.5925,
                16.5414,
                18.9548,
                27.7168,
                31.9201,
                45.6830,
                22.2524,
                33.9805,
                6.8425,
                35.1677,
                33.5682,
                43.3659,
                13.3835,
                25.7273,
                21.0230,
                10.9755,
                1.5323,
                45.4416,
                14.5431,
                22.4313,
                29.0144,
                25.2675,
                15.5095,
                9.6297,
                8.3009,
                30.8694,
                43.3299,
            ]
        )

        # Data: velocity magnitudes
        v_data = jnp.array(
            [
                -1.2026,
                1.7053,
                0.5410,
                1.1477,
                1.2447,
                0.9428,
                -0.1360,
                -0.7542,
                -0.3396,
                0.7057,
                -0.8509,
                -0.1201,
                -1.2193,
                -1.0448,
                -0.7723,
                0.4342,
                0.1154,
                0.2868,
                0.3558,
                -0.5090,
                -0.0842,
                0.6021,
                0.1197,
                -0.1827,
                0.1806,
                0.5395,
                0.2072,
                0.1466,
                -0.2672,
                -0.3038,
            ]
        )

        # Data: angles of incidence
        p_data = jnp.array(
            [
                2.5736,
                2.7078,
                2.6613,
                2.0374,
                2.1553,
                2.2195,
                2.4077,
                2.4772,
                2.6409,
                2.2981,
                2.5073,
                1.8380,
                2.5236,
                2.5015,
                2.6186,
                0.4947,
                0.6062,
                0.5588,
                0.4772,
                0.4184,
                0.9051,
                0.5035,
                0.5723,
                0.6437,
                0.6013,
                0.5111,
                0.4679,
                0.4590,
                0.6666,
                0.8630,
            ]
        )

        # Compute model predictions
        x2 = x_data**2
        x3 = x_data**3

        # Magnitude polynomial
        magnitude = c0 + c1 * x_data + c2 * x2 + c3 * x3

        # Phase polynomial
        phase = d0 + d1 * x_data + d2 * x2 + d3 * x3 - p_data

        # Model prediction
        v_model = magnitude * jnp.cos(phase)

        # Compute residuals
        residuals = v_model - v_data

        # Return sum of squared residuals
        return jnp.sum(residuals**2)

    @property
    def expected_result(self):
        """Expected solution (not provided in SIF)."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value from SIF file."""
        return jnp.array(0.15644607137)
