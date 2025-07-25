import jax
import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# TODO: Human review needed to verify the implementation matches the problem definition
class KIRBY2LS(AbstractUnconstrainedMinimisation):
    """NIST Data fitting problem KIRBY2.

    Fit: y = (b1 + b2*x + b3*x**2) / (1 + b4*x + b5*x**2) + e

    Source: Problem from the NIST nonlinear regression test set
    http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml

    Reference: Kirby, R., NIST (197?).
    Scanning electron microscope line width standards.

    SIF input: Nick Gould and Tyrone Rees, Oct 2015
    Classification: SUR2-MN-5-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Set of valid starting point IDs
    valid_ids = frozenset([0, 1])

    # Starting point ID (0 or 1)
    y0_id: int = 0

    def __check_init__(self):
        """Validate that y0_id is a valid starting point ID."""
        if self.y0_id not in self.valid_ids:
            raise ValueError(f"y0_id must be one of {self.valid_ids}")

    def model(self, x, params):
        """Compute the model function: (b1 + b2*x + b3*x^2) / (1 + b4*x + b5*x^2)"""
        b1, b2, b3, b4, b5 = params
        numerator = b1 + b2 * x + b3 * x**2
        denominator = 1.0 + b4 * x + b5 * x**2
        return numerator / denominator

    def objective(self, y, args):
        """Compute the objective function value.

        The model is: y = (b1 + b2*x + b3*x^2) / (1 + b4*x + b5*x^2) + e

        The objective is the sum of squares of the residuals.
        """
        # Calculate the predicted values using the model
        x_values = jnp.array(
            [
                9.65,
                10.74,
                11.81,
                12.88,
                14.06,
                15.28,
                16.63,
                18.19,
                19.88,
                21.84,
                24.0,
                26.25,
                28.86,
                31.85,
                35.79,
                40.18,
                44.74,
                49.53,
                53.94,
                58.29,
                62.63,
                67.03,
                71.25,
                75.22,
                79.33,
                83.56,
                87.75,
                91.93,
                96.1,
                100.28,
                104.46,
                108.66,
                112.71,
                116.88,
                121.33,
                125.79,
                125.79,
                128.74,
                130.27,
                133.33,
                134.79,
                137.93,
                139.33,
                142.46,
                143.9,
                146.91,
                148.51,
                151.41,
                153.17,
                155.97,
                157.76,
                160.56,
                162.30,
                165.21,
                166.9,
                169.92,
                170.32,
                171.54,
                173.79,
                174.57,
                176.25,
                177.34,
                179.19,
                181.02,
                182.08,
                183.88,
                185.75,
                186.80,
                188.63,
                190.45,
                191.48,
                193.35,
                195.22,
                196.23,
                198.05,
                199.97,
                201.06,
                202.83,
                204.69,
                205.86,
                207.58,
                209.50,
                210.65,
                212.33,
                215.43,
                217.16,
                220.21,
                221.98,
                225.06,
                226.79,
                229.92,
                231.69,
                234.77,
                236.6,
                239.63,
                241.50,
                244.48,
                246.40,
                249.35,
                251.32,
                254.22,
                256.24,
                259.11,
                261.18,
                264.02,
                266.13,
                268.94,
                271.09,
                273.87,
                276.08,
                278.83,
                281.08,
                283.81,
                286.11,
                288.81,
                291.08,
                293.75,
                295.99,
                298.64,
                300.84,
                302.02,
                303.48,
                305.65,
                308.27,
                310.41,
                313.01,
                315.12,
                317.71,
                319.79,
                322.36,
                324.42,
                326.98,
                329.01,
                331.56,
                333.56,
                336.1,
                338.08,
                340.6,
                342.57,
                345.08,
                347.02,
                349.52,
                351.44,
                353.93,
                355.83,
                358.32,
                360.2,
                362.67,
                364.53,
                367.0,
                371.3,
            ]
        )
        y_pred = jax.vmap(lambda x: self.model(x, y))(x_values)

        # Calculate the residuals
        y_values = jnp.array(
            [
                0.0082,
                0.0112,
                0.0149,
                0.0198,
                0.0248,
                0.0324,
                0.042,
                0.0549,
                0.0719,
                0.0963,
                0.1291,
                0.171,
                0.2314,
                0.3227,
                0.4809,
                0.7084,
                1.022,
                1.458,
                1.952,
                2.541,
                3.223,
                3.999,
                4.852,
                5.732,
                6.727,
                7.835,
                9.025,
                10.267,
                11.578,
                12.944,
                14.377,
                15.856,
                17.331,
                18.885,
                20.575,
                22.32,
                22.303,
                23.46,
                24.06,
                25.272,
                25.853,
                27.11,
                27.658,
                28.924,
                29.511,
                30.71,
                31.35,
                32.52,
                33.23,
                34.33,
                35.06,
                36.17,
                36.84,
                38.01,
                38.67,
                39.87,
                40.03,
                40.5,
                41.37,
                41.67,
                42.31,
                42.73,
                43.46,
                44.14,
                44.55,
                45.22,
                45.92,
                46.3,
                47.0,
                47.68,
                48.06,
                48.74,
                49.41,
                49.76,
                50.43,
                51.11,
                51.5,
                52.12,
                52.76,
                53.18,
                53.78,
                54.46,
                54.83,
                55.4,
                56.43,
                57.03,
                58.0,
                58.61,
                59.58,
                60.11,
                61.1,
                61.65,
                62.59,
                63.12,
                64.03,
                64.62,
                65.49,
                66.03,
                66.89,
                67.42,
                68.23,
                68.77,
                69.59,
                70.11,
                70.86,
                71.43,
                72.16,
                72.7,
                73.4,
                73.93,
                74.6,
                75.16,
                75.82,
                76.34,
                76.98,
                77.48,
                78.08,
                78.6,
                79.17,
                79.62,
                79.88,
                80.19,
                80.66,
                81.22,
                81.66,
                82.16,
                82.59,
                83.14,
                83.5,
                84.0,
                84.4,
                84.89,
                85.26,
                85.74,
                86.07,
                86.54,
                86.89,
                87.32,
                87.65,
                88.1,
                88.43,
                88.83,
                89.12,
                89.54,
                89.85,
                90.25,
                90.55,
                90.93,
                91.2,
                91.55,
                92.2,
            ]
        )
        residuals = y_pred - y_values

        # Return the sum of squared residuals
        return jnp.sum(residuals**2)

    @property
    def y0(self):
        """Initial point based on the y0_id parameter."""
        if self.y0_id == 0:
            return jnp.array([2.0, -0.1, 0.003, -0.001, 0.00001])  # START1
        elif self.y0_id == 1:
            return jnp.array([1.5, -0.15, 0.0025, -0.0015, 0.00002])  # START2
        else:
            assert False, "Invalid y0_id"

    @property
    def args(self):
        """No additional arguments needed."""
        return None

    @property
    def expected_result(self):
        """The solution is not specified in the SIF file."""
        return None

    @property
    def expected_objective_value(self):
        """The minimal sum of squares."""
        return None
