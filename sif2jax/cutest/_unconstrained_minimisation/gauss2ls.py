import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractUnconstrainedMinimisation


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class GAUSS2LS(AbstractUnconstrainedMinimisation):
    """The GAUSS2LS function.

    NIST Data fitting problem GAUSS2.

    Fit: y = b1*exp(-b2*x) + b3*exp(-(x-b4)^2/b5^2) + b6*exp(-(x-b7)^2/b8^2) + e

    Similar to GAUSS1LS but with different data.

    Source: Problem from the NIST nonlinear regression test set
    http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml

    Reference: Rust, B., NIST (1996).

    SIF input: Nick Gould and Tyrone Rees, Oct 2015

    Classification SUR2-MN-8-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 8

    def objective(self, y, args):
        """Compute the objective function value.

        Args:
            y: The parameters [b1, b2, b3, b4, b5, b6, b7, b8]
            args: None

        Returns:
            The sum of squared residuals.
        """
        del args
        b1, b2, b3, b4, b5, b6, b7, b8 = y

        # Create the x data points (1 to 250)
        x = jnp.arange(1.0, 251.0)

        # Model function:
        # b1*exp(-b2*x) + b3*exp(-(x-b4)^2/b5^2) + b6*exp(-(x-b7)^2/b8^2)
        term1 = b1 * jnp.exp(-b2 * x)
        term2 = b3 * jnp.exp(-((x - b4) ** 2) / (b5**2))
        term3 = b6 * jnp.exp(-((x - b7) ** 2) / (b8**2))
        model = term1 + term2 + term3

        # Actual y values from GAUSS2LS dataset (250 data points)
        y_data = jnp.array(
            [
                97.58776,
                97.76344,
                96.56705,
                92.52037,
                91.15097,
                95.21728,
                90.21355,
                89.29235,
                91.51479,
                89.60966,
                86.56187,
                85.55316,
                87.13054,
                85.67940,
                80.04851,
                82.18925,
                87.24081,
                80.79407,
                81.28570,
                81.56940,
                79.22715,
                79.43275,
                77.90195,
                76.75468,
                77.17377,
                74.27348,
                73.11900,
                73.84826,
                72.47870,
                71.92292,
                66.92176,
                67.93835,
                69.56207,
                69.07066,
                66.53983,
                63.87883,
                69.71537,
                63.60588,
                63.37154,
                60.01835,
                62.67481,
                65.80666,
                59.14304,
                56.62951,
                61.21785,
                54.38790,
                62.93443,
                56.65144,
                57.13362,
                58.29689,
                58.91744,
                58.50172,
                55.22885,
                58.30375,
                57.43237,
                51.69407,
                49.93132,
                53.70760,
                55.39712,
                52.89709,
                52.31649,
                53.98720,
                53.54158,
                56.45046,
                51.32276,
                53.11676,
                53.28631,
                49.80555,
                54.69564,
                56.41627,
                54.59362,
                54.38520,
                60.15354,
                59.78773,
                60.49995,
                65.43885,
                60.70001,
                63.71865,
                67.77139,
                64.70934,
                70.78457,
                72.60052,
                76.31874,
                80.64547,
                84.16975,
                90.55489,
                91.54231,
                95.12414,
                96.75860,
                105.4188,
                105.2340,
                111.3428,
                119.7987,
                128.7170,
                135.0829,
                139.5125,
                147.0778,
                148.6145,
                152.3789,
                153.4344,
                157.2717,
                160.3754,
                163.0221,
                166.3512,
                164.7679,
                166.0469,
                166.8891,
                167.0413,
                166.3324,
                164.9013,
                163.4345,
                162.4011,
                159.5542,
                159.0683,
                156.6131,
                154.0747,
                150.5208,
                147.2626,
                144.6678,
                141.2104,
                136.6325,
                133.8588,
                129.8454,
                127.1705,
                123.8618,
                118.8808,
                116.1449,
                110.8962,
                108.8716,
                105.0548,
                100.8115,
                97.40024,
                94.39029,
                89.28144,
                87.41980,
                83.47345,
                79.84738,
                75.74938,
                72.47966,
                67.44325,
                64.80276,
                61.14639,
                57.69679,
                54.52768,
                50.79986,
                48.28143,
                45.40880,
                41.99568,
                40.22090,
                37.48413,
                34.70748,
                32.58973,
                30.45053,
                28.29478,
                26.42606,
                24.47091,
                22.93869,
                21.09999,
                19.74830,
                18.39985,
                17.18445,
                15.95254,
                14.95448,
                13.93692,
                13.08890,
                12.18996,
                11.46404,
                10.75802,
                10.10669,
                9.473758,
                8.916876,
                8.411934,
                7.957354,
                7.554634,
                7.191984,
                6.866404,
                6.576644,
                6.321004,
                6.096764,
                5.902824,
                5.737284,
                5.598784,
                5.485884,
                5.396784,
                5.329884,
                5.283804,
                5.257264,
                5.249204,
                5.258684,
                5.284884,
                5.326964,
                5.384244,
                5.456164,
                5.542324,
                5.642404,
                5.756244,
                5.883804,
                6.025044,
                6.179924,
                6.348444,
                6.530564,
                6.726324,
                6.935724,
                7.158764,
                7.395484,
                7.645924,
                7.910124,
                8.188164,
                8.480084,
                8.785924,
                9.105764,
                9.439644,
                9.787644,
                10.14984,
                10.52628,
                10.91704,
                11.32220,
                11.74184,
                12.17604,
                12.62488,
                13.08844,
                13.56680,
                14.06004,
                14.56824,
                15.09148,
                15.62984,
                16.18340,
                16.75224,
                17.33644,
                17.93608,
                18.55124,
                19.18200,
                19.82844,
                20.49064,
                21.16868,
                21.86264,
                22.57260,
                23.29864,
                24.04084,
                24.79928,
                25.57404,
                26.36520,
                27.17284,
                27.99704,
                28.83788,
                29.69544,
                30.56980,
                31.46104,
                32.36916,
                33.29436,
            ]
        )

        # Sum of squared residuals (least squares objective)
        residuals = model - y_data
        return jnp.sum(residuals**2)

    def y0(self):
        """Return the starting point from the SIF file."""
        # START1 values from GAUSS2LS.SIF file
        return inexact_asarray(
            jnp.array([96.0, 0.009, 103.0, 106.0, 18.0, 72.0, 151.0, 18.0])
        )

    def args(self):
        """Return None as no additional args are needed."""
        return None

    def expected_result(self):
        """Return None as the exact solution is not specified."""
        return None

    def expected_objective_value(self):
        # The minimum objective value is not specified
        return None
