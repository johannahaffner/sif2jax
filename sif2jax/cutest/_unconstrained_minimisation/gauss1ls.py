import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractUnconstrainedMinimisation


# TODO: claude seems to struggle adding all the data and starting points provided.
# Perhaps this is just longer than the context window it has?
# TODO: This implementation requires human review and verification against
# another CUTEst interface
class GAUSS1LS(AbstractUnconstrainedMinimisation):
    """The GAUSS1LS function.

    NIST Data fitting problem GAUSS1.

    Fit: y = b1*exp(-b2*x) + b3*exp(-(x-b4)^2/b5^2) + b6*exp(-(x-b7)^2/b8^2) + e

    Source: Problem from the NIST nonlinear regression test set
    http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml

    Reference: Rust, B., NIST (1996).

    SIF input: Nick Gould and Tyrone Rees, Oct 2015

    Classification SUR2-MN-8-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Number of variables
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

        # Actual y values from the dataset (hard-coded from the SIF file)
        y = jnp.array(
            [
                97.62227,
                97.80724,
                96.62247,
                92.59022,
                91.23869,
                95.32704,
                90.35040,
                89.46235,
                91.72520,
                89.86916,
                86.88076,
                85.94360,
                87.60686,
                86.25839,
                80.74976,
                83.03551,
                88.25837,
                82.01316,
                82.74098,
                83.30034,
                81.27850,
                81.85506,
                80.75195,
                80.09573,
                81.07633,
                78.81542,
                78.38596,
                79.93386,
                79.48474,
                79.95942,
                76.10691,
                78.39830,
                81.43060,
                82.48867,
                81.65462,
                80.84323,
                88.68663,
                84.74438,
                86.83934,
                85.97739,
                91.28509,
                97.22411,
                93.51733,
                94.10159,
                101.91760,
                98.43134,
                110.42420,
                107.65120,
                108.50280,
                109.57720,
                110.61180,
                118.40160,
                118.57090,
                117.83310,
                118.34060,
                125.01420,
                127.47080,
                128.77090,
                125.51320,
                127.85910,
                127.91430,
                130.63490,
                131.99530,
                130.02780,
                129.40070,
                131.68850,
                129.30070,
                130.37540,
                131.83070,
                127.58230,
                126.91600,
                126.35160,
                127.87090,
                124.39170,
                124.33210,
                124.43990,
                118.75770,
                122.34180,
                123.68870,
                120.51850,
                119.33640,
                118.01970,
                121.78050,
                117.80480,
                116.35320,
                119.19680,
                114.72050,
                114.52610,
                115.16320,
                114.49200,
                112.98540,
                111.29250,
                109.89150,
                111.99670,
                110.72050,
                106.68180,
                107.12240,
                105.84030,
                104.84780,
                106.73160,
                104.89430,
                105.42150,
                105.73260,
                102.51360,
                101.73970,
                102.45350,
                100.95730,
                98.74206,
                100.50000,
                99.23275,
                97.75669,
                97.87366,
                94.30035,
                96.73041,
                96.10015,
                93.01642,
                95.62249,
                91.87976,
                93.60611,
                92.98297,
                91.31426,
                90.22386,
                91.18551,
                88.51763,
                91.32220,
                88.77680,
                88.16876,
                90.13254,
                86.50886,
                88.07228,
                86.96506,
                86.23732,
                86.31581,
                83.61736,
                85.50830,
                86.29587,
                84.72216,
                83.35141,
                84.35362,
                82.32418,
                82.51047,
                83.23610,
                82.00940,
                81.78757,
                82.10106,
                80.51560,
                81.66325,
                80.18857,
                81.10371,
                79.74307,
                77.84018,
                80.09155,
                78.22557,
                78.79576,
                77.63917,
                77.98667,
                77.37449,
                76.38426,
                76.21322,
                77.43166,
                74.62957,
                74.99614,
                74.36894,
                74.99986,
                73.43523,
                74.24097,
                73.48867,
                72.56952,
                72.79146,
                72.16523,
                72.23625,
                71.48302,
                73.51476,
                71.44837,
                70.46214,
                71.13686,
                69.85378,
                70.34612,
                70.36524,
                69.91600,
                68.52615,
                70.21768,
                68.82444,
                68.53114,
                69.63062,
                68.13274,
                69.32359,
                67.60641,
                68.25302,
                67.29528,
                68.89251,
                66.78339,
                67.96781,
                66.32833,
                66.78809,
                66.07882,
                67.34123,
                65.82567,
                64.96388,
                65.78449,
                66.41563,
                63.52857,
                64.65115,
                63.99120,
                63.12951,
                64.60763,
                63.29909,
                63.15549,
                62.93062,
                62.17658,
                62.45113,
                62.42105,
                61.96395,
                61.79597,
                62.77611,
                60.38479,
                61.00254,
                59.69350,
                61.23137,
                60.33176,
                60.55156,
                60.23767,
                59.11427,
                60.35052,
                59.29149,
                59.45965,
                59.10577,
                58.89726,
                58.19446,
                57.79246,
                59.25160,
                57.20736,
                58.15593,
                57.88078,
                57.11308,
                57.54192,
                56.73338,
                57.06495,
                57.69955,
                56.32157,
                56.46548,
                56.96916,
                55.74716,
                56.95495,
                55.68436,
                56.57139,
                55.89287,
            ]
        )

        # Compute residuals
        residuals = model - y

        # Return sum of squared residuals
        return jnp.sum(residuals**2)

    def y0(self):
        """Initial parameter guess."""
        # From START POINT 1 in the SIF file
        return inexact_asarray(
            jnp.array([96.0, 0.009, 103.0, 106.0, 18.0, 72.0, 151.0, 18.0])
        )

    def args(self):
        """No additional arguments."""
        return None

    def expected_result(self):
        """The expected result is not specified in the SIF file."""
        return None

    def expected_objective_value(self):
        """The expected objective value is not specified in the SIF file."""
        return None
