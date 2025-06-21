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
                110.4214,
                107.6628,
                111.7288,
                116.5115,
                120.7609,
                123.9553,
                124.2437,
                130.7996,
                133.2960,
                130.7788,
                132.0565,
                138.6584,
                142.9252,
                142.7215,
                144.1249,
                147.4377,
                148.2647,
                152.0519,
                147.3863,
                149.2074,
                148.9537,
                144.5876,
                148.1226,
                148.0144,
                143.8893,
                140.9088,
                143.4434,
                139.3938,
                135.9878,
                136.3927,
                126.7262,
                124.4487,
                122.8647,
                113.8557,
                113.7037,
                106.8407,
                107.0034,
                102.46290,
                96.09296,
                94.57555,
                86.98824,
                84.90154,
                81.18023,
                76.40117,
                67.09200,
                72.67155,
                68.10848,
                67.99088,
                63.34094,
                60.55253,
                56.18687,
                53.64482,
                53.70307,
                48.07893,
                42.21258,
                45.65181,
                41.69728,
                41.24946,
                39.21349,
                37.71696,
                36.68395,
                37.30393,
                37.43277,
                37.45012,
                32.64648,
                31.84347,
                31.39951,
                26.68912,
                32.25323,
                27.61008,
                33.58649,
                28.10714,
                30.26428,
                28.01648,
                29.11021,
                23.02099,
                25.65091,
                28.50295,
                25.23701,
                26.13828,
                33.53260,
                29.25195,
                27.09847,
                26.52999,
                25.52401,
                26.69218,
                24.55269,
                27.71763,
                25.20297,
                25.61483,
                25.06893,
                27.63930,
                24.94851,
                25.86806,
                22.48183,
                26.90045,
                25.39919,
                17.90614,
                23.76039,
                25.89689,
                27.64231,
                22.86101,
                26.47003,
                23.72888,
                27.54334,
                30.52683,
                28.07261,
                34.92815,
                28.29194,
                34.19161,
                35.41207,
                37.09336,
                40.98330,
                39.53923,
                47.80123,
                47.46305,
                51.04166,
                54.58065,
                57.53001,
                61.42089,
                62.79032,
                68.51455,
                70.23053,
                74.42776,
                76.59911,
                81.62053,
                83.42208,
                79.17451,
                88.56985,
                85.66525,
                86.55502,
                90.65907,
                84.27290,
                85.72220,
                83.10702,
                82.16884,
                80.42568,
                78.15692,
                79.79691,
                77.84378,
                74.50327,
                71.57289,
                65.88031,
                65.01385,
                60.19582,
                59.66726,
                52.95478,
                53.87792,
                44.91274,
                41.09909,
                41.68018,
                34.53379,
                34.86419,
                33.14787,
                29.58864,
                27.29462,
                21.91439,
                19.08159,
                24.90290,
                19.82341,
                16.75551,
                18.24558,
                17.23549,
                16.34934,
                13.71285,
                14.75676,
                13.97169,
                12.42867,
                14.35519,
                7.703309,
                10.234410,
                11.78315,
                13.87768,
                4.535700,
                10.059280,
                8.424824,
                10.533120,
                9.602255,
                7.877514,
                6.258121,
                8.899865,
                7.877754,
                12.51191,
                10.66205,
                6.035400,
                6.790655,
                8.783535,
                4.600288,
                8.400915,
                7.216561,
                10.017410,
                7.331278,
                6.527863,
                2.842001,
                10.325070,
                4.790995,
                8.377101,
                6.264445,
                2.706213,
                8.362329,
                8.983658,
                3.362571,
                1.182746,
                4.875359,
            ]
        )

        # Sum of squared residuals (least squares objective)
        residuals = model - y
        return jnp.sum(residuals**2)

    def y0(self):
        """Return the starting point from the SIF file."""
        # START1 values from SIF file
        return inexact_asarray(
            jnp.array([97.0, 0.009, 100.0, 65.0, 20.0, 70.0, 178.0, 16.5])
        )

    def args(self):
        """Return None as no additional args are needed."""
        return None

    def expected_result(self):
        """Return None as the exact solution is not specified in the SIF file."""
        # The problem doesn't specify the exact minimum point
        return None

    def expected_objective_value(self):
        # The problem doesn't specify the minimum objective value in the SIF file
        return None


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


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class GAUSS3LS(AbstractUnconstrainedMinimisation):
    """The GAUSS3LS function.

    NIST Data fitting problem GAUSS3.

    Fit: y = b1*exp(-b2*x) + b3*exp(-(x-b4)^2/b5^2) + b6*exp(-(x-b7)^2/b8^2) + e

    Similar to GAUSS1LS but with different data.

    Source: Problem from the NIST nonlinear regression test set
    http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml

    Reference: Rust, B., NIST (1996).

    SIF input: Nick Gould and Tyrone Rees, Oct 2015

    Classification SUR2-MN-8-0
    """

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

        # Actual y values from GAUSS3LS dataset (250 data points)
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
        """Return the starting point from the NIST GAUSS3 dataset."""
        # Starting point values from GAUSS3.dat
        return inexact_asarray(
            jnp.array([94.9, 0.009, 90.1, 113.0, 20.0, 73.8, 140.0, 20.0])
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
