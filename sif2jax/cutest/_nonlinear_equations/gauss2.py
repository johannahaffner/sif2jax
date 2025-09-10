import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class GAUSS2(AbstractNonlinearEquations):
    """NIST Data fitting problem GAUSS2.

    NIST Data fitting problem GAUSS2 given as an inconsistent set of
    nonlinear equations.

    Fit: y = b1*exp( -b2*x ) + b3*exp( -(x-b4)**2 / b5**2 )
                             + b6*exp( -(x-b7)**2 / b8**2 ) + e

    Source: Problem from the NIST nonlinear regression test set
        http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml

    Reference: Rust, B., NIST (1996).

    SIF input: Nick Gould and Tyrone Rees, Oct 2015

    Classification: NOR2-MN-8-250
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Number of variables
    n_var: int = 8

    # Number of constraints (equations)
    n_con: int = 250

    @property
    def y0(self):
        """Return the starting point from START1 in the SIF file."""
        return jnp.array([96.0, 0.009, 103.0, 106.0, 18.0, 72.0, 151.0, 18.0])

    @property
    def args(self):
        """Return None as no additional args are needed."""
        return None

    @property
    def bounds(self):
        """Return unbounded constraints."""
        return None

    @property
    def expected_result(self):
        """Return None as the exact solution is not specified."""
        return None

    @property
    def expected_objective_value(self):
        """Return None as the minimum objective value is not specified."""
        return None

    def constraint(self, y, args=None):
        """Compute the constraint residuals.

        Args:
            y: The parameters [b1, b2, b3, b4, b5, b6, b7, b8]
            args: None

        Returns:
            Tuple of (equality_constraints, inequality_constraints).
            For nonlinear equations, all constraints are equalities.
        """
        del args
        b1, b2, b3, b4, b5, b6, b7, b8 = y

        # X values from 1 to 250
        x = jnp.arange(1.0, 251.0)

        # Y data values from the SIF file
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
                70.78193,
                70.38651,
                77.22359,
                79.52665,
                80.13077,
                85.67823,
                85.20647,
                90.24548,
                93.61953,
                95.86509,
                93.46992,
                105.8137,
                107.8269,
                114.0607,
                115.5019,
                118.5110,
                119.6177,
                122.1940,
                126.9903,
                125.7005,
                123.7447,
                130.6543,
                129.7168,
                131.8240,
                131.8759,
                131.9994,
                132.1221,
                133.4414,
                133.8252,
                133.6695,
                128.2851,
                126.5182,
                124.7550,
                118.4016,
                122.0334,
                115.2059,
                118.7856,
                110.7387,
                110.2003,
                105.17290,
                103.44720,
                94.54280,
                94.40526,
                94.57964,
                88.76605,
                87.28747,
                92.50443,
                86.27997,
                82.44307,
                80.47367,
                78.36608,
                78.74307,
                76.12786,
                79.13108,
                76.76062,
                77.60769,
                77.76633,
                81.28220,
                79.74307,
                81.97964,
                80.02952,
                85.95232,
                85.96838,
                79.94789,
                87.17023,
                90.50992,
                93.23373,
                89.14803,
                93.11492,
                90.34337,
                93.69421,
                95.74256,
                91.85105,
                96.74503,
                87.60996,
                90.47012,
                88.11690,
                85.70673,
                85.01361,
                78.53040,
                81.34148,
                75.19295,
                72.66115,
                69.85504,
                66.29476,
                63.58502,
                58.33847,
                57.50766,
                52.80498,
                50.79319,
                47.03490,
                46.47090,
                43.09016,
                34.11531,
                39.28235,
                32.68386,
                30.44056,
                31.98932,
                23.63330,
                23.69643,
                20.26812,
                19.07074,
                17.59544,
                16.08785,
                18.94267,
                18.61354,
                17.25800,
                16.62285,
                13.48367,
                15.37647,
                13.47208,
                15.96188,
                12.32547,
                16.33880,
                10.438330,
                9.628715,
                13.12268,
                8.772417,
                11.76143,
                12.55020,
                11.33108,
                11.20493,
                7.816916,
                6.800675,
                14.26581,
                10.66285,
                8.911574,
                11.56733,
                11.58207,
                11.59071,
                9.730134,
                11.44237,
                11.22912,
                10.172130,
                12.50905,
                6.201493,
                9.019605,
                10.80607,
                13.09625,
                3.914271,
                9.567886,
                8.038448,
                10.231040,
                9.367410,
                7.695971,
                6.118575,
                8.793207,
                7.796692,
                12.45065,
                10.61601,
                6.001003,
                6.765098,
                8.764653,
                4.586418,
                8.390783,
                7.209202,
                10.012090,
                7.327461,
                6.525136,
                2.840065,
                10.323710,
                4.790035,
                8.376431,
                6.263980,
                2.705892,
                8.362109,
                8.983507,
                3.362469,
                1.182678,
                4.875312,
            ]
        )

        # Compute the model using vectorized operations
        # First exponential term: b1 * exp(-b2 * x)
        term1 = b1 * jnp.exp(-b2 * x)

        # Second Gaussian term: b3 * exp(-(x-b4)^2 / b5^2)
        term2 = b3 * jnp.exp(-((x - b4) ** 2) / (b5**2))

        # Third Gaussian term: b6 * exp(-(x-b7)^2 / b8^2)
        term3 = b6 * jnp.exp(-((x - b7) ** 2) / (b8**2))

        # Full model
        model = term1 + term2 + term3

        # Return residuals as equality constraints (model - data = 0)
        # For nonlinear equations, return (equalities, None)
        return model - y_data, None
