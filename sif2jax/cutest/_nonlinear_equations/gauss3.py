import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class GAUSS3(AbstractNonlinearEquations):
    """NIST Data fitting problem GAUSS3.

    NIST Data fitting problem GAUSS3 given as an inconsistent set of
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
        return jnp.array([94.9, 0.009, 90.1, 113.0, 20.0, 73.8, 140.0, 20.0])

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
                89.60965,
                86.56187,
                85.55315,
                87.13053,
                85.67938,
                80.04849,
                82.18922,
                87.24078,
                80.79401,
                81.28564,
                81.56932,
                79.22703,
                79.43259,
                77.90174,
                76.75438,
                77.17338,
                74.27296,
                73.11830,
                73.84732,
                72.47746,
                71.92128,
                66.91962,
                67.93554,
                69.55841,
                69.06592,
                66.53371,
                63.87094,
                69.70526,
                63.59295,
                63.35509,
                59.99747,
                62.64843,
                65.77345,
                59.10141,
                56.57750,
                61.15313,
                54.30767,
                62.83535,
                56.52957,
                56.98427,
                58.11459,
                58.69576,
                58.23322,
                54.90490,
                57.91442,
                56.96629,
                51.13831,
                49.27123,
                52.92668,
                54.47693,
                51.81710,
                51.05401,
                52.51731,
                51.83710,
                54.48196,
                49.05859,
                50.52315,
                50.32755,
                46.44419,
                50.89281,
                52.13203,
                49.78741,
                49.01637,
                54.18198,
                53.17456,
                53.20827,
                57.43459,
                51.95282,
                54.20282,
                57.46687,
                53.60268,
                58.86728,
                57.66652,
                63.71034,
                65.24244,
                65.10878,
                69.96313,
                68.85475,
                73.32574,
                76.21241,
                78.06311,
                75.37701,
                87.54449,
                89.50588,
                95.82098,
                97.48390,
                100.86070,
                102.48510,
                105.7311,
                111.3489,
                111.0305,
                110.1920,
                118.3581,
                118.8086,
                122.4249,
                124.0953,
                125.9337,
                127.8533,
                131.0361,
                133.3343,
                135.1278,
                131.7113,
                131.9151,
                132.1107,
                127.6898,
                133.2148,
                128.2296,
                133.5902,
                127.2539,
                128.3482,
                124.8694,
                124.6031,
                117.0648,
                118.1966,
                119.5408,
                114.7946,
                114.2780,
                120.3484,
                114.8647,
                111.6514,
                110.1826,
                108.4461,
                109.0571,
                106.5308,
                109.4691,
                106.8709,
                107.3192,
                106.9000,
                109.6526,
                107.1602,
                108.2509,
                104.96310,
                109.3601,
                107.6696,
                99.77286,
                104.96440,
                106.1376,
                106.5816,
                100.12860,
                101.66910,
                96.44254,
                97.34169,
                96.97412,
                90.73460,
                93.37949,
                82.12331,
                83.01657,
                78.87360,
                74.86971,
                72.79341,
                65.14744,
                67.02127,
                60.16136,
                57.13996,
                54.05769,
                50.42265,
                47.82430,
                42.85748,
                42.45495,
                38.30808,
                36.95794,
                33.94543,
                34.19017,
                31.66097,
                23.56172,
                29.61143,
                23.88765,
                22.49812,
                24.86901,
                17.29481,
                18.09291,
                15.34813,
                14.77997,
                13.87832,
                12.88891,
                16.20763,
                16.29024,
                15.29712,
                14.97839,
                12.11330,
                14.24168,
                12.53824,
                15.19818,
                11.70478,
                15.83745,
                10.035850,
                9.307574,
                12.86800,
                8.571671,
                11.60415,
                12.42772,
                11.23627,
                11.13198,
                7.761117,
                6.758250,
                14.23375,
                10.63876,
                8.893581,
                11.55398,
                11.57221,
                11.58347,
                9.724857,
                11.43854,
                11.22636,
                10.170150,
                12.50765,
                6.200494,
                9.018902,
                10.80557,
                13.09591,
                3.914033,
                9.567723,
                8.038338,
                10.230960,
                9.367358,
                7.695937,
                6.118552,
                8.793192,
                7.796682,
                12.45064,
                10.61601,
                6.001000,
                6.765096,
                8.764652,
                4.586417,
                8.390782,
                7.209201,
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
