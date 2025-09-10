import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractBoundedQuadraticProblem


class BQPGABIM(AbstractBoundedQuadraticProblem):
    """BQPGABIM problem - first 50 variables from BQPGAUSS with fixed variables.

    The first 50 variable subproblem from BQPGAUSS.
    but with variables 1, 15, 42 and 50 fixed at zero

    SIF input: N. Gould, July 1990.

    classification QBR2-AN-50-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return 50

    def objective(self, y, args):
        """Compute the objective.

        The objective is a quadratic function with linear and quadratic terms.
        """
        del args

        # Linear coefficients (from VARIABLES section)
        c = jnp.array(
            [
                5.6987e-02,
                -6.1847e-03,
                5.2516e-03,
                1.1729e-02,
                4.9596e-03,
                -4.9271e-03,
                1.2185e-02,
                1.3238e-02,
                -1.5134e-02,
                -1.2247e-02,
                2.3741e-02,
                -9.7666e-02,
                9.8702e-02,
                7.8901e-04,
                5.1663e-04,
                -1.7477e-04,
                1.1795e-03,
                -1.7351e-02,
                1.3439e-03,
                -5.6977e-02,
                1.0040e-02,
                -8.3380e-02,
                -3.7526e-03,
                -9.4555e-04,
                -4.9258e-03,
                -1.3959e-03,
                -4.3749e-03,
                -4.3677e-03,
                -2.7985e-02,
                1.8839e-03,
                -1.2340e-03,
                -6.8139e-04,
                -3.5838e-02,
                -3.4857e-02,
                2.8724e-03,
                1.6625e-02,
                1.3571e-02,
                -7.2447e-03,
                -4.6034e-04,
                -1.6225e-02,
                2.2034e-05,
                5.8844e-02,
                3.0725e-03,
                2.8227e-03,
                -2.0681e-02,
                -5.4952e-03,
                6.2552e-04,
                3.3782e-02,
                -4.8584e-03,
                -1.4371e-03,
            ]
        )

        # Build the Hessian matrix from the quadratic terms
        # Initialize Hessian with zeros
        H = jnp.zeros((50, 50))

        # Diagonal elements (from D entries in GROUP USES section)
        # These are multiplied by 0.5 since the element function is 0.5 * x * x
        diag_values = jnp.array(
            [
                1.0624e03,
                1.0624e03,
                1.0624e03,
                1.0624e03,
                1.0624e03,
                1.0624e03,
                1.0624e03,
                1.0624e03,
                1.0624e03,
                1.0624e03,
                7.8331e02,
                1.0000e02,
                1.0000e02,
                1.0000e02,
                1.0000e02,
                1.0000e02,
                1.0000e02,
                1.0000e02,
                1.0000e02,
                7.8331e02,
                1.0000e02,
                1.0000e02,
                1.0000e02,
                1.0000e02,
                1.0000e02,
                1.0000e02,
                1.0000e02,
                1.0000e02,
                7.8331e02,
                1.0000e02,
                1.0000e02,
                1.0000e02,
                1.0000e02,
                1.0000e02,
                1.0000e02,
                7.8331e02,
                1.0000e02,
                1.0000e02,
                1.0000e02,
                1.0000e02,
                7.8331e02,
                1.0000e02,
                1.0000e02,
                1.0000e02,
                1.0000e02,
                1.0000e02,
                1.0000e02,
                1.0000e02,
                7.8331e02,
                1.0000e02,
            ]
        )
        H = H.at[jnp.diag_indices(50)].set(diag_values)

        # Off-diagonal elements (from O entries in GROUP USES section)
        # These are symmetric, so we set both (i,j) and (j,i)
        # Format: (i-1, j-1, value) since SIF uses 1-based indexing
        off_diag = [
            (0, 10, -9.9819e01),
            (0, 11, -9.9709e01),
            (10, 11, 1.0000e02),
            (0, 19, -1.0000e02),
            (0, 20, -1.0000e02),
            (19, 20, 1.0000e02),
            (0, 28, 9.0362e01),
            (0, 35, 6.5103e01),
            (0, 36, 6.5140e01),
            (35, 36, 1.0000e02),
            (0, 40, 7.5507e01),
            (0, 41, 7.5507e01),
            (40, 41, 1.0000e02),
            (0, 48, -9.7537e01),
            (1, 10, -9.9213e01),
            (1, 12, -9.9709e01),
            (10, 12, 9.9608e01),
            (1, 19, -9.9698e01),
            (1, 21, -1.0000e02),
            (19, 21, 9.9608e01),
            (1, 28, 8.9945e01),
            (1, 29, 9.0300e01),
            (28, 29, 9.9608e01),
            (1, 35, 6.4885e01),
            (1, 37, 6.5140e01),
            (35, 37, 9.9608e01),
            (1, 40, 7.5197e01),
            (1, 48, -9.7167e01),
            (2, 10, 8.1209e01),
            (2, 19, 8.1463e01),
            (2, 22, -1.0000e02),
            (19, 22, -8.1463e01),
            (2, 28, -7.3536e01),
            (2, 35, -5.3119e01),
            (2, 40, -6.1506e01),
            (2, 42, 7.5507e01),
            (40, 42, -8.1463e01),
            (2, 48, 7.9480e01),
            (2, 49, -9.7566e01),
            (48, 49, -8.1463e01),
            (3, 10, 2.8141e01),
            (3, 13, -9.9709e01),
            (10, 13, -2.8225e01),
            (3, 19, 2.8228e01),
            (3, 28, -2.5487e01),
            (3, 30, 9.0300e01),
            (28, 30, -2.8225e01),
            (3, 35, -1.8370e01),
            (3, 40, -2.1312e01),
            (3, 43, 7.5507e01),
            (40, 43, -2.8225e01),
            (3, 48, 2.7539e01),
            (4, 10, 2.6350e01),
            (4, 14, -9.9709e01),
            (10, 14, -2.6427e01),
            (4, 19, 2.6427e01),
            (4, 23, -1.0000e02),
            (19, 23, -2.6427e01),
            (4, 28, -2.3863e01),
            (4, 31, 9.0300e01),
            (28, 31, -2.6427e01),
            (4, 35, -1.7205e01),
            (4, 38, 6.5140e01),
            (35, 38, -2.6427e01),
            (4, 40, -1.9971e01),
            (4, 44, 7.5507e01),
            (40, 44, -2.6427e01),
            (4, 48, 2.5757e01),
            (5, 10, 9.9709e01),
            (5, 15, -9.9709e01),
            (10, 15, -1.0000e02),
            (5, 19, 1.0000e02),
            (5, 24, -1.0000e02),
            (19, 24, -1.0000e02),
            (5, 28, -9.0289e01),
            (5, 32, 9.0300e01),
            (28, 32, -1.0000e02),
            (5, 35, -6.5144e01),
            (5, 40, -7.5509e01),
            (5, 45, 7.5507e01),
            (40, 45, -1.0000e02),
            (5, 48, 9.7565e01),
            (6, 10, -9.9320e01),
            (6, 16, -9.9709e01),
            (10, 16, 9.9610e01),
            (6, 19, -9.9631e01),
            (6, 28, 8.9946e01),
            (6, 33, 9.0300e01),
            (28, 33, 9.9610e01),
            (6, 35, 6.4890e01),
            (6, 40, 7.5199e01),
            (6, 48, -9.7188e01),
            (7, 10, 9.7157e01),
            (7, 19, 9.7417e01),
            (7, 28, -8.7973e01),
            (7, 35, -6.3446e01),
            (7, 39, 6.5140e01),
            (35, 39, -9.7431e01),
            (7, 40, -7.3586e01),
            (7, 48, 9.5052e01),
            (8, 10, -2.9055e00),
            (8, 19, -2.9605e00),
            (8, 25, -1.0000e02),
            (19, 25, 2.9604e00),
            (8, 28, 2.6517e00),
            (8, 34, 9.0300e01),
            (28, 34, 2.9604e00),
            (8, 35, 1.9168e00),
            (8, 40, 2.2464e00),
            (8, 48, -2.9243e00),
            (9, 10, 2.9135e01),
            (9, 19, 2.9241e01),
            (9, 28, -2.6379e01),
            (9, 35, -1.9046e01),
            (9, 40, -2.2065e01),
            (9, 46, 7.5507e01),
            (40, 46, -2.9232e01),
            (10, 17, -1.0000e02),
            (19, 26, -1.0000e02),
            (10, 18, -1.0000e02),
            (19, 27, -1.0000e02),
            (40, 47, -1.0000e02),
        ]

        # Vectorized approach: extract indices and values
        if off_diag:
            off_diag_array = jnp.array(off_diag)
            i_indices = off_diag_array[:, 0].astype(int)
            j_indices = off_diag_array[:, 1].astype(int)
            values = off_diag_array[:, 2]

            # Set both (i,j) and (j,i) for symmetric matrix
            all_i = jnp.concatenate([i_indices, j_indices])
            all_j = jnp.concatenate([j_indices, i_indices])
            all_values = jnp.concatenate([values, values])

            H = H.at[all_i, all_j].set(all_values)

        # Compute the quadratic objective: 0.5 * y^T H y + c^T y
        return 0.5 * jnp.dot(y, jnp.dot(H, y)) + jnp.dot(c, y)

    @property
    def y0(self):
        """Initial guess (default to zero)."""
        return inexact_asarray(jnp.zeros(self.n))

    @property
    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    @property
    def bounds(self):
        """Variable bounds from SIF file."""
        lower = jnp.full(50, -0.1)  # Default lower bound
        upper = jnp.full(50, 0.1)  # Default upper bound

        # Specific bounds from the SIF file (1-indexed in SIF, 0-indexed here)
        # Variable 1 is fixed at 0.0
        lower = lower.at[0].set(0.0)
        upper = upper.at[0].set(0.0)
        lower = lower.at[1].set(-3.9206e-03)
        upper = upper.at[2].set(9.9999e-02)
        lower = lower.at[3].set(-1.0001e-01)
        upper = upper.at[3].set(9.9990e-02)
        upper = upper.at[4].set(9.9997e-02)
        lower = lower.at[5].set(-9.9994e-02)
        upper = upper.at[5].set(6.1561e-06)
        lower = lower.at[6].set(-3.9119e-03)
        upper = upper.at[6].set(9.9986e-02)
        lower = lower.at[7].set(-1.0001e-01)
        upper = upper.at[7].set(2.5683e-02)
        lower = lower.at[8].set(-9.9987e-02)
        upper = upper.at[8].set(1.0001e-01)
        lower = lower.at[9].set(-9.9988e-02)
        upper = upper.at[9].set(1.0001e-01)
        lower = lower.at[10].set(-1.0001e-01)
        upper = upper.at[10].set(2.8998e-03)
        lower = lower.at[11].set(-9.9952e-02)
        upper = upper.at[11].set(4.7652e-05)
        lower = lower.at[12].set(-4.5551e-05)
        upper = upper.at[12].set(9.9955e-02)
        lower = lower.at[13].set(
            -9.9999e-02
        )  # Variable 14: LO BOUND 14 (no UP BOUND, uses default 0.1)
        # Variable 15 is fixed at 0.0
        lower = lower.at[14].set(0.0)
        upper = upper.at[14].set(0.0)
        lower = lower.at[15].set(-7.2801e-02)  # Line 104 in SIF (variable 16)
        # lower[16] default -0.1, upper[16] default 0.1
        # lower[17] default -0.1, upper[17] default 0.1
        lower = lower.at[17].set(-9.9992e-02)  # Line 104 in SIF (variable 18)
        upper = upper.at[17].set(8.3681e-06)  # Line 105 in SIF (variable 18)
        # lower[18] default -0.1, upper[18] default 0.1
        lower = lower.at[19].set(-9.9956e-02)  # Line 106 in SIF (variable 20)
        upper = upper.at[19].set(4.3809e-05)  # Line 107 in SIF (variable 20)
        # lower[20] default -0.1, upper[20] default 0.1
        lower = lower.at[21].set(-9.9961e-02)  # Line 108 in SIF (variable 22)
        upper = upper.at[21].set(3.9248e-05)  # Line 109 in SIF (variable 22)
        # lower[22] default -0.1, upper[22] default 0.1
        # lower[23] default -0.1, upper[23] default 0.1
        lower = lower.at[24].set(-4.1110e-03)  # Line 110 in SIF (variable 25)
        # lower[25] default -0.1, upper[25] default 0.1
        # lower[26] default -0.1, upper[26] default 0.1
        # lower[27] default -0.1, upper[27] default 0.1
        lower = lower.at[28].set(-9.6988e-02)  # Line 111 in SIF (variable 29)
        upper = upper.at[28].set(1.0002e-01)  # Line 112 in SIF (variable 29)
        # lower[29] default -0.1, upper[29] default 0.1
        # lower[30] default -0.1, upper[30] default 0.1
        lower = lower.at[31].set(-5.8439e-02)  # Line 113 in SIF (variable 32)
        lower = lower.at[32].set(-4.5616e-06)  # Line 114 in SIF (variable 33)
        upper = upper.at[32].set(9.9995e-02)  # Line 115 in SIF (variable 33)
        lower = lower.at[33].set(-9.9999e-02)  # Line 116 in SIF (variable 34)
        upper = upper.at[33].set(7.3117e-07)  # Line 117 in SIF (variable 34)
        lower = lower.at[34].set(-9.9991e-02)  # Line 118 in SIF (variable 35)
        upper = upper.at[34].set(9.3168e-06)  # Line 119 in SIF (variable 35)
        lower = lower.at[35].set(-9.9977e-02)  # Line 120 in SIF (variable 36)
        upper = upper.at[35].set(1.0002e-01)  # Line 121 in SIF (variable 36)
        lower = lower.at[36].set(-9.9984e-02)  # Line 122 in SIF (variable 37)
        upper = upper.at[36].set(1.5812e-05)  # Line 123 in SIF (variable 37)
        # lower[37] default -0.1, upper[37] default 0.1
        lower = lower.at[38].set(-3.9611e-06)  # Line 124 in SIF (variable 39)
        upper = upper.at[38].set(9.9996e-02)  # Line 125 in SIF (variable 39)
        lower = lower.at[39].set(-8.8262e-06)  # Line 126 in SIF (variable 40)
        upper = upper.at[39].set(9.9991e-02)  # Line 127 in SIF (variable 40)
        lower = lower.at[40].set(-1.0001e-01)  # Line 128 in SIF (variable 41)
        upper = upper.at[40].set(9.9986e-02)  # Line 129 in SIF (variable 41)
        # Variable 42 is fixed at 0.0
        lower = lower.at[41].set(0.0)
        upper = upper.at[41].set(0.0)
        lower = lower.at[42].set(-1.9873e-06)  # Line 130 in SIF (variable 43)
        upper = upper.at[42].set(9.9998e-02)  # Line 131 in SIF (variable 43)
        # lower[43] default -0.1, upper[43] default 0.1
        lower = lower.at[44].set(-9.9993e-02)  # Line 132 in SIF (variable 45)
        upper = upper.at[44].set(7.4220e-06)  # Line 133 in SIF (variable 45)
        lower = lower.at[45].set(-9.9999e-02)  # Line 134 in SIF (variable 46)
        upper = upper.at[45].set(8.2308e-07)  # Line 135 in SIF (variable 46)
        lower = lower.at[46].set(-3.0424e-06)  # Line 136 in SIF (variable 47)
        upper = upper.at[46].set(9.9997e-02)  # Line 137 in SIF (variable 47)
        lower = lower.at[47].set(-9.9985e-02)  # Line 138 in SIF (variable 48)
        upper = upper.at[47].set(1.5119e-05)  # Line 139 in SIF (variable 48)
        lower = lower.at[48].set(-1.0004e-01)  # Line 140 in SIF (variable 49)
        upper = upper.at[48].set(2.4305e-02)  # Line 141 in SIF (variable 49)
        # Variable 50 is fixed at 0.0
        lower = lower.at[49].set(0.0)
        upper = upper.at[49].set(0.0)

        return lower, upper

    @property
    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        # From SIF file comment
        return jnp.array(-5.519814e-5)
