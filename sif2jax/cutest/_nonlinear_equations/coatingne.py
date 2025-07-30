import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


# TODO: Human review needed
# Attempts made: Vectorized the residual computation to avoid for-loops
# Suspected issues: Large Jacobian differences (max 13.45 at element 7109)
#   - Element 7109 is Jac[53,7] which is dF(54)/dX8
#   - Our implementation gives 0.0 (correct based on SIF file analysis)
#   - Pycutest apparently expects ~13.45
# Resources needed: Access to pycutest source or MINPACK-2 original formulation
class COATINGNE(AbstractNonlinearEquations):
    """
    The MINPACK 2 Coating Thickness Standardization problem (section 3.3)
    Nonlinear-equation formulation of COATING.SIF

    Source:
    The MINPACK-2 test problem collection,
    Brett M, Averick, Richard G. Carter, Jorge J. More and Guo-iang Xue,
    Mathematics and Computer Science Division,
    Preprint MCS-P153-0692, June 1992

    SIF input: Nick Gould, Jan 2020

    classification NOR2-MN-134-252
    """

    m: int = 252
    n: int = 134
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Data values
    scale1: float = 4.08
    scale2: float = 0.417

    # Eta values - first set of 63 values for eta1
    eta1_data = jnp.array(
        [
            0.7140,
            0.7169,
            0.7232,
            0.7151,
            0.6848,
            0.7070,
            0.7177,
            0.7073,
            0.6734,
            0.7174,
            0.7125,
            0.6947,
            0.7121,
            0.7166,
            0.6894,
            0.6897,
            0.7024,
            0.7026,
            0.6800,
            0.6957,
            0.6987,
            0.7111,
            0.7097,
            0.6809,
            0.7139,
            0.7046,
            0.6950,
            0.7032,
            0.7019,
            0.6975,
            0.6955,
            0.7056,
            0.6965,
            0.6848,
            0.6995,
            0.6105,
            0.6027,
            0.6084,
            0.6081,
            0.6057,
            0.6116,
            0.6052,
            0.6136,
            0.6032,
            0.6081,
            0.6092,
            0.6122,
            0.6157,
            0.6191,
            0.6169,
            0.5483,
            0.5371,
            0.5576,
            0.5521,
            0.5495,
            0.5499,
            0.4937,
            0.5092,
            0.5433,
            0.5018,
            0.5363,
            0.4977,
            0.5296,
        ]
    )

    # Eta values - second set of 63 values for eta2
    eta2_data = jnp.array(
        [
            5.145,
            5.241,
            5.389,
            5.211,
            5.154,
            5.105,
            5.191,
            5.013,
            5.582,
            5.208,
            5.142,
            5.284,
            5.262,
            6.838,
            6.215,
            6.817,
            6.889,
            6.732,
            6.717,
            6.468,
            6.776,
            6.574,
            6.465,
            6.090,
            6.350,
            4.255,
            4.154,
            4.211,
            4.287,
            4.104,
            4.007,
            4.261,
            4.150,
            4.040,
            4.155,
            5.086,
            5.021,
            5.040,
            5.247,
            5.125,
            5.136,
            4.949,
            5.253,
            5.154,
            5.227,
            5.120,
            5.291,
            5.294,
            5.304,
            5.209,
            5.384,
            5.490,
            5.563,
            5.532,
            5.372,
            5.423,
            7.237,
            6.944,
            6.957,
            7.138,
            7.009,
            7.074,
            7.046,
        ]
    )

    # Y data values (126 values)
    y_data = jnp.array(
        [
            9.3636,
            9.3512,
            9.4891,
            9.1888,
            9.3161,
            9.2585,
            9.2913,
            9.3914,
            9.4524,
            9.4995,
            9.4179,
            9.468,
            9.4799,
            11.2917,
            11.5062,
            11.4579,
            11.3977,
            11.3688,
            11.3897,
            11.3104,
            11.3882,
            11.3629,
            11.3149,
            11.2474,
            11.2507,
            8.1678,
            8.1017,
            8.3506,
            8.3651,
            8.2994,
            8.1514,
            8.2229,
            8.1027,
            8.3785,
            8.4118,
            8.0955,
            8.0613,
            8.0979,
            8.1364,
            8.1700,
            8.1684,
            8.0885,
            8.1839,
            8.1478,
            8.1827,
            8.029,
            8.1000,
            8.2579,
            8.2248,
            8.2540,
            6.8518,
            6.8547,
            6.8831,
            6.9137,
            6.8984,
            6.8888,
            8.5189,
            8.5308,
            8.5184,
            8.5222,
            8.5705,
            8.5353,
            8.5213,
            8.3158,
            8.1995,
            8.2283,
            8.1857,
            8.2738,
            8.2131,
            8.2613,
            8.2315,
            8.2078,
            8.2996,
            8.3026,
            8.0995,
            8.2990,
            9.6753,
            9.6687,
            9.5704,
            9.5435,
            9.6780,
            9.7668,
            9.7827,
            9.7844,
            9.7011,
            9.8006,
            9.7610,
            9.7813,
            7.3073,
            7.2572,
            7.4686,
            7.3659,
            7.3587,
            7.3132,
            7.3542,
            7.2339,
            7.4375,
            7.4022,
            10.7914,
            10.6554,
            10.7359,
            10.7583,
            10.7735,
            10.7907,
            10.6465,
            10.6994,
            10.7756,
            10.7402,
            10.6800,
            10.7000,
            10.8160,
            10.6921,
            10.8677,
            12.3495,
            12.4424,
            12.4303,
            12.5086,
            12.4513,
            12.4625,
            16.2290,
            16.2781,
            16.2082,
            16.2715,
            16.2464,
            16.1626,
            16.1568,
        ]
    )

    def starting_point(self) -> Array:
        return jnp.ones(self.n, dtype=jnp.float64)

    def num_residuals(self) -> int:
        return self.m

    def residual(self, y: Array, args) -> Array:
        """Compute the residuals of the coating thickness problem"""
        x = y  # Use consistent naming with SIF file
        m_div_4 = self.m // 4

        # Get eta values for first M/4 elements
        e1 = self.eta1_data[:m_div_4]
        e2 = self.eta2_data[:m_div_4]
        e1e2 = e1 * e2

        # Index arrays
        i_vals = jnp.arange(m_div_4)

        # Process first M/4 residuals - vectorized
        # F(i) components from the SIF file
        f_i = (
            x[0]
            + e1 * x[1]
            + e2 * x[2]
            + e1e2 * x[3]
            + x[1] * x[i_vals + 8]
            + x[2] * x[i_vals + m_div_4 + 8]
            + e2 * x[3] * x[i_vals + 8]
            + e1 * x[3] * x[i_vals + m_div_4 + 8]
            + x[3] * x[i_vals + 8] * x[i_vals + m_div_4 + 8]
        )
        residuals_1 = f_i - self.y_data[:m_div_4]

        # Process second M/4 residuals - vectorized
        # F(i2) components from the SIF file
        # IMPORTANT: The SIF file shows that F(I2) still uses X(IP8) and X(I2P8)
        # where IP8 = I+8 and I2P8 = (I+M/4)+8, NOT (I2+8)
        f_i2 = (
            x[4]
            + e1 * x[5]
            + e2 * x[6]
            + e1e2 * x[7]
            + x[5] * x[i_vals + 8]
            + x[6] * x[i_vals + m_div_4 + 8]
            + e2 * x[7] * x[i_vals + 8]
            + e1 * x[7] * x[i_vals + m_div_4 + 8]
            + x[7] * x[i_vals + 8] * x[i_vals + m_div_4 + 8]
        )
        residuals_2 = f_i2 - self.y_data[m_div_4 : 2 * m_div_4]

        # Process third M/4 residuals - vectorized
        # For F(127) to F(189), no Y values in SIF file, so RHS = 0
        residuals_3 = self.scale1 * x[i_vals + 8]

        # Process fourth M/4 residuals - vectorized
        # For F(190) to F(252), no Y values in SIF file, so RHS = 0
        residuals_4 = self.scale2 * x[i_vals + m_div_4 + 8]

        # Concatenate all residuals
        residuals = jnp.concatenate(
            [residuals_1, residuals_2, residuals_3, residuals_4]
        )

        return residuals

    @property
    def y0(self) -> Array:
        """Initial guess for the optimization problem."""
        return self.starting_point()

    @property
    def args(self):
        """Additional arguments for the residual function."""
        return None

    @property
    def expected_result(self) -> Array:
        """Expected result of the optimization problem."""
        # Solution is not provided in the SIF file
        return jnp.ones(self.n, dtype=jnp.float64)

    @property
    def expected_objective_value(self) -> Array:
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)

    def constraint(self, y):
        """Returns the residuals as equality constraints."""
        return self.residual(y, self.args), None

    @property
    def bounds(self) -> tuple[Array, Array] | None:
        """No bounds for this problem."""
        return None
