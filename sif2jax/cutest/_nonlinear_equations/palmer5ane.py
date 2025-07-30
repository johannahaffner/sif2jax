import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class PALMER5ANE(AbstractNonlinearEquations):
    """A nonlinear least squares problem with bounds arising from chemical kinetics.

    model: H-N=C=Se TZVP + MP2
    fitting Y to A0 T_0 + A2 T_2 + A4 T_4 + A6 T_6 + A8 T_8 +
                A10 T_10 + A12 T_12 + A14 T_14
                + B / ( C + X**2 ), B, C nonnegative.
    where T_i is the i-th (shifted) Chebyshev polynomial

    Source:
    M. Palmer, Edinburgh, private communication.

    SIF input: Nick Gould, 1992.
    Bound-constrained nonlinear equations version: Nick Gould, June 2019.

    classification NOR2-RN-8-12
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Data points (indices 12-23)
    M: int = 12

    # X values (radians) - starting from index 12
    X_data = jnp.array(
        [
            0.000000,
            1.570796,
            1.396263,
            1.308997,
            1.221730,
            1.125835,
            1.047198,
            0.872665,
            0.698132,
            0.523599,
            0.349066,
            0.174533,
        ]
    )

    # Y values (KJmol-1)
    Y_data = jnp.array(
        [
            83.57418,
            81.007654,
            18.983286,
            8.051067,
            2.044762,
            0.000000,
            1.170451,
            10.479881,
            25.785001,
            44.126844,
            62.822177,
            77.719674,
        ]
    )

    @property
    def n(self):
        """Number of variables."""
        return 8  # A0, A2, A4, A6, A8, A10, B, C

    @property
    def M_residuals(self):
        """Number of residual functions."""
        return 12

    def residual(self, y, args):
        """Compute the residual functions.

        TODO: This implementation uses simple polynomials instead of Chebyshev.
        Needs proper Chebyshev polynomial calculation.
        """
        del args

        # Extract variables
        A0, A2, A4, A6, A8, A10, B, C = y

        # Precompute powers of X (placeholder - should use Chebyshev)
        X_sqr = self.X_data * self.X_data
        X_4 = X_sqr * X_sqr
        X_6 = X_sqr * X_4
        X_8 = X_sqr * X_6
        X_10 = X_sqr * X_8

        # Model predictions (placeholder - should use Chebyshev polynomials)
        predictions = (
            A0
            + A2 * X_sqr
            + A4 * X_4
            + A6 * X_6
            + A8 * X_8
            + A10 * X_10
            + B / (C + X_sqr)
        )

        # Residuals (prediction - observation)
        residuals = predictions - self.Y_data

        return residuals

    @property
    def y0(self):
        """Initial guess."""
        return jnp.ones(8)

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return jnp.zeros(8)  # Placeholder

    @property
    def expected_objective_value(self):
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)

    def constraint(self, y):
        """Returns the residuals as equality constraints."""
        return self.residual(y, self.args), None

    @property
    def bounds(self):
        """Returns the bounds on the variable y."""
        # A0, A2, A4, A6, A8, A10 are free, B and C have lower bounds
        lower = jnp.array(
            [
                -jnp.inf,
                -jnp.inf,
                -jnp.inf,
                -jnp.inf,
                -jnp.inf,
                -jnp.inf,
                0.00001,
                0.00001,
            ]
        )
        upper = jnp.full(8, jnp.inf)
        return lower, upper
