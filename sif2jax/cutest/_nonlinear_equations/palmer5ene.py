from jax import numpy as jnp
from jaxtyping import Array, Float

from ..._problem import AbstractNonlinearEquations


class PALMER5ENE(AbstractNonlinearEquations):
    """A nonlinear least squares problem
    arising from chemical kinetics.

    model: H-N=C=Se TZVP + MP2
    fitting Y to A0 T_0 + A2 T_2 + A4 T_4 + A6 T_6 + A8 T_8 +
                 A10 T_10 + L * EXP( -K X**2 )
    where T_i is the i-th (shifted) Chebyshev polynomial

    Source:
    M.  Palmer, Edinburgh, private communication.

    SIF input: Nick Gould, 1992.
    Bound-constrained nonlinear equations version: Nick Gould, June 2019.

    classification NOR2-RN-8-12
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def residual(self, y, args) -> Float[Array, "12"]:
        """Residual function for the nonlinear equations."""
        a0, a2, a4, a6, a8, a10, k, l = y

        # X data (radians) - only indices 12-23 from original problem
        x_data = jnp.array(
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

        # Y data (KJmol-1)
        y_data = jnp.array(
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

        # Compute shifted Chebyshev polynomials
        # T_0 = 1
        # Y = 2*(X - A)/(B - A) - 1 where A = -pi/2, B = pi/2
        # So Y = 2*X/pi
        pi = jnp.pi
        a_val = -pi / 2
        b_val = pi / 2
        diff = b_val - a_val  # = pi

        # Model values
        residuals = []
        for i in range(12):
            x = x_data[i]

            # Compute shifted Chebyshev polynomials
            t0 = 1.0
            y_shift = 2.0 * (x - a_val) / diff - 1.0  # = 2*x/pi - 1
            t1 = y_shift

            # T_2 through T_10 via recursion
            t = [t0, t1]
            for j in range(2, 11):  # j = 2, 3, ..., 10
                t_j = 2.0 * y_shift * t[j - 1] - t[j - 2]
                t.append(t_j)

            # Model: Y = A0*T_0 + A2*T_2 + A4*T_4 + A6*T_6 + A8*T_8 + A10*T_10
            #          + L*exp(-K*X^2)
            model = (
                a0 * t[0]
                + a2 * t[2]
                + a4 * t[4]
                + a6 * t[6]
                + a8 * t[8]
                + a10 * t[10]
                + l * jnp.exp(-k * x * x)
            )

            residuals.append(model - y_data[i])

        return jnp.array(residuals)

    def y0(self) -> Float[Array, "8"]:
        """Initial guess for the optimization problem."""
        # From START POINT in SIF
        return jnp.array(
            [
                1.9264e01,  # A0
                -1.7302e00,  # A2
                4.0794e01,  # A4
                8.3021e-01,  # A6
                3.7090e00,  # A8
                -1.7723e-01,  # A10
                10.0,  # K
                1.0,  # L (default)
            ]
        )

    def args(self):
        """Additional arguments for the residual function."""
        return None

    def expected_result(self) -> Float[Array, "8"] | None:
        """Expected result of the optimization problem."""
        # The SIF file doesn't provide a solution
        return None

    def expected_objective_value(self) -> Float[Array, ""] | None:
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)

    def num_constraints(self) -> tuple[int, int, int]:
        """Returns the number of constraints."""
        # 12 equality constraints (the residuals) + 1 finite bound (lower bound on K)
        num_equalities = 12
        num_inequalities = 0
        num_bounds = 1  # 1 finite lower bound on K
        return num_equalities, num_inequalities, num_bounds

    def bounds(self) -> tuple[Float[Array, "8"], Float[Array, "8"]]:
        """Bounds on variables."""
        # Lower bounds: K has implicit lower bound (likely 0)
        lower = jnp.array(
            [-jnp.inf, -jnp.inf, -jnp.inf, -jnp.inf, -jnp.inf, -jnp.inf, 0.0, -jnp.inf]
        )
        # Upper bounds: all inf
        upper = jnp.array(
            [jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf]
        )
        return lower, upper
