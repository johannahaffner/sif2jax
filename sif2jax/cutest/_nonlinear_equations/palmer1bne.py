from jax import numpy as jnp
from jaxtyping import Array, Float

from ..._problem import AbstractNonlinearEquations


class PALMER1BNE(AbstractNonlinearEquations):
    """A nonlinear least squares problem with bounds
    arising from chemical kinetics.

    model: H-N=N=N TZVP+MP2
    fitting Y to A2 X**2 + A4 X**4
                 + B / ( C + X**2 ), B, C nonnegative.

    Source:
    M. Palmer, Edinburgh, private communication.

    SIF input: Nick Gould, 1990.
    Bound-constrained nonlinear equations version: Nick Gould, June 2019.

    classification NOR2-RN-4-35
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def residual(self, y, args) -> Float[Array, "35"]:
        """Residual function for the nonlinear equations."""
        a2, a4, b, c = y

        # X data (radians)
        x_data = jnp.array(
            [
                -1.788963,
                -1.745329,
                -1.658063,
                -1.570796,
                -1.483530,
                -1.396263,
                -1.308997,
                -1.218612,
                -1.134464,
                -1.047198,
                -0.872665,
                -0.698132,
                -0.523599,
                -0.349066,
                -0.174533,
                0.0000000,
                1.788963,
                1.745329,
                1.658063,
                1.570796,
                1.483530,
                1.396263,
                1.308997,
                1.218612,
                1.134464,
                1.047198,
                0.872665,
                0.698132,
                0.523599,
                0.349066,
                0.174533,
                -1.8762289,
                -1.8325957,
                1.8762289,
                1.8325957,
            ]
        )

        # Y data (KJmol-1)
        y_data = jnp.array(
            [
                78.596218,
                65.77963,
                43.96947,
                27.038816,
                14.6126,
                6.2614,
                1.538330,
                0.000000,
                1.188045,
                4.6841,
                16.9321,
                33.6988,
                52.3664,
                70.1630,
                83.4221,
                88.3995,
                78.596218,
                65.77963,
                43.96947,
                27.038816,
                14.6126,
                6.2614,
                1.538330,
                0.000000,
                1.188045,
                4.6841,
                16.9321,
                33.6988,
                52.3664,
                70.1630,
                83.4221,
                108.18086,
                92.733676,
                108.18086,
                92.733676,
            ]
        )

        # Model: Y = A2*X^2 + A4*X^4 + B/(C + X^2)
        x_sqr = x_data * x_data
        x_quart = x_sqr * x_sqr
        model = a2 * x_sqr + a4 * x_quart + b / (c + x_sqr)

        # Residuals
        return model - y_data

    def y0(self) -> Float[Array, "4"]:
        """Initial guess for the optimization problem."""
        return jnp.array([1.0, 1.0, 1.0, 1.0])

    def args(self):
        """Additional arguments for the residual function."""
        return None

    def expected_result(self) -> Float[Array, "4"] | None:
        """Expected result of the optimization problem."""
        # The SIF file doesn't provide a solution
        return None

    def expected_objective_value(self) -> Float[Array, ""] | None:
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)

    def num_constraints(self) -> tuple[int, int, int]:
        """Returns the number of constraints."""
        # 35 equality constraints (the residuals) + 2 finite bounds
        # (lower bounds on B and C)
        num_equalities = 35
        num_inequalities = 0
        num_bounds = 2  # 2 finite lower bounds on B and C
        return num_equalities, num_inequalities, num_bounds

    def bounds(self) -> tuple[Float[Array, "4"], Float[Array, "4"]]:
        """Bounds on variables."""
        # Lower bounds: B and C have lower bounds of 0.00001
        lower = jnp.array([-jnp.inf, -jnp.inf, 0.00001, 0.00001])
        # Upper bounds: all inf
        upper = jnp.array([jnp.inf, jnp.inf, jnp.inf, jnp.inf])
        return lower, upper
