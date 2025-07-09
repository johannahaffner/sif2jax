from jax import numpy as jnp
from jaxtyping import Array, Float

from ..._problem import AbstractNonlinearEquations


class PALMER7ANE(AbstractNonlinearEquations):
    """A nonlinear least squares problem with bounds
    arising from chemical kinetics.

    model: H-N=C=Se TZVP + MP2
    fitting Y to A0 + A2 X**2 + A4 X**4 + A6 X**6
                 + B / ( C + X**2 ), B, C nonnegative.

    Source:
    M. Palmer, Edinburgh, private communication.

    SIF input: Nick Gould, 1992.
    Bound-constrained nonlinear equations version: Nick Gould, June 2019.

    classification NOR2-RN-6-13
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def residual(self, y, args) -> Float[Array, "13"]:
        """Residual function for the nonlinear equations."""
        a0, a2, a4, a6, b, c = y

        # X data (radians) - indices 12-24
        x_data = jnp.array(
            [
                0.000000,
                0.139626,
                0.261799,
                0.436332,
                0.565245,
                0.512942,
                0.610865,
                0.785398,
                0.959931,
                1.134464,
                1.308997,
                1.483530,
                1.658063,
            ]
        )

        # Y data (KJmol-1)
        y_data = jnp.array(
            [
                4.419446,
                3.564931,
                2.139067,
                0.404686,
                0.000000,
                0.035152,
                0.146813,
                2.718058,
                9.474417,
                26.132221,
                41.451561,
                72.283164,
                117.630959,
            ]
        )

        # Model: Y = A0 + A2*X^2 + A4*X^4 + A6*X^6 + B/(C + X^2)
        x_sqr = x_data * x_data
        x_quart = x_sqr * x_sqr
        x_sext = x_quart * x_sqr
        model = a0 + a2 * x_sqr + a4 * x_quart + a6 * x_sext + b / (c + x_sqr)

        # Residuals
        return model - y_data

    def y0(self) -> Float[Array, "6"]:
        """Initial guess for the optimization problem."""
        return jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    def args(self):
        """Additional arguments for the residual function."""
        return None

    def expected_result(self) -> Float[Array, "6"] | None:
        """Expected result of the optimization problem."""
        # The SIF file doesn't provide a solution
        return None

    def expected_objective_value(self) -> Float[Array, ""] | None:
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)

    def num_constraints(self) -> tuple[int, int, int]:
        """Returns the number of constraints."""
        # 13 equality constraints (the residuals) + 2 finite bounds
        # (lower bounds on B and C)
        num_equalities = 13
        num_inequalities = 0
        num_bounds = 2  # 2 finite lower bounds on B and C
        return num_equalities, num_inequalities, num_bounds

    def bounds(self) -> tuple[Float[Array, "6"], Float[Array, "6"]]:
        """Bounds on variables."""
        # Lower bounds: B and C have lower bounds of 0.00001
        lower = jnp.array([-jnp.inf, -jnp.inf, -jnp.inf, -jnp.inf, 0.00001, 0.00001])
        # Upper bounds: all inf
        upper = jnp.array([jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf])
        return lower, upper
