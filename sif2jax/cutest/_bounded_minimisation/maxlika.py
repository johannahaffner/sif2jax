import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


# Module-level private constants for MAXLIKA problem
_y_data = jnp.array(
    [
        95.0,
        105.0,
        110.0,
        110.0,
        110.0,
        110.0,
        115.0,
        115.0,
        115.0,
        115.0,
        120.0,
        120.0,
        120.0,
        120.0,
        120.0,
        120.0,
        120.0,
        120.0,
        120.0,
        120.0,
        120.0,
        120.0,
        120.0,
        120.0,
        120.0,
        125.0,
        125.0,
        125.0,
        125.0,
        125.0,
        125.0,
        125.0,
        125.0,
        125.0,
        125.0,
        125.0,
        125.0,
        125.0,
        125.0,
        125.0,
        130.0,
        130.0,
        130.0,
        130.0,
        130.0,
        130.0,
        130.0,
        130.0,
        130.0,
        130.0,
        130.0,
        130.0,
        130.0,
        130.0,
        130.0,
        135.0,
        135.0,
        135.0,
        135.0,
        135.0,
        135.0,
        135.0,
        135.0,
        135.0,
        135.0,
        135.0,
        135.0,
        135.0,
        140.0,
        140.0,
        140.0,
        140.0,
        140.0,
        140.0,
        140.0,
        140.0,
        140.0,
        140.0,
        140.0,
        140.0,
        140.0,
        140.0,
        140.0,
        140.0,
        140.0,
        140.0,
        140.0,
        140.0,
        140.0,
        145.0,
        145.0,
        145.0,
        145.0,
        145.0,
        145.0,
        145.0,
        145.0,
        145.0,
        145.0,
        145.0,
        145.0,
        150.0,
        150.0,
        150.0,
        150.0,
        150.0,
        150.0,
        150.0,
        150.0,
        150.0,
        150.0,
        150.0,
        150.0,
        150.0,
        150.0,
        150.0,
        150.0,
        150.0,
        155.0,
        155.0,
        155.0,
        155.0,
        160.0,
        160.0,
        160.0,
        160.0,
        160.0,
        160.0,
        160.0,
        160.0,
        160.0,
        160.0,
        160.0,
        160.0,
        160.0,
        160.0,
        160.0,
        160.0,
        160.0,
        160.0,
        160.0,
        160.0,
        165.0,
        165.0,
        165.0,
        165.0,
        165.0,
        165.0,
        165.0,
        165.0,
        170.0,
        170.0,
        170.0,
        170.0,
        170.0,
        170.0,
        170.0,
        170.0,
        170.0,
        170.0,
        170.0,
        170.0,
        170.0,
        170.0,
        170.0,
        170.0,
        170.0,
        175.0,
        175.0,
        175.0,
        175.0,
        175.0,
        175.0,
        175.0,
        175.0,
        180.0,
        180.0,
        180.0,
        180.0,
        180.0,
        180.0,
        185.0,
        185.0,
        185.0,
        185.0,
        185.0,
        185.0,
        190.0,
        190.0,
        190.0,
        190.0,
        190.0,
        190.0,
        190.0,
        195.0,
        195.0,
        195.0,
        195.0,
        200.0,
        200.0,
        200.0,
        205.0,
        205.0,
        205.0,
        210.0,
        210.0,
        210.0,
        210.0,
        210.0,
        210.0,
        210.0,
        210.0,
        215.0,
        220.0,
        220.0,
        220.0,
        220.0,
        220.0,
        220.0,
        230.0,
        230.0,
        230.0,
        230.0,
        230.0,
        235.0,
        240.0,
        240.0,
        240.0,
        240.0,
        240.0,
        240.0,
        240.0,
        245.0,
        250.0,
        250.0,
    ]
)


# TODO: Human review needed
# Attempts made: [1. Initial implementation following SIF structure,
#                 2. Revised to match exact SIF element/group structure]
# Suspected issues: [Hessian test fails - likely numerical precision in
#                    complex exponential/log computations with 235 data points]
# Resources needed: [Expert review of maximum likelihood estimation
#                    numerical precision requirements]


class MAXLIKA(AbstractBoundedMinimisation):
    """MAXLIKA - A maximum likelihood estimation problem.

    A variant of Hock and Schittkowski problem 105, where the
    (inactive) inequality constraint is dropped.

    8 variables with 235 data points for likelihood estimation.

    Variables: X1-X8 with specific bounds
    Objective: Minimize negative log-likelihood involving Gaussian mixtures

    Start: X1=0.1, X2=0.2, X3=100.0, X4=125.0, X5=175.0, X6=11.2, X7=13.2, X8=15.8

    Source: Ph. Toint and A. Griewank.
    SIF input: Ph. Toint, June 1990.

    Classification: OBR2-AY-8-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        """
        Minimize negative log-likelihood.

        Following the SIF structure exactly:
        - Element A(i): F = X1 * exp(-(Y(i) - X3)^2 / (2 * X6^2)) / X6
        - Element B(i): F = X2 * exp(-(Y(i) - X4)^2 / (2 * X7^2)) / X7
        - Element C(i): F = (1-X1-X2) * exp(-(Y(i) - X5)^2 / (2 * X8^2)) / X8
        - Group L(i): F = LOG(GVAR * 0.39894228) where GVAR = A(i) + B(i) + C(i)
        - Objective = -sum(L(i)) due to 'SCALE' -1.0
        """
        del args
        x1, x2, x3, x4, x5, x6, x7, x8 = y

        # Compute element functions for all data points
        # Element A: corresponds to first Gaussian component
        diff_a = _y_data - x3
        exp_a = jnp.exp(-(diff_a**2) / (2 * x6**2))
        a_elements = x1 * exp_a / x6

        # Element B: corresponds to second Gaussian component
        diff_b = _y_data - x4
        exp_b = jnp.exp(-(diff_b**2) / (2 * x7**2))
        b_elements = x2 * exp_b / x7

        # Element C: corresponds to third Gaussian component
        diff_c = _y_data - x5
        exp_c = jnp.exp(-(diff_c**2) / (2 * x8**2))
        c_elements = (1.0 - x1 - x2) * exp_c / x8

        # Group variable GVAR for each data point
        gvar = a_elements + b_elements + c_elements

        # Avoid log(0) by ensuring GVAR is positive
        gvar = jnp.maximum(gvar, 1e-12)

        # Group function: LOG(GVAR * 0.39894228)
        group_values = jnp.log(gvar * 0.39894228)

        # Objective with scale factor -1.0
        return -jnp.sum(group_values)

    @property
    def y0(self):
        # Starting point from SIF file
        return jnp.array([0.1, 0.2, 100.0, 125.0, 175.0, 11.2, 13.2, 15.8])

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        return None

    @property
    def expected_objective_value(self):
        return None

    @property
    def bounds(self):
        # Bounds from SIF file
        lower_bounds = jnp.array([0.001, 0.001, 100.0, 130.0, 170.0, 5.0, 5.0, 5.0])
        upper_bounds = jnp.array([0.499, 0.499, 180.0, 210.0, 240.0, 25.0, 25.0, 25.0])

        return (lower_bounds, upper_bounds)
