from jax import numpy as jnp
from jaxtyping import Array, Float

from ..._problem import AbstractNonlinearEquations


class NELSON(AbstractNonlinearEquations):
    """NIST Data fitting problem NELSON given as an inconsistent set of
    nonlinear equations.

    Fit: log[y] = b1 - b2*x1 * exp[-b3*x2] + e

    Source:  Problem from the NIST nonlinear regression test set
      http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml

    Reference: Nelson, W. (1981).
      Analysis of Performance-Degradation Data.
      IEEE Transactions on Reliability. Vol. 2, R-30, No. 2, pp. 149-155.

    SIF input: Nick Gould and Tyrone Rees, Oct 2015

    classification NOR2-MN-3-128
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0, 1})

    def constraint(self, y, args=None) -> tuple[Float[Array, "128"], None]:
        """Constraint function for the nonlinear equations."""
        b1, b2, b3 = y

        # X1 data
        x1_data = jnp.array(
            [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                8.0,
                8.0,
                8.0,
                8.0,
                8.0,
                8.0,
                8.0,
                8.0,
                8.0,
                8.0,
                8.0,
                8.0,
                8.0,
                8.0,
                8.0,
                8.0,
                16.0,
                16.0,
                16.0,
                16.0,
                16.0,
                16.0,
                16.0,
                16.0,
                16.0,
                16.0,
                16.0,
                16.0,
                16.0,
                16.0,
                16.0,
                16.0,
                32.0,
                32.0,
                32.0,
                32.0,
                32.0,
                32.0,
                32.0,
                32.0,
                32.0,
                32.0,
                32.0,
                32.0,
                32.0,
                32.0,
                32.0,
                32.0,
                48.0,
                48.0,
                48.0,
                48.0,
                48.0,
                48.0,
                48.0,
                48.0,
                48.0,
                48.0,
                48.0,
                48.0,
                48.0,
                48.0,
                48.0,
                48.0,
                64.0,
                64.0,
                64.0,
                64.0,
                64.0,
                64.0,
                64.0,
                64.0,
                64.0,
                64.0,
                64.0,
                64.0,
                64.0,
                64.0,
                64.0,
                64.0,
            ]
        )

        # X2 data
        x2_data = jnp.array(
            [
                180.0,
                180.0,
                180.0,
                180.0,
                225.0,
                225.0,
                225.0,
                225.0,
                250.0,
                250.0,
                250.0,
                250.0,
                275.0,
                275.0,
                275.0,
                275.0,
                180.0,
                180.0,
                180.0,
                180.0,
                225.0,
                225.0,
                225.0,
                225.0,
                250.0,
                250.0,
                250.0,
                250.0,
                275.0,
                275.0,
                275.0,
                275.0,
                180.0,
                180.0,
                180.0,
                180.0,
                225.0,
                225.0,
                225.0,
                225.0,
                250.0,
                250.0,
                250.0,
                250.0,
                275.0,
                275.0,
                275.0,
                275.0,
                180.0,
                180.0,
                180.0,
                180.0,
                225.0,
                225.0,
                225.0,
                225.0,
                250.0,
                250.0,
                250.0,
                250.0,
                275.0,
                275.0,
                275.0,
                275.0,
                180.0,
                180.0,
                180.0,
                180.0,
                225.0,
                225.0,
                225.0,
                225.0,
                250.0,
                250.0,
                250.0,
                250.0,
                275.0,
                275.0,
                275.0,
                275.0,
                180.0,
                180.0,
                180.0,
                180.0,
                225.0,
                225.0,
                225.0,
                225.0,
                250.0,
                250.0,
                250.0,
                250.0,
                275.0,
                275.0,
                275.0,
                275.0,
                180.0,
                180.0,
                180.0,
                180.0,
                225.0,
                225.0,
                225.0,
                225.0,
                250.0,
                250.0,
                250.0,
                250.0,
                275.0,
                275.0,
                275.0,
                275.0,
                180.0,
                180.0,
                180.0,
                180.0,
                225.0,
                225.0,
                225.0,
                225.0,
                250.0,
                250.0,
                250.0,
                250.0,
                275.0,
                275.0,
                275.0,
                275.0,
            ]
        )

        # Y data
        y_data = jnp.array(
            [
                15.00,
                17.00,
                15.50,
                16.50,
                15.50,
                15.00,
                16.00,
                14.50,
                15.00,
                14.50,
                12.50,
                11.00,
                14.00,
                13.00,
                14.00,
                11.50,
                14.00,
                16.00,
                13.00,
                13.50,
                13.00,
                13.50,
                12.50,
                12.50,
                12.50,
                12.00,
                11.50,
                12.00,
                13.00,
                11.50,
                13.00,
                12.50,
                13.50,
                17.50,
                17.50,
                13.50,
                12.50,
                12.50,
                15.00,
                13.00,
                12.00,
                13.00,
                12.00,
                13.50,
                10.00,
                11.50,
                11.00,
                9.50,
                15.00,
                15.00,
                15.50,
                16.00,
                13.00,
                10.50,
                13.50,
                14.00,
                12.50,
                12.00,
                11.50,
                11.50,
                6.50,
                5.50,
                6.00,
                6.00,
                18.50,
                17.00,
                15.30,
                16.00,
                13.00,
                14.00,
                12.50,
                11.00,
                12.00,
                12.00,
                11.50,
                12.00,
                6.00,
                6.00,
                5.00,
                5.50,
                12.50,
                13.00,
                16.00,
                12.00,
                11.00,
                9.50,
                11.00,
                11.00,
                11.00,
                10.00,
                10.50,
                10.50,
                2.70,
                2.70,
                2.50,
                2.40,
                13.00,
                13.50,
                16.50,
                13.60,
                11.50,
                10.50,
                13.50,
                12.00,
                7.00,
                6.90,
                8.80,
                7.90,
                1.20,
                1.50,
                1.00,
                1.50,
                13.00,
                12.50,
                16.50,
                16.00,
                11.00,
                11.50,
                10.50,
                10.00,
                7.27,
                7.50,
                6.70,
                7.60,
                1.50,
                1.00,
                1.20,
                1.20,
            ]
        )

        # Compute log of y values
        log_y = jnp.log(y_data)

        # Compute predicted values: log[y] = b1 - b2*x1 * exp[-b3*x2]
        pred = b1 - b2 * x1_data * jnp.exp(-b3 * x2_data)

        # Return residuals as equality constraints
        return pred - log_y, None

    @property
    def n_var(self) -> int:
        """Number of variables (3 parameters)."""
        return 3

    @property
    def n_con(self) -> int:
        """Number of constraints (128 data points)."""
        return 128

    @property
    def y0s(self):
        """Starting points for the problem."""
        y0_0 = jnp.array([2.0, 0.0001, -0.01])
        y0_1 = jnp.array([2.5, 0.000000005, -0.05])
        return {0: y0_0, 1: y0_1}

    @property
    def y0(self):
        """Default starting point."""
        return self.y0s[self.y0_iD]

    @property
    def bounds(self):
        """No bounds for this problem."""
        return None

    @property
    def args(self):
        """No additional arguments."""
        return ()

    @property
    def expected_objective_value(self):
        """Expected objective value (zero for nonlinear equations)."""
        return jnp.array(0.0)

    @property
    def expected_result(self):
        """Expected result is not known analytically."""
        return None
