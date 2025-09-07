import jax
import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class ECKERLE4LS(AbstractUnconstrainedMinimisation):
    """ECKERLE4LS - Nonlinear Least-Squares problem (NIST dataset).

    This problem involves a nonlinear least squares fit to a Gaussian peak function,
    arising from a circular interference transmittance study.

    Source: Problem 7 from
    NIST nonlinear least squares test set
    http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml

    SIF input: Ph. Toint, April 1997.

    Classification: SUR2-MN-3-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Allow selecting which starting point to use (0-based indexing)
    start_point: int = 0  # 0 or 1
    _allowed_start_points = frozenset({0, 1})

    def __check_init__(self):
        if self.start_point not in self._allowed_start_points:
            allowed = self._allowed_start_points
            msg = f"start_point must be in {allowed}, got {self.start_point}"
            raise ValueError(msg)

    def objective(self, y, args):
        del args

        # Extract parameters
        b1, b2, b3 = y

        # NIST dataset: x and y values
        x_data = jnp.array(
            [
                400.0,
                405.0,
                410.0,
                415.0,
                420.0,
                425.0,
                430.0,
                435.0,
                436.5,
                438.0,
                439.5,
                441.0,
                442.5,
                444.0,
                445.5,
                447.0,
                448.5,
                450.0,
                451.5,
                453.0,
                454.5,
                456.0,
                457.5,
                459.0,
                460.5,
                462.0,
                463.5,
                465.0,
                470.0,
                475.0,
                480.0,
                485.0,
                490.0,
                495.0,
                500.0,
            ]
        )

        y_data = jnp.array(
            [
                0.0001575,
                0.0001699,
                0.0002350,
                0.0003102,
                0.0004917,
                0.0008710,
                0.0017418,
                0.0046400,
                0.0065895,
                0.0097302,
                0.0149002,
                0.0237310,
                0.0401683,
                0.0712559,
                0.1264458,
                0.2073413,
                0.2902366,
                0.3445623,
                0.3698049,
                0.3668534,
                0.3106727,
                0.2078154,
                0.1164354,
                0.0616764,
                0.0337200,
                0.0194023,
                0.0117831,
                0.0074357,
                0.0022732,
                0.0008800,
                0.0004579,
                0.0002345,
                0.0001586,
                0.0001143,
                0.0000710,
            ]
        )

        # Model function: b1/b2 * exp[-0.5*((x-b3)/b2)^2]
        def model_func(x):
            exponent = -0.5 * ((x - b3) / b2) ** 2
            return (b1 / b2) * jnp.exp(exponent)

        # Calculate model predictions for all x values at once
        predictions = jax.vmap(model_func)(x_data)

        # Compute residuals
        residuals = predictions - y_data

        # Return the sum of squared residuals
        return jnp.sum(residuals**2)

    @property
    def y0(self):
        # Two starting points from the SIF file
        if self.start_point == 0:
            # Starting point 1: b1 = 1.0, b2 = 10.0, b3 = 500.0
            return jnp.array([1.0, 10.0, 500.0])
        else:  # self.start_point == 1
            # Starting point 2: b1 = 1.5, b2 = 5.0, b3 = 450.0
            return jnp.array([1.5, 5.0, 450.0])

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        # Should use certified values from NIST
        return None

    @property
    def expected_objective_value(self):
        # Should use certified minimum value from NIST
        return None
