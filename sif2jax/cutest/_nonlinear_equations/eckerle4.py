import jax.numpy as jnp
from jax import Array

from ..._misc import inexact_asarray
from ..._problem import AbstractNonlinearEquations


class ECKERLE4(AbstractNonlinearEquations):
    """NIST Data fitting problem ECKERLE4 as nonlinear equations.

    Fit: y = (b1/b2) * exp[-0.5*((x-b3)/b2)**2] + e

    Source: Problem from the NIST nonlinear regression test set
    http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml

    Reference: Eckerle, K., NIST (197?).
    Circular Interference Transmittance Study.

    SIF input: Nick Gould and Tyrone Rees, Oct 2015

    Classification: NOR2-MN-3-35
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0, 1})

    # Problem parameters
    M: int = 35  # Number of data points
    N: int = 3  # Number of variables

    @property
    def n(self):
        """Number of variables."""
        return self.N

    @property
    def m(self):
        """Number of equations."""
        return self.M

    def constraint(self, y: Array):
        """Compute the residuals for the data fitting problem."""
        b1, b2, b3 = y

        # Data points (x values)
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

        # Data points (y values)
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

        # Model: y = (b1/b2) * exp[-0.5*((x-b3)/b2)**2]
        diff = x_data - b3
        exponent = -0.5 * (diff / b2) ** 2
        model_values = (b1 / b2) * jnp.exp(exponent)

        # Residuals: model - data
        residuals = model_values - y_data

        return residuals, None

    @property
    def y0(self):
        """Initial guess."""
        if self.y0_iD == 0:
            # START1
            return inexact_asarray(jnp.array([1.0, 10.0, 500.0]))
        else:
            # START2
            return inexact_asarray(jnp.array([1.5, 5.0, 450.0]))

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def bounds(self):
        """No explicit bounds."""
        return None

    @property
    def expected_result(self):
        """Expected optimal solution."""
        # Not provided in SIF file
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        # Least squares problems are bounded below by zero
        return jnp.array(0.0)
