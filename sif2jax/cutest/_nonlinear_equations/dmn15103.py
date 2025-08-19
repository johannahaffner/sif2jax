"""DMN15103 - Diamond Light Source powder diffraction data fitting (3-parameter).

A nonlinear equations problem fitting X-ray diffraction data using
3-parameter Lorentzian peaks (weight, width, position).

Source: Data from Aaron Parsons, I14: Hard X-ray Nanoprobe,
Diamond Light Source, Harwell, Oxfordshire, England, EU.

SIF input: Nick Gould and Tyrone Rees, Feb 2016, corrected May 2024

Classification: NOR2-MN-99-4643
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from ..._problem import AbstractNonlinearEquations


def _load_dmn15103_data():
    """Load problem data from NPZ file."""
    data_file = Path(__file__).parent / "data" / "dmn15103.npz"
    return np.load(data_file)


# Load data once at module level
_DMN15103_DATA = _load_dmn15103_data()


class DMN15103(AbstractNonlinearEquations):
    """DMN15103 Diamond Light Source powder diffraction fitting (3-parameter).

    Fits X-ray powder diffraction data using a sum of Lorentzian peaks.
    Each peak has 3 parameters: weight, width, and position.

    Variables (99 total):
    - For each peak i: weight_i, width_i, position_i (3*33 = 99)

    Equations (4643): One per data point, fitting the observed intensities.

    Classification: NOR2-MN-99-4643
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0, 1})

    @property
    def n(self):
        """Number of variables."""
        return int(_DMN15103_DATA["n"])

    @property
    def m(self):
        """Number of equations."""
        return int(_DMN15103_DATA["m"])

    def num_residuals(self):
        """Number of residuals (equations)."""
        return int(_DMN15103_DATA["m"])

    def residual(self, y, args):
        """Compute the residuals.

        Each residual is: model(x_i) - y_i = 0
        where model is the sum of Lorentzian peaks.
        """
        del args
        nvec = int(_DMN15103_DATA["nvec"])
        del nvec  # Loaded for consistency but not used

        # Variables are grouped by parameter type: weights, widths, positions
        # y = [weight1, width1, pos1, weight2, width2, pos2, ...]
        weights = y[0::3]  # Indices: 0, 3, 6, ... (every 3rd starting from 0)
        widths = y[1::3]  # Indices: 1, 4, 7, ... (every 3rd starting from 1)
        positions = y[2::3]  # Indices: 2, 5, 8, ... (every 3rd starting from 2)

        # Load data
        x_data = jnp.array(_DMN15103_DATA["x_data"])
        y_data = jnp.array(_DMN15103_DATA["y_data"])

        # Compute Lorentzian peaks
        # Vectorized computation
        x_expanded = x_data[:, jnp.newaxis]  # (m, 1)
        pos_expanded = positions[jnp.newaxis, :]  # (1, nvec)
        weights_expanded = weights[jnp.newaxis, :]  # (1, nvec)
        widths_expanded = widths[jnp.newaxis, :]  # (1, nvec)

        # Compute denominators for all peaks at all points
        denom = (x_expanded - pos_expanded) ** 2 + widths_expanded**2

        # Compute Lorentzian values
        pi_inv = 1.0 / jnp.pi
        lorentzians = weights_expanded * pi_inv * widths_expanded / denom

        # Sum over peaks for each data point
        model = jnp.sum(lorentzians, axis=1)

        # Return residuals
        return model - y_data

    def constraint(self, y):
        """Return constraints as equality constraints.

        For nonlinear equations, the residuals are equality constraints.
        """
        residuals = self.residual(y, self.args)
        return residuals, None  # (equality constraints, inequality constraints)

    @property
    def y0(self):
        """Starting point."""
        if self.y0_iD == 0:
            return jnp.array(_DMN15103_DATA["start1"])
        else:
            return jnp.array(_DMN15103_DATA["start2"])

    @property
    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    @property
    def bounds(self):
        """Variable bounds (none for this problem)."""
        return None

    @property
    def expected_result(self):
        """Expected solution (not available for this problem)."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value (not available for this problem)."""
        return None
