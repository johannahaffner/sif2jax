"""DMN37143LS - Diamond Light Source powder diffraction data fitting (3-parameter LS).

A least squares problem fitting X-ray diffraction data using 
3-parameter Lorentzian peaks (weight, width, position).

Source: Data from Aaron Parsons, I14: Hard X-ray Nanoprobe,
Diamond Light Source, Harwell, Oxfordshire, England, EU.

SIF input: Nick Gould and Tyrone Rees, Feb 2016, corrected May 2024

Classification: SUR2-MN-99-0
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from ..._problem import AbstractUnconstrainedMinimisation


def _load_dmn37143ls_data():
    """Load problem data from NPZ file."""
    data_file = Path(__file__).parent / "data" / "dmn37143ls.npz"
    return np.load(data_file)


# Load data once at module level
_DMN37143LS_DATA = _load_dmn37143ls_data()


class DMN37143LS(AbstractUnconstrainedMinimisation):
    """DMN37143LS Diamond Light Source powder diffraction fitting (3-parameter LS).

    Fits X-ray powder diffraction data using a sum of Lorentzian peaks
    by minimizing the sum of squared residuals.

    Variables (99 total):
    - For each peak i: weight_i, width_i, position_i (3*33 = 99)

    Objective: Sum of squared residuals over 4643 data points.

    Classification: SUR2-MN-99-0

    TODO: Human review needed
    Likely gradient/Hessian precision issues similar to corresponding NOR2 version
    DMN37143 (0.001-0.002 differences). Issue likely due to numerical precision
    in large-scale automatic differentiation. Relative error may be acceptable.
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0, 1})

    @property
    def n(self):
        """Number of variables."""
        return int(_DMN37143LS_DATA["n"])

    def objective(self, y, args):
        """Compute the sum of squared residuals.

        Minimizes: sum((model(x_i) - y_i)^2)
        where model is the sum of Lorentzian peaks.
        """
        del args
        nvec = int(_DMN37143LS_DATA["nvec"])
        del nvec  # Loaded for consistency but not used

        # Variables are grouped by parameter type: weights, widths, positions
        # y = [weight1, width1, pos1, weight2, width2, pos2, ...]
        weights = y[0::3]  # Indices: 0, 3, 6, ...
        widths = y[1::3]  # Indices: 1, 4, 7, ...
        positions = y[2::3]  # Indices: 2, 5, 8, ...

        # Load data
        x_data = jnp.array(_DMN37143LS_DATA["x_data"])
        y_data = jnp.array(_DMN37143LS_DATA["y_data"])

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

        # Compute sum of squared residuals
        residuals = model - y_data
        return jnp.sum(residuals**2)

    @property
    def y0(self):
        """Starting point."""
        if self.y0_iD == 0:
            return jnp.array(_DMN37143LS_DATA["start1"])
        else:
            return jnp.array(_DMN37143LS_DATA["start2"])

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
