"""DMN15102LS - Diamond Light Source powder diffraction data fitting (least squares).

A least squares problem fitting X-ray diffraction data using
2-parameter Lorentzian peaks.

Source: Data from Aaron Parsons, I14: Hard X-ray Nanoprobe,
Diamond Light Source, Harwell, Oxfordshire, England, EU.

SIF input: Nick Gould and Tyrone Rees, Feb 2016, corrected May 2024

Classification: SUR2-MN-66-0
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from ..._problem import AbstractUnconstrainedMinimisation


def _load_dmn15102ls_data():
    """Load problem data from NPZ file."""
    data_file = Path(__file__).parent / "data" / "dmn15102ls.npz"
    return np.load(data_file)


# Load data once at module level
_DMN15102LS_DATA = _load_dmn15102ls_data()


class DMN15102LS(AbstractUnconstrainedMinimisation):
    """DMN15102LS Diamond Light Source powder diffraction fitting (least squares).

    Fits X-ray powder diffraction data using a sum of Lorentzian peaks
    by minimizing the sum of squared residuals.

    Variables (66 total):
    - weights[0:33]: Peak weights
    - widths[33:66]: Peak widths

    Objective: Sum of squared residuals over 4643 data points.

    Classification: SUR2-MN-66-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0, 1})

    @property
    def n(self):
        """Number of variables."""
        return int(_DMN15102LS_DATA["n"])

    def objective(self, y, args):
        """Compute the sum of squared residuals.

        Minimizes: sum((model(x_i) - y_i)^2)
        where model is the sum of Lorentzian peaks.
        """
        del args
        nvec = int(_DMN15102LS_DATA["nvec"])
        del nvec  # Loaded for consistency but not used

        # Variables are interleaved: weight1, width1, weight2, width2, ...
        weights = y[0::2]  # Even indices: 0, 2, 4, ...
        widths = y[1::2]  # Odd indices: 1, 3, 5, ...

        # Load data
        x_data = jnp.array(_DMN15102LS_DATA["x_data"])
        y_data = jnp.array(_DMN15102LS_DATA["y_data"])
        positions = jnp.array(_DMN15102LS_DATA["positions"])

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
            return jnp.array(_DMN15102LS_DATA["start1"])
        else:
            return jnp.array(_DMN15102LS_DATA["start2"])

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
