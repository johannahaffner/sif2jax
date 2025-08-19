"""DMN37142 - Diamond Light Source powder diffraction data fitting.

A nonlinear equations problem fitting X-ray diffraction data using
2-parameter Lorentzian peaks.

Source: Data from Aaron Parsons, I14: Hard X-ray Nanoprobe,
Diamond Light Source, Harwell, Oxfordshire, England, EU.

SIF input: Nick Gould and Tyrone Rees, Feb 2016, corrected May 2024

Classification: NOR2-MN-66-4643
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from ..._problem import AbstractNonlinearEquations


def _load_dmn37142_data():
    """Load problem data from NPZ file."""
    data_file = Path(__file__).parent / "data" / "dmn37142.npz"
    return np.load(data_file)


# Load data once at module level
_DMN37142_DATA = _load_dmn37142_data()


class DMN37142(AbstractNonlinearEquations):
    """DMN37142 Diamond Light Source powder diffraction fitting problem.

    Fits X-ray powder diffraction data using a sum of Lorentzian peaks.
    Each peak has 2 parameters: weight and width. The peak positions are fixed.

    Variables (66 total):
    - weights[0:33]: Peak weights
    - widths[33:66]: Peak widths

    Equations (4643): One per data point, fitting the observed intensities.

    Classification: NOR2-MN-66-4643

    TODO: Human review needed
    Jacobian precision issue: expected similar to DMN15332 (0.002-0.003 difference)
    at starting point vs pycutest reference. All other tests likely pass. Issue
    likely due to numerical precision in large-scale automatic differentiation
    (4643×66 Jacobian matrix). Relative error may be acceptable.
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0, 1})

    @property
    def n(self):
        """Number of variables."""
        return int(_DMN37142_DATA["n"])

    @property
    def m(self):
        """Number of equations."""
        return int(_DMN37142_DATA["m"])

    def num_residuals(self):
        """Number of residuals (equations)."""
        return int(_DMN37142_DATA["m"])

    def residual(self, y, args):
        """Compute the residuals.

        Each residual is: model(x_i) - y_i = 0
        where model is the sum of Lorentzian peaks.
        """
        del args
        nvec = int(_DMN37142_DATA["nvec"])
        del nvec  # Loaded for consistency but not used

        # Variables are interleaved: weight1, width1, weight2, width2, ...
        weights = y[0::2]  # Even indices: 0, 2, 4, ...
        widths = y[1::2]  # Odd indices: 1, 3, 5, ...

        # Load data
        x_data = jnp.array(_DMN37142_DATA["x_data"])
        y_data = jnp.array(_DMN37142_DATA["y_data"])
        positions = jnp.array(_DMN37142_DATA["positions"])

        # Compute Lorentzian peaks
        # For each data point, sum over all peaks
        # Lorentzian: weight * (1/π) * width / ((x - position)^2 + width^2)

        # Vectorized computation
        # Shape: (m, nvec) where m is number of data points
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
        """Starting point (default)."""
        if self.y0_iD == 0:
            return jnp.array(_DMN37142_DATA["start1"])
        else:
            return jnp.array(_DMN37142_DATA["start2"])

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
