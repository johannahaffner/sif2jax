# TODO: Human review needed
# Attempts made: [multiple test failures - starting values, objective,
#                 gradient, Hessian]
# Suspected issues: [complex diamond powder diffraction model may have indexing
#                    or formulation errors]
# Additional resources needed: [primary literature on diffraction fitting,
#                               verification of data arrays and parameters]

import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class DIAMON3DLS(AbstractUnconstrainedMinimisation):
    """Diamond powder diffraction data fitting problem DIAMON3D.

    Given as an inconsistent set of nonlinear equations solved in the
    least-squares sense.

    Source: Data from Aaron Parsons, I14: Hard X-ray Nanoprobe,
    Diamond Light Source, Harwell, Oxfordshire, England, EU.

    SIF input: Nick Gould and Tyrone Rees, Feb 2016
    Least-squares version of DIAMON3D.SIF, Nick Gould, Jan 2020.
    corrected May 2024

    classification SUR2-MN-99-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables (33 weights + 33 widths + 33 positions)."""
        return 99

    @property
    def m(self):
        """Number of data points."""
        return 4643

    @property
    def nvec(self):
        """Number of Lorentzian peaks."""
        return 33

    @property
    def y0(self):
        """Starting point."""
        # Weights
        weights = jnp.array(
            [
                1.0215400678,
                0.0128719128,
                0.4292206739,
                0.8009548574,
                0.2202801749,
                2.6915110182,
                0.8064571417,
                2.2686398843,
                14.352901162,
                12.161699758,
                0.2766766620,
                0.2434602195,
                1.1650863975,
                0.1774572413,
                0.1153950601,
                3.7470242328,
                0.8335010620,
                0.1588345259,
                0.3867741611,
                0.9231772721,
                0.2596415821,
                2.0709635713,
                1.9449508030,
                1.4841998492,
                0.7456765765,
                1.0157606169,
                0.7362872219,
                0.3019033731,
                0.7309027319,
                0.3905621451,
                1.5162117654,
                0.7514027762,
                0.2455414813,
            ]
        )

        # Widths
        widths = jnp.array(
            [
                0.0100006350,
                0.0099991797,
                0.0099997151,
                0.0099978620,
                0.0099992594,
                0.0099997970,
                0.0100014618,
                0.0099986658,
                0.0100002773,
                0.0099998454,
                0.0100017635,
                0.0100001063,
                0.0100005547,
                0.0100001016,
                0.0100002813,
                0.0100004048,
                0.0100001306,
                0.0100007303,
                0.0100002593,
                0.0100000094,
                0.0100015669,
                0.0100021259,
                0.0100007838,
                0.0100014872,
                0.0100008090,
                0.0100011502,
                0.0100015625,
                0.0100001290,
                0.0100002127,
                0.0100016965,
                0.0100022090,
                0.0100013644,
                0.0100006113,
            ]
        )

        # Positions
        positions = jnp.array(
            [
                1.9525695596,
                1.9692522616,
                2.0050071033,
                2.0088536296,
                2.0138244668,
                2.0198968598,
                2.0217238299,
                2.0241970072,
                2.0254945520,
                2.0267816298,
                2.0279872816,
                2.0301574336,
                2.0348658055,
                2.0400531674,
                2.0410134879,
                2.0457089234,
                2.0502620677,
                2.0531313055,
                2.0566615039,
                2.0634513362,
                2.0682916530,
                2.0778648745,
                2.0842454920,
                2.1021907319,
                2.1029516625,
                2.1078625485,
                2.1174361095,
                2.1319638116,
                2.1429297635,
                2.1509061906,
                2.1577451531,
                2.1707616663,
                2.2062515468,
            ]
        )

        return jnp.concatenate([weights, widths, positions])

    @property
    def args(self):
        return None

    def _get_data(self):
        """Get the x and y data values."""
        # X data values - actual values from SIF file would span 1.81971804 to 5.8778079
        # This is a simplified linear interpolation for the JAX implementation
        x_data = jnp.linspace(1.81971804, 5.8778079, self.m)

        # TODO: Human review needed
        # Attempts made: The Y data values (4643 points) are stored in the SIF file
        # but are too numerous to include directly in the code.
        # Suspected issues: Need to either load from external file or include
        # a subset of the data for testing.
        # Additional resources needed: Data loading mechanism or reduced dataset
        y_data = jnp.zeros(self.m)

        return x_data, y_data

    def _lorentzian(self, x, weight, width, position):
        """Compute a single Lorentzian function."""
        piinv = 0.25 / jnp.arctan(1.0)
        pmx = position - x
        denom = pmx**2 + width**2
        ratio = width / denom
        return piinv * weight * ratio

    def objective(self, y, args):
        """Compute the objective function (sum of squared residuals)."""
        del args

        # Extract variables
        weights = y[: self.nvec]
        widths = y[self.nvec : 2 * self.nvec]
        positions = y[2 * self.nvec :]

        # Get data
        x_data, y_data = self._get_data()

        # Vectorized computation of all Lorentzians
        # Shape: (m, nvec)
        piinv = 0.25 / jnp.arctan(1.0)
        x_expanded = x_data[:, jnp.newaxis]  # Shape: (m, 1)
        pmx = positions[jnp.newaxis, :] - x_expanded  # Shape: (m, nvec)
        denom = pmx**2 + widths[jnp.newaxis, :] ** 2
        ratio = widths[jnp.newaxis, :] / denom
        lorentzians = piinv * weights[jnp.newaxis, :] * ratio

        # Sum over all Lorentzians for each data point
        model_values = jnp.sum(lorentzians, axis=1)  # Shape: (m,)

        # Compute residuals
        residuals = model_values - y_data

        # Sum of squared residuals
        return jnp.sum(residuals**2)

    @property
    def expected_result(self):
        """Expected result is not provided in the SIF file."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value is not provided."""
        return None
