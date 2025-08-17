import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations
from ..data import load_fbrain_data


# Load FBRAINNE specific data
A_coeffs, A_lambdas, B_coeffs, B_lambdas, R_values = load_fbrain_data("fbrainne")


class FBRAINNE(AbstractNonlinearEquations):
    """FBRAINNE - Brain tissue shear stress model as nonlinear equations.

    Match a model of the shear stress in the human brain to data,
    formulated as a set of nonlinear equations.

    Source: an example in
    L.A. Mihai, S. Budday, G.A. Holzapfel, E. Kuhl and A. Goriely.
    "A family of hyperelastic models for human brain tissue",
    Journal of Mechanics and Physics of Solids,
    DOI: 10.1016/j.jmps.2017.05.015 (2017).

    as conveyed by Angela Mihai (U. Cardiff)

    SIF input: Nick Gould, June 2017.
    Bound-constrained nonlinear equations version: Nick Gould, June 2019.

    Classification: NOR2-AN-2-2211
    N = 11 (number of data sets)
    M = 200 (number of discretizations)
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def y0(self):
        return jnp.array([-4.025, -0.0939])

    @property
    def args(self):
        return None

    def constraint(self, y):
        alpha, c0 = y

        # Compute beta exponent
        beta = 2.0 * alpha - 1.0

        # Flatten arrays in column-major order to match SIF constraint ordering
        # SIF iterates: DO J (1 to N), DO I (0 to M) -> (I,J) pairs
        # Cast to same dtype as y to avoid type promotion errors
        a_coeffs_flat = A_coeffs.T.ravel().astype(y.dtype)
        a_lambdas_flat = A_lambdas.T.ravel().astype(y.dtype)
        b_coeffs_flat = B_coeffs.T.ravel().astype(y.dtype)
        b_lambdas_flat = B_lambdas.T.ravel().astype(y.dtype)
        r_values_flat = R_values.T.ravel().astype(y.dtype)

        # Compute element values
        a_values = c0 * a_coeffs_flat * (a_lambdas_flat**beta)
        b_values = c0 * b_coeffs_flat * (b_lambdas_flat**beta)

        # Sum A and B for each (i,j)
        model_values = a_values + b_values

        # Compute residuals (constraints)
        constraints = model_values - r_values_flat

        return constraints, None

    @property
    def bounds(self):
        # SIF bounds: LO 'DEFAULT' -5.0, UP 'DEFAULT' 5.0
        lower = jnp.array([-5.0, -5.0])
        upper = jnp.array([5.0, 5.0])
        return lower, upper

    @property
    def expected_result(self):
        return None

    @property
    def expected_objective_value(self):
        return jnp.array(0.0)
