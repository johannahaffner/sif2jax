import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations
from ..data import load_fbrain_data


# Load FBRAIN data (FBRAIN2NE uses same data as FBRAIN)
A_coeffs, A_lambdas, B_coeffs, B_lambdas, R_values = load_fbrain_data("fbrain")


class FBRAIN2NE(AbstractNonlinearEquations):
    """FBRAIN2NE - Brain tissue shear stress model with 4 parameters as NE.

    Match a model of the shear stress in the human brain to data,
    formulated as a set of nonlinear equations with 4 parameters.

    Source: an example in
    L.A. Mihai, S. Budday, G.A. Holzapfel, E. Kuhl and A. Goriely.
    "A family of hyperelastic models for human brain tissue",
    Journal of Mechanics and Physics of Solids,
    DOI: 10.1016/j.jmps.2017.05.015 (2017).

    as conveyed by Angela Mihai (U. Cardiff)

    SIF input: Nick Gould, June 2017.
    Bound-constrained nonlinear equations version: Nick Gould, June 2019.

    Classification: NOR2-AN-4-2211
    N = 11 (number of data sets)
    M = 200 (number of discretizations)
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0, 1})

    @property
    def y0(self):
        if self.y0_iD == 0:
            return jnp.array([-4.0, -0.1, 4.0, 0.1])
        else:  # y0_iD == 1
            return jnp.array([1.0, 1.0, 1.0, 1.0])

    @property
    def args(self):
        return None

    def constraint(self, y):
        alpha1, c01, alpha2, c02 = y

        # Compute betas
        beta1 = 2.0 * alpha1 - 1.0
        beta2 = 2.0 * alpha2 - 1.0

        # Flatten arrays in column-major order to match SIF constraint ordering
        # SIF iterates: DO J (1 to N), DO I (0 to M) -> (I,J) pairs
        # Cast to same dtype as y to avoid type promotion errors
        a_coeffs_flat = A_coeffs.T.ravel().astype(y.dtype)
        a_lambdas_flat = A_lambdas.T.ravel().astype(y.dtype)
        b_coeffs_flat = B_coeffs.T.ravel().astype(y.dtype)
        b_lambdas_flat = B_lambdas.T.ravel().astype(y.dtype)
        r_values_flat = R_values.T.ravel().astype(y.dtype)

        # Compute all 4 element values as per SIF structure:
        # A(I,J) = C01 * AC(I,J) * AL(I,J)^(2*ALPHA1-1)
        # B(I,J) = C01 * BC(I,J) * BL(I,J)^(2*ALPHA1-1)
        # C(I,J) = C02 * AC(I,J) * AL(I,J)^(2*ALPHA2-1)
        # D(I,J) = C02 * BC(I,J) * BL(I,J)^(2*ALPHA2-1)
        a_values = c01 * a_coeffs_flat * (a_lambdas_flat**beta1)
        b_values = c01 * b_coeffs_flat * (b_lambdas_flat**beta1)
        c_values = c02 * a_coeffs_flat * (a_lambdas_flat**beta2)
        d_values = c02 * b_coeffs_flat * (b_lambdas_flat**beta2)

        # Sum all 4 element functions for each (i,j)
        model_values = a_values + b_values + c_values + d_values

        # Compute residuals (constraints)
        constraints = model_values - r_values_flat

        return constraints, None

    @property
    def bounds(self):
        # SIF bounds: LO 'DEFAULT' -5.0, UP 'DEFAULT' 5.0
        lower = jnp.array([-5.0, -5.0, -5.0, -5.0])
        upper = jnp.array([5.0, 5.0, 5.0, 5.0])
        return lower, upper

    @property
    def expected_result(self):
        return None

    @property
    def expected_objective_value(self):
        return jnp.array(0.0)
