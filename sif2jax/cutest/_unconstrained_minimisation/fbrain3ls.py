import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation
from ..data import load_fbrain_data


# Load FBRAIN data (FBRAIN3LS uses same data as FBRAIN)
A_coeffs, A_lambdas, B_coeffs, B_lambdas, R_values = load_fbrain_data("fbrain")


class FBRAIN3LS(AbstractUnconstrainedMinimisation):
    """FBRAIN3LS - Nonlinear Least-Squares problem for brain tissue modeling.

    This problem involves fitting a model of the shear stress in human brain tissue
    to experimental data, formulated as a nonlinear least-squares problem.

    The model function is:
    f(λ) = C0_1 * λ^(2*α_1-1) + C0_2 * λ^(2*α_2-1) + C0_3 * λ^(2*α_3-1)

    where λ represents the shear deformation ratio.

    Source: an example in
    L.A. Mihai, S. Budday, G.A. Holzapfel, E. Kuhl and A. Goriely.
    "A family of hyperelastic models for human brain tissue",
    Journal of Mechanics and Physics of Solids,
    DOI: 10.1016/j.jmps.2017.05.015 (2017).

    As conveyed by Angela Mihai (U. Cardiff)

    SIF input: Nick Gould, June 2017.

    Classification: SUR2-AN-6-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        del args

        # Extract the 6 parameters
        alpha1, c01, alpha2, c02, alpha3, c03 = y

        # Compute betas
        beta1 = 2.0 * alpha1 - 1.0
        beta2 = 2.0 * alpha2 - 1.0
        beta3 = 2.0 * alpha3 - 1.0

        # Flatten arrays in column-major order to match SIF constraint ordering
        # SIF iterates: DO J (1 to N), DO I (0 to M) -> (I,J) pairs
        # Cast to same dtype as y to avoid type promotion errors
        a_coeffs_flat = A_coeffs.T.ravel().astype(y.dtype)
        a_lambdas_flat = A_lambdas.T.ravel().astype(y.dtype)
        b_coeffs_flat = B_coeffs.T.ravel().astype(y.dtype)
        b_lambdas_flat = B_lambdas.T.ravel().astype(y.dtype)
        r_values_flat = R_values.T.ravel().astype(y.dtype)

        # Compute all 6 element values as per SIF structure:
        # A(I,J) = C01 * AC(I,J) * AL(I,J)^(2*ALPHA1-1)
        # B(I,J) = C01 * BC(I,J) * BL(I,J)^(2*ALPHA1-1)
        # C(I,J) = C02 * AC(I,J) * AL(I,J)^(2*ALPHA2-1)
        # D(I,J) = C02 * BC(I,J) * BL(I,J)^(2*ALPHA2-1)
        # E(I,J) = C03 * AC(I,J) * AL(I,J)^(2*ALPHA3-1)
        # F(I,J) = C03 * BC(I,J) * BL(I,J)^(2*ALPHA3-1)
        a_values = c01 * a_coeffs_flat * (a_lambdas_flat**beta1)
        b_values = c01 * b_coeffs_flat * (b_lambdas_flat**beta1)
        c_values = c02 * a_coeffs_flat * (a_lambdas_flat**beta2)
        d_values = c02 * b_coeffs_flat * (b_lambdas_flat**beta2)
        e_values = c03 * a_coeffs_flat * (a_lambdas_flat**beta3)
        f_values = c03 * b_coeffs_flat * (b_lambdas_flat**beta3)

        # Sum all 6 element functions for each (i,j)
        model_values = a_values + b_values + c_values + d_values + e_values + f_values

        # Compute residuals
        residuals = model_values - r_values_flat

        # Return the sum of squared residuals (least squares objective)
        return jnp.sum(residuals**2)

    @property
    def y0(self):
        # Starting point from the SIF file
        return jnp.array([-4.0, -0.1, 4.0, 0.1, -2.0, -0.1])

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        # The exact solution is not specified in the SIF file
        return None

    @property
    def expected_objective_value(self):
        # The minimum objective value is not precisely specified in the SIF file
        # But the lower bound is given as 0.0
        return jnp.array(0.0)
