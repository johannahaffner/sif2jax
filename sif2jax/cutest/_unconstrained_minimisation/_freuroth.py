import jax
import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractUnconstrainedMinimisation


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class FREUROTH(AbstractUnconstrainedMinimisation):
    """The FREUROTH function.

    The Freudentstein and Roth test problem.

    Source: problem 2 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Toint#33, Buckley#24
    SIF input: Ph. Toint, Dec 1989.

    Classification: SUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 5000  # Default dimension in SIF file
    supported_dims = [2, 10, 50, 100, 500, 1000, 5000]

    def __post_init__(self):
        if self.n not in self.supported_dims:
            raise ValueError(
                f"Unsupported dimension: {self.n}. "
                f"Supported dimensions are: {self.supported_dims}."
            )

    def objective(self, y, args):
        del args

        # Number of group sets (one less than the number of variables)
        ngs = self.n - 1

        # Vectorized implementation using jax
        def compute_residual_r(i):
            xi = y[i]
            xi_plus_1 = y[i + 1]

            # The FRDRTH element used with coefficients 5.0 and -1.0
            elv = xi_plus_1
            coeff = 5.0
            xcoeff = -1.0

            # Formula from the SIF file's ELEMENTS section
            elv2 = elv * elv
            xcelv = xcoeff * elv
            element_result = (coeff + xcelv) * elv2

            # Residual r_i = x_i - 2*x_{i+1} + element_result - 13
            residual = xi - 2.0 * xi_plus_1 + element_result - 13.0
            return residual

        def compute_residual_s(i):
            xi = y[i]
            xi_plus_1 = y[i + 1]

            # The FRDRTH element used with coefficients 1.0 and 1.0
            elv = xi_plus_1
            coeff = 1.0
            xcoeff = 1.0

            # Formula from the SIF file's ELEMENTS section
            elv2 = elv * elv
            xcelv = xcoeff * elv
            element_result = (coeff + xcelv) * elv2

            # Residual s_i = x_i - 14*x_{i+1} + element_result - 29
            residual = xi - 14.0 * xi_plus_1 + element_result - 29.0
            return residual

        # Compute all residuals using vmap
        indices = jnp.arange(ngs)
        r_residuals = jax.vmap(compute_residual_r)(indices)
        s_residuals = jax.vmap(compute_residual_s)(indices)

        # Sum of squared residuals (least squares objective)
        return jnp.sum(r_residuals**2 + s_residuals**2)

    def y0(self):
        # Starting point from SIF file: x1=0.5, x2=-2.0, rest are zeros
        x0 = jnp.zeros(self.n)
        x0 = x0.at[0].set(0.5)
        x0 = x0.at[1].set(-2.0)
        return inexact_asarray(x0)

    def args(self):
        return None

    def expected_result(self):
        return None

    def expected_objective_value(self):
        # Expected objective values from the SIF file for different dimensions
        if self.n == 2:
            return jnp.array(48.984)  # Approximate value from SIF file
        elif self.n == 10:
            return jnp.array(1.0141e3)
        elif self.n == 50:
            return jnp.array(5.8810e3)
        elif self.n == 100:
            return jnp.array(1.1965e4)
        elif self.n == 500:
            return jnp.array(6.0634e4)
        elif self.n == 1000:
            return jnp.array(1.2147e5)
        elif self.n == 5000:
            return jnp.array(6.0816e5)
        else:
            # This should never happen due to the check in __post_init__
            raise ValueError(f"No expected objective value for dimension {self.n}")


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class FREURONE(AbstractUnconstrainedMinimisation):
    """The FREURONE function.

    The Freudentstein and Roth test problem. This is a nonlinear equation
    version of problem FREUROTH.

    Source: problem 2 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Toint#33, Buckley#24
    SIF input: Ph. Toint, Dec 1989.
    Modification as a set of nonlinear equations: Nick Gould, Oct 2015.

    Classification: NOR2-AN-V-V
    """

    n: int = 2  # Default dimension in SIF file
    supported_dims = [2, 10, 50, 100, 500, 1000, 5000]

    def __post_init__(self):
        if self.n not in self.supported_dims:
            raise ValueError(
                f"Unsupported dimension: {self.n}. "
                f"Supported dimensions are: {self.supported_dims}."
            )

    def objective(self, y, args):
        del args

        # Number of group sets (one less than the number of variables)
        ngs = self.n - 1

        # Vectorized implementation using jax
        def compute_residual_r(i):
            xi = y[i]
            xi_plus_1 = y[i + 1]

            # The FRDRTH element used with coefficients 5.0 and -1.0
            elv = xi_plus_1
            coeff = 5.0
            xcoeff = -1.0

            # Formula from the SIF file's ELEMENTS section
            elv2 = elv * elv
            xcelv = xcoeff * elv
            element_result = (coeff + xcelv) * elv2

            # Residual r_i = x_i - 2*x_{i+1} + element_result - 13
            # In the nonlinear equation version, we don't square the residuals
            residual = xi - 2.0 * xi_plus_1 + element_result - 13.0
            return residual

        def compute_residual_s(i):
            xi = y[i]
            xi_plus_1 = y[i + 1]

            # The FRDRTH element used with coefficients 1.0 and 1.0
            elv = xi_plus_1
            coeff = 1.0
            xcoeff = 1.0

            # Formula from the SIF file's ELEMENTS section
            elv2 = elv * elv
            xcelv = xcoeff * elv
            element_result = (coeff + xcelv) * elv2

            # Residual s_i = x_i - 14*x_{i+1} + element_result - 29
            # In the nonlinear equation version, we don't square the residuals
            residual = xi - 14.0 * xi_plus_1 + element_result - 29.0
            return residual

        # Compute all residuals using vmap
        indices = jnp.arange(ngs)
        r_residuals = jax.vmap(compute_residual_r)(indices)
        s_residuals = jax.vmap(compute_residual_s)(indices)

        # Combine all residuals into a single vector
        all_residuals = jnp.concatenate([r_residuals, s_residuals])

        # Sum of squared residuals (nonlinear equations formulation)
        return jnp.sum(all_residuals**2)

    def y0(self):
        # Starting point from SIF file: x1=0.5, x2=-2.0, rest are zeros
        x0 = jnp.zeros(self.n)
        x0 = x0.at[0].set(0.5)
        x0 = x0.at[1].set(-2.0)
        return inexact_asarray(x0)

    def args(self):
        return None

    def expected_result(self):
        return None

    def expected_objective_value(self):
        # Expected objective values from the SIF file for different dimensions
        if self.n == 2:
            return jnp.array(48.984)  # Approximate value from SIF file
        elif self.n == 10:
            return jnp.array(1.0141e3)
        elif self.n == 50:
            return jnp.array(5.8810e3)
        elif self.n == 100:
            return jnp.array(1.1965e4)
        elif self.n == 500:
            return jnp.array(6.0634e4)
        elif self.n == 1000:
            return jnp.array(1.2147e5)
        elif self.n == 5000:
            return jnp.array(6.0816e5)
        else:
            # This should never happen due to the check in __post_init__
            raise ValueError(f"No expected objective value for dimension {self.n}")
