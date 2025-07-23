import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class DECONVNE(AbstractNonlinearEquations):
    """
    A problem arising in deconvolution analysis (nonlinear equation version).

    Source:
    J.P. Rasson, Private communication, 1996.

    SIF input: Ph. Toint, Nov 1996.
    unititialized variables fixed at zero, Nick Gould, Feb, 2013

    classification NOR2-MN-61-0
    """

    lgsg: int = 11
    lgtr: int = 40
    n: int = 51  # 40 + 11 variables: C(1:40) and SG(1:11)
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # TR data values
    tr_data = jnp.array(
        [
            0.0,
            0.0,
            1.600000e-03,
            5.400000e-03,
            7.020000e-02,
            0.1876000000,
            0.3320000000,
            0.7640000000,
            0.9320000000,
            0.8120000000,
            0.3464000000,
            0.2064000000,
            8.300000e-02,
            3.400000e-02,
            6.179999e-02,
            1.2000000000,
            1.8000000000,
            2.4000000000,
            9.0000000000,
            2.4000000000,
            1.8010000000,
            1.3250000000,
            7.620000e-02,
            0.2104000000,
            0.2680000000,
            0.5520000000,
            0.9960000000,
            0.3600000000,
            0.2400000000,
            0.1510000000,
            2.480000e-02,
            0.2432000000,
            0.3602000000,
            0.4800000000,
            1.8000000000,
            0.4800000000,
            0.3600000000,
            0.2640000000,
            6.000000e-03,
            6.000000e-03,
        ]
    )

    # SSG data values
    ssg_data = jnp.array(
        [
            1.000000e-02,
            2.000000e-02,
            0.4000000000,
            0.6000000000,
            0.8000000000,
            3.0000000000,
            0.8000000000,
            0.6000000000,
            0.4400000000,
            1.000000e-02,
            1.000000e-02,
        ]
    )

    def starting_point(self) -> Array:
        # C(-11) to C(0) are fixed at 0, so not included as variables
        # Only C(1) to C(40) are free variables, plus SG(1) to SG(11)
        # That's 40 + 11 = 51 free variables
        c_free = jnp.zeros(40, dtype=jnp.float64)  # C(1) to C(40)
        sg_values = self.ssg_data  # SG(1) to SG(11)
        return jnp.concatenate([c_free, sg_values])

    def num_residuals(self) -> int:
        return self.lgtr

    def residual(self, y: Array, args) -> Array:
        """Compute the residuals of the deconvolution problem"""
        # Split variables
        c = y[:40]  # C(1) to C(40)
        sg = y[40:]  # SG(1) to SG(11)

        # Create full C array with zeros for C(-11) to C(0)
        c_full = jnp.concatenate([jnp.zeros(12, dtype=jnp.float64), c])

        # Initialize residuals
        residuals = jnp.zeros(self.lgtr, dtype=jnp.float64)

        # Compute residuals R(K) for K = 1 to LGTR
        for k in range(self.lgtr):
            r_k = 0.0
            for i in range(self.lgsg):
                k_minus_i_plus_1 = k - i  # This gives index into c_full
                if k_minus_i_plus_1 >= 0:
                    # The element PROD(K,I) computes sg[i] * c[k-i+1] when idx > 0
                    # idx = k-i+1 in 1-based indexing, but we need 0-based
                    r_k += (
                        sg[i] * c_full[k_minus_i_plus_1 + 11]
                    )  # +11 to account for C(-11) offset
            residuals = residuals.at[k].set(r_k - self.tr_data[k])

        return residuals

    @property
    def y0(self) -> Array:
        """Initial guess for the optimization problem."""
        return self.starting_point()

    @property
    def args(self):
        """Additional arguments for the residual function."""
        return None

    def expected_result(self) -> Array:
        """Expected result of the optimization problem."""
        # Solution is not provided in the SIF file
        return self.starting_point()

    def expected_objective_value(self) -> Array:
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)

    def constraint(self, y):
        """Returns the residuals as equality constraints."""
        return self.residual(y, self.args), None

    @property
    def bounds(self) -> tuple[Array, Array] | None:
        """No bounds for this problem."""
        return None
