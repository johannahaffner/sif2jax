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
    n: int = 63  # 52 C variables (C(-11:40)) + 11 SG variables (SG(1:11))
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
        # C(-11) to C(0) are fixed at 0, but included as variables
        # C(-11) to C(40) gives 52 variables, plus SG(1) to SG(11) gives 11 more
        # That's 52 + 11 = 63 total variables
        c_fixed = jnp.zeros(12, dtype=jnp.float64)  # C(-11) to C(0), fixed at 0
        c_free = jnp.zeros(40, dtype=jnp.float64)  # C(1) to C(40)
        sg_values = self.ssg_data  # SG(1) to SG(11)
        return jnp.concatenate([c_fixed, c_free, sg_values])

    def num_residuals(self) -> int:
        return self.lgtr

    def residual(self, y: Array, args) -> Array:
        """Compute the residuals of the deconvolution problem"""
        # Split variables
        c_full = y[:52]  # C(-11) to C(40), all 52 C variables
        sg = y[52:]  # SG(1) to SG(11)

        # Initialize residuals
        residuals = jnp.zeros(self.lgtr, dtype=jnp.float64)

        # Compute residuals R(K) for K = 1 to LGTR
        for k in range(self.lgtr):
            r_k = 0.0
            for i in range(self.lgsg):
                # IDX = K-I+1 in 1-based indexing
                idx = (k + 1) - (i + 1) + 1  # Convert to 1-based, compute IDX
                if idx > 0:
                    # The element PROD(K,I) computes sg[i] * c[idx] when IDX > 0
                    # C(idx) is at position idx + 11 in c_full
                    # (since C(-11) is at index 0)
                    r_k += sg[i] * c_full[idx + 11]
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

    @property
    def expected_result(self) -> Array:
        """Expected result of the optimization problem."""
        # Solution is not provided in the SIF file
        return self.starting_point()

    @property
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
