import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class FREURONE(AbstractNonlinearEquations):
    """
    The Freudentstein and Roth test problem. This is a nonlinear equation
    version of problem FREUROTH

    Source: problem 2 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Toint#33, Buckley#24
    SIF input: Ph. Toint, Dec 1989.
    Modification as a set of nonlinear equations: Nick Gould, Oct 2015.

    classification NOR2-AN-V-V
    """

    n: int = 2  # Default value from SIF
    m: int = 2  # Number of equations = 2 * (n-1)
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def starting_point(self) -> Array:
        # From SIF file
        x0 = jnp.zeros(self.n, dtype=jnp.float64)
        x0 = x0.at[0].set(0.5)
        x0 = x0.at[1].set(-2.0)
        return x0

    def num_residuals(self) -> int:
        return self.m

    def residual(self, y: Array, args) -> Array:
        """Compute the residuals of the Freudenstein-Roth problem"""
        x = y
        n = self.n
        ngs = n - 1  # Number of group sets

        # Initialize residuals - we have 2 residuals per group set
        residuals = jnp.zeros(2 * ngs, dtype=jnp.float64)

        for i in range(ngs):
            i_plus_1 = i + 1

            # Element A(i): FRDRTH with ELV=x(i+1), COEFF=5.0, XCOEFF=-1.0
            elv = x[i_plus_1]
            elv2 = elv * elv
            xcelv = -1.0 * elv
            a_i = (5.0 + xcelv) * elv2  # (COEFF + XCELV) * ELV2

            # Element B(i): FRDRTH with ELV=x(i+1), COEFF=1.0, XCOEFF=1.0
            xcelv_b = 1.0 * elv
            b_i = (1.0 + xcelv_b) * elv2  # (COEFF + XCELV) * ELV2

            # R(i) = x(i) - 2*x(i+1) + A(i) - 13.0
            residuals = residuals.at[2 * i].set(x[i] - 2.0 * x[i_plus_1] + a_i - 13.0)

            # S(i) = x(i) - 14*x(i+1) + B(i) - 29.0
            residuals = residuals.at[2 * i + 1].set(
                x[i] - 14.0 * x[i_plus_1] + b_i - 29.0
            )

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
        # Solution is not explicitly provided in the SIF file
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
