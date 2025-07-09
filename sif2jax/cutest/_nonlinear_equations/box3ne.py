from jax import numpy as jnp
from jaxtyping import Array, Float

from ..._problem import AbstractNonlinearEquations


class BOX3NE(AbstractNonlinearEquations):
    """Box problem in 3 variables. This is a nonlinear equation version
    of problem BOX3.

    Source: Problem 12 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#BOX663
    SIF input: Ph. Toint, Dec 1989.
    Modification as a set of nonlinear equations: Nick Gould, Oct 2015.

    classification NOR2-AN-3-10
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def residual(self, y, args) -> Float[Array, "10"]:
        """Residual function for the nonlinear equations."""
        x1, x2, x3 = y

        # Number of groups
        m = 10

        # Compute residuals for each group
        residuals = []
        for i in range(m):
            ri = float(i + 1)  # i = 1 to 10 in 1-indexed
            ti = -0.1 * ri

            # Linear coefficient for x3
            emti = jnp.exp(-0.1 * ri)
            emri = jnp.exp(-ri)
            coeff = -emti + emri

            # Nonlinear elements
            a_i = jnp.exp(ti * x1)  # exp(-0.1*i*x1)
            b_i = jnp.exp(ti * x2)  # exp(-0.1*i*x2)

            # Group equation: coeff * x3 + a_i - b_i = 0
            residuals.append(coeff * x3 + a_i - b_i)

        return jnp.array(residuals)

    def y0(self) -> Float[Array, "3"]:
        """Initial guess for the optimization problem."""
        return jnp.array([0.0, 10.0, 1.0])

    def args(self):
        """Additional arguments for the residual function."""
        return None

    def expected_result(self) -> Float[Array, "3"] | None:
        """Expected result of the optimization problem."""
        # The SIF file doesn't provide a solution
        return None

    def expected_objective_value(self) -> Float[Array, ""] | None:
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)
