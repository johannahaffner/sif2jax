from jax import numpy as jnp
from jaxtyping import Array, Float

from ..._problem import AbstractNonlinearEquations


class ARTIF(AbstractNonlinearEquations):
    """An artificial nonlinear system.

    Source:
    K.M. Irani, M.P. Kamat, C.J. Ribbens, H.F.Walker and L.T. Watson,
    "Experiments with conjugate gradient algorithms for homotopy curve
     tracking" ,
    SIAM Journal on Optimization, May 1991, pp. 222-251, 1991.

    SIF input: Ph. Toint, May 1990.

    classification NOR2-AN-V-V
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})
    n: int = 5000  # Problem dimension (number of equations)

    def residual(self, y, args) -> Float[Array, "5000"]:
        """Residual function for the nonlinear equations."""
        # Add fixed boundary values (0 at both ends)
        x = jnp.concatenate([jnp.array([0.0]), y, jnp.array([0.0])])

        # Compute residuals for each equation
        residuals = []
        for i in range(1, self.n + 1):
            # Linear part: -0.05 * (X(i-1) + X(i) + X(i+1))
            linear_part = -0.05 * (x[i - 1] + x[i] + x[i + 1])

            # Nonlinear part: arctan(sin(i * X(i)))
            fact = float(i % 100)
            nonlinear_part = jnp.arctan(jnp.sin(fact * x[i]))

            residuals.append(linear_part + nonlinear_part)

        return jnp.array(residuals)

    def y0(self) -> Float[Array, "5000"]:
        """Initial guess for the optimization problem."""
        # All variables start at 1.0 (except fixed boundary values)
        return jnp.ones(self.n)

    def args(self):
        """Additional arguments for the residual function."""
        return None

    def expected_result(self) -> Float[Array, "5000"] | None:
        """Expected result of the optimization problem."""
        # The SIF file doesn't provide a solution
        return None

    def expected_objective_value(self) -> Float[Array, ""] | None:
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)
