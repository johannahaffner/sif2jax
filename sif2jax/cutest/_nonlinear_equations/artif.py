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
    n: int = 5002  # Total number of variables including fixed boundary values
    n_eq: int = 5000  # Number of equations

    def residual(self, y, args) -> Float[Array, "5000"]:
        """Residual function for the nonlinear equations."""
        # y already contains all variables including fixed boundary values
        x = y

        # Compute residuals for each equation
        residuals = []
        for i in range(1, self.n_eq + 1):
            # Linear part: -0.05 * (X(i-1) + X(i) + X(i+1))
            linear_part = -0.05 * (x[i - 1] + x[i] + x[i + 1])

            # Nonlinear part: arctan(sin(i * X(i)))
            fact = float(i % 100)
            nonlinear_part = jnp.arctan(jnp.sin(fact * x[i]))

            residuals.append(linear_part + nonlinear_part)

        return jnp.array(residuals)

    @property
    def y0(self) -> Float[Array, "5002"]:
        """Initial guess for the optimization problem."""
        # All variables start at 1.0, including fixed boundary values
        y = jnp.ones(self.n)
        # Set boundary values to 0
        y = y.at[0].set(0.0)
        y = y.at[-1].set(0.0)
        return y

    @property
    def args(self):
        """Additional arguments for the residual function."""
        return None

    @property
    def expected_result(self) -> None:
        """Expected result of the optimization problem."""
        # The SIF file doesn't provide a solution
        return None

    @property
    def expected_objective_value(self) -> Float[Array, ""]:
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)

    def constraint(self, y):
        """Returns the residuals as equality constraints."""
        return self.residual(y, self.args), None

    @property
    def bounds(self) -> tuple[Array, Array] | None:
        """Bounds for the variables."""
        # X(0) and X(N+1) are fixed at 0
        lower = jnp.full(self.n, -jnp.inf)
        upper = jnp.full(self.n, jnp.inf)

        # Fix boundary values
        lower = lower.at[0].set(0.0)
        upper = upper.at[0].set(0.0)
        lower = lower.at[-1].set(0.0)
        upper = upper.at[-1].set(0.0)

        return lower, upper
