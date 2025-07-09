import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class BRYBNDNE(AbstractNonlinearEquations):
    """
    Broyden banded system of nonlinear equations.
    This is a nonlinear equation variant of BRYBND

    Source: problem 31 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#73 (p. 41) and Toint#18

    SIF input: Ph. Toint, Dec 1989.
              Nick Gould (nonlinear equation version), Jan 2019

    classification NOR2-AN-V-V
    """

    n: int = 5000
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Problem parameters
    kappa1: float = 2.0
    kappa2: float = 5.0
    kappa3: float = 1.0
    lb: int = 5  # Lower bandwidth
    ub: int = 1  # Upper bandwidth

    def starting_point(self) -> Array:
        return jnp.ones(self.n, dtype=jnp.float64)

    def num_residuals(self) -> int:
        return self.n

    def residual(self, y: Array, args) -> Array:
        """Compute the residuals of the Broyden banded nonlinear equations"""
        n = self.n
        lb = self.lb
        ub = self.ub
        kappa1 = self.kappa1
        kappa2 = self.kappa2
        kappa3 = self.kappa3

        residuals = jnp.zeros(n, dtype=y.dtype)

        for i in range(n):
            # Linear part: kappa1 * y[i]
            res_i = kappa1 * y[i]

            # Nonlinear part: kappa2 * (y[i]^3 or y[i]^2 depending on position)
            # The pattern from the SIF file shows:
            # - For i < lb: use CB (cubic) for y[i]
            # - For lb <= i < n-ub: use SQ (square) for y[i]
            # - For i >= n-ub: use CB (cubic) for y[i]
            if i < lb or i >= n - ub:
                res_i += kappa2 * y[i] ** 3
            else:
                res_i += kappa2 * y[i] ** 2

            # Add contributions from other variables
            # Lower band contributions
            for j in range(max(0, i - lb), i):
                if i < lb:
                    # Upper left corner: use SQ for j < i
                    res_i -= kappa3 * y[j] ** 2
                elif i >= n - ub:
                    # Lower right corner: use SQ for all j in range
                    res_i -= kappa3 * y[j] ** 2
                else:
                    # Middle part: use CB for j in lower band
                    res_i -= kappa3 * y[j] ** 3

            # Upper band contributions
            for j in range(i + 1, min(n, i + ub + 1)):
                # Always use SQ for upper band
                res_i -= kappa3 * y[j] ** 2

            residuals = residuals.at[i].set(res_i)

        return residuals

    def y0(self) -> Array:
        """Initial guess for the optimization problem."""
        return self.starting_point()

    def args(self):
        """Additional arguments for the residual function."""
        return None

    def expected_result(self) -> Array | None:
        """Expected result of the optimization problem."""
        # The SIF file mentions solution value 0.0 but not the exact solution vector
        return None

    def expected_objective_value(self) -> Array | None:
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)
