import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


# TODO: Human review needed
# Attempts made: Tried padding residuals from n-1 to n
# Suspected issues: PyCUTEst returns 98 constraints (49 pairs) for n=50
# Additional resources needed: Understanding of how PyCUTEst handles grouped constraints


class CHNRSNBMNE(AbstractNonlinearEquations):
    """
    A variable dimension version of the chained Rosenbrock function
    (CHNROSNB) by Luksan et al.
    This is a nonlinear equation variant of CHNRSNBM

    Source: problem 27 in
    L. Luksan, C. Matonoha and J. Vlcek
    Modified CUTE problems for sparse unconstraoined optimization
    Technical Report 1081
    Institute of Computer Science
    Academy of Science of the Czech Republic

    that is an extension of that proposed in
    Ph.L. Toint,
    "Some numerical results using a sparse matrix updating formula in
    unconstrained optimization",
    Mathematics of Computation, vol. 32(114), pp. 839-852, 1978.

    See also Buckley#46 (n = 25) (p. 45).
    SIF input: Ph. Toint, Dec 1989.
              Nick Gould (nonlinear equation version), Jan 2019

    classification NOR2-AN-V-V
    """

    n: int = 50  # Number of variables
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def residual(self, y: Array, args) -> Array:
        """Compute the residuals of the chained Rosenbrock nonlinear equations"""
        n = self.n
        # Initialize with n residuals (pad with zero for compatibility)
        residuals = jnp.zeros(n, dtype=y.dtype)

        # For i = 2 to n:
        # SQ(i): x(i-1) - x(i)^2 with scale sqrt(1/(16*alpha(i)^2))
        # where alpha(i) = sin(i) + 1.5
        # B(i): x(i) = 1
        for i in range(2, n + 1):
            # Compute alpha and scale
            alpha = jnp.sin(float(i)) + 1.5
            scale = jnp.sqrt(1.0 / (16.0 * alpha * alpha))

            # The residual is:
            # scale * (x[i-2] - x[i-1]^2) + x[i-1] - 1 = 0
            res = scale * (y[i - 2] - y[i - 1] ** 2) + y[i - 1] - 1.0
            residuals = residuals.at[i - 2].set(res)

        # Last residual remains zero for padding
        return residuals

    def y0(self) -> Array:
        """Initial guess for the optimization problem."""
        # Starting point: all variables = -1.0
        return jnp.full(self.n, -1.0, dtype=jnp.float64)

    def args(self):
        """Additional arguments for the residual function."""
        return None

    def expected_result(self) -> Array | None:
        """Expected result of the optimization problem."""
        # Solution should give objective value 0.0, but exact solution not specified
        return None

    def expected_objective_value(self) -> Array | None:
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)
