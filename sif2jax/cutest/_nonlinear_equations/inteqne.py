import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class INTEQNE(AbstractNonlinearEquations):
    """
    The discrete integral problem (INTEGREQ) without fixed variables

    Source:  Problem 29 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    SIF input: Ph. Toint, Feb 1990.
    Modification to remove fixed variables: Nick Gould, Oct 2015.

    classification NOR2-AN-V-V
    """

    n: int = 10  # Number of free variables
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def num_residuals(self) -> int:
        """Number of residuals."""
        return self.n + 2  # Total discretization points

    def starting_point(self) -> Array:
        """Return the starting point for the problem."""
        # N+2 discretization points, but x(0) and x(N+1) are fixed at 0
        h = 1.0 / (self.n + 1)
        x = jnp.zeros(self.n + 2, dtype=jnp.float64)

        # Starting values for x(1) to x(N)
        for i in range(1, self.n + 1):
            t_i = i * h
            x = x.at[i].set(t_i * (t_i - 1.0))

        return x

    def residual(self, y: Array, args) -> Array:
        """Compute the residual vector.

        The problem discretizes an integral equation. The residuals are:
        G(0): x(0) = 0
        G(i): x(i) + sum of weighted cubic terms = 0, for i = 1..N
        G(N+1): x(N+1) = 0
        """
        x = y
        n = self.n
        h = 1.0 / (n + 1)
        halfh = 0.5 * h

        residuals = jnp.zeros(n + 2, dtype=jnp.float64)

        # G(0): x(0)
        residuals = residuals.at[0].set(x[0])

        # G(i) for i = 1..N
        for i in range(1, n + 1):
            t_i = i * h

            # Initialize residual with x(i)
            res_i = x[i]

            # First sum: j from 1 to i
            p1 = (1.0 - t_i) * halfh
            for j in range(1, i + 1):
                t_j = j * h
                w_il = p1 * t_j
                # Element A(j): (x(j) + (1 + t_j))^3
                vplusb = x[j] + (1.0 + t_j)
                res_i += w_il * (vplusb**3)

            # Second sum: j from i+1 to N
            p2 = t_i * halfh
            for j in range(i + 1, n + 1):
                t_j = j * h
                w_iu = p2 * (1.0 - t_j)
                # Element A(j): (x(j) + (1 + t_j))^3
                vplusb = x[j] + (1.0 + t_j)
                res_i += w_iu * (vplusb**3)

            residuals = residuals.at[i].set(res_i)

        # G(N+1): x(N+1)
        residuals = residuals.at[n + 1].set(x[n + 1])

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
        # Solution should have x(0) = 0, x(N+1) = 0
        return jnp.zeros(self.n + 2, dtype=jnp.float64)

    def expected_objective_value(self) -> Array:
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)

    def constraint(self, y):
        """Returns the residuals as equality constraints."""
        return self.residual(y, self.args), None

    @property
    def bounds(self) -> tuple[Array, Array] | None:
        """Free bounds for all variables."""
        return None
