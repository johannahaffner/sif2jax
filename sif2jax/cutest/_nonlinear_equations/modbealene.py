import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class MODBEALENE(AbstractNonlinearEquations):
    """
    A variation on Beale's problem in 2 variables
    This is a nonlinear equation variant of MODBEALE

    Source: An adaptation by Ph. Toint of Problem 5 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#89.
    SIF input: Ph. Toint, Mar 2003.
               Nick Gould (nonlinear equation version), Jan 2019

    classification NOR2-AN-V-V
    """

    n_half: int = 10000  # N/2 parameter
    alpha: float = 50.0
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def __init__(self, n_half: int = 10000):
        self.n_half = n_half

    @property
    def n(self) -> int:
        """Total number of variables is 2 * N/2."""
        return 2 * self.n_half

    def num_residuals(self) -> int:
        """Number of residuals = 3 * N/2 + (N/2 - 1)."""
        return 4 * self.n_half - 1

    def starting_point(self) -> Array:
        """Return the starting point for the problem."""
        return jnp.ones(self.n, dtype=jnp.float64)

    def _prodb_element(self, v1: Array, v2: Array, power: float) -> Array:
        """Product type element: v1 * (1 - v2^power)."""
        t = 1.0 - v2**power
        return v1 * t

    def residual(self, y: Array, args) -> Array:
        """Compute the residual vector."""
        n_half = self.n_half
        ralphinv = jnp.sqrt(1.0 / self.alpha)

        residuals = []

        # Process groups for i = 1 to N/2-1 (BA, BB, BC, L interleaved)
        for i in range(1, n_half):
            # Index calculations
            j = 2 * (i - 1)  # 0-based indexing for array access

            # BA(i) group: AE(i) - 1.5
            # AE(i) = X(2i-1) * (1 - X(2i)^1)
            ae = self._prodb_element(y[j], y[j + 1], 1.0)
            residuals.append(ae - 1.5)

            # BB(i) group: BE(i) - 2.25
            # BE(i) = X(2i-1) * (1 - X(2i)^2)
            be = self._prodb_element(y[j], y[j + 1], 2.0)
            residuals.append(be - 2.25)

            # BC(i) group: CE(i) - 2.625
            # CE(i) = X(2i-1) * (1 - X(2i)^3)
            ce = self._prodb_element(y[j], y[j + 1], 3.0)
            residuals.append(ce - 2.625)

            # L(i) = ralphinv * (6.0 * X(j+1) - X(j+2))
            # Note: j+1 in SIF is j+1 in 0-based, j+2 in SIF is j+2 in 0-based
            l_val = ralphinv * (6.0 * y[j + 1] - y[j + 2])
            residuals.append(l_val)

        # Process final group i = N/2 (BA, BB, BC only, no L)
        i = n_half
        j = 2 * (i - 1)

        # BA(N/2)
        ae = self._prodb_element(y[j], y[j + 1], 1.0)
        residuals.append(ae - 1.5)

        # BB(N/2)
        be = self._prodb_element(y[j], y[j + 1], 2.0)
        residuals.append(be - 2.25)

        # BC(N/2)
        ce = self._prodb_element(y[j], y[j + 1], 3.0)
        residuals.append(ce - 2.625)

        return jnp.array(residuals, dtype=jnp.float64)

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
        # Not explicitly given, but for nonlinear equations should satisfy F(x*) = 0
        return jnp.zeros(self.n, dtype=jnp.float64)

    def expected_objective_value(self) -> Array:
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)

    def constraint(self, y):
        """Returns the residuals as equality constraints."""
        return self.residual(y, self.args), None

    @property
    def bounds(self) -> tuple[Array, Array] | None:
        """Bounds for variables - free variables."""
        return None
