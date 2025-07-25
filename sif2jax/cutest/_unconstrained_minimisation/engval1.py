import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class ENGVAL1(AbstractUnconstrainedMinimisation):
    """The ENGVAL1 problem.

    This problem is a sum of 2n-2 groups, n-1 of which contain 2 nonlinear elements.

    Source: problem 31 in
    Ph.L. Toint,
    "Test problems for partially separable optimization and results
    for the routine PSPMIN",
    Report 83/4, Department of Mathematics, FUNDP (Namur, B), 1983.

    See also Buckley#172 (p. 52)

    SIF input: Ph. Toint and N. Gould, Dec 1989.

    Classification: OUR2-AN-V-0
    """

    _n: int = 5000
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return self._n

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def y0(self):
        """Initial guess."""
        return jnp.full(self.n, 2.0)

    def objective(self, y, args):
        """Compute the objective function.

        The objective is the sum of:
        - E(i): squared L2 norm of (Y(i) + Z(i)) where Y(i) = y[i]^2, Z(i) = y[i+1]^2
        - L(i): linear group with coefficient -4.0 for y[i] and constant -3.0

        Total: sum_{i=1}^{n-1} [(y[i]^2 + y[i+1]^2)^2 + (-4*y[i] - 3)]
        """
        del args  # Not used

        # Vectorized computation for all groups
        # Extract y[0:n-1] and y[1:n]
        y_i = y[:-1]
        y_i_plus_1 = y[1:]

        # Elements Y(i) = y[i]^2, Z(i) = y[i+1]^2
        y_squared = y_i**2
        z_squared = y_i_plus_1**2

        # Group E(i): L2 group type with elements Y(i) and Z(i)
        # So E(i) = (Y(i) + Z(i))^2 = (y[i]^2 + y[i+1]^2)^2
        e = (y_squared + z_squared) ** 2

        # Group L(i): linear group with y[i] coefficient -4.0 and constant -3.0
        # In SIF files, constants are added to the group value, not subtracted
        l = -4.0 * y_i + 3.0

        # Sum all groups
        obj = jnp.sum(e + l)

        return obj

    @property
    def expected_result(self):
        """Expected optimal solution."""
        # The optimal solution has all variables equal to 1
        return jnp.ones(self.n)

    @property
    def expected_objective_value(self):
        """Expected objective value at solution."""
        # At y = ones: each E(i) = (1 + 1)^2 = 4, each L(i) = -4*1 - 3 = -7
        # Total per group: 4 - 7 = -3
        # With n-1 groups: -3*(n-1)
        return jnp.array(-3.0 * (self.n - 1))
