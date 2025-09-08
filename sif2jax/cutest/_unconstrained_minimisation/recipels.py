import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class RECIPELS(AbstractUnconstrainedMinimisation):
    """RECIPELS problem - Least-squares version of RECIPE.

    Source: problem 155 (p. 88) in
    A.R. Buckley,
    "Test functions for unconstrained minimization",
    TR 1989CS-3, Mathematics, statistics and computing centre,
    Dalhousie University, Halifax (CDN), 1989.

    SIF input: Ph. Toint, Dec 1989.
    Least-squares version of RECIPE.SIF, Nick Gould, Jan 2020.

    Classification: SUR2-AY-3-0

    This is the least-squares version of RECIPE, minimizing:
    (X1 - 5)² + (X2²)² + (X3/(X2-X1))²

    TODO: Human review needed
    Issue: Test timeouts on 3-variable problem despite simple structure
    Note: RECIPE (nonlinear equations version) works correctly, but this
          least-squares version times out unexpectedly. Suspect JAX compilation
          or gradient computation issue with the division term.
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return jnp.array(3)

    def objective(self, y, args):
        """Compute the objective function."""
        del args
        x1, x2, x3 = y[0], y[1], y[2]

        # G1: (X1 - 5)²
        term1 = (x1 - 5.0) ** 2

        # G2: (X2²)² = X2⁴
        term2 = x2**4

        # G3: (X3/(X2-X1))²
        u = x2 - x1
        term3 = (x3 / u) ** 2

        return term1 + term2 + term3

    @property
    def y0(self):
        """Initial guess."""
        return jnp.array([2.0, 5.0, 1.0])

    @property
    def args(self):
        """Additional arguments."""
        return None

    @property
    def bounds(self):
        """Bounds on variables."""
        # All variables are free (FR RECIPE 'DEFAULT')
        return None

    @property
    def expected_result(self):
        """Expected optimal solution."""
        # From the least-squares structure, minimum occurs when each term is zero:
        # G1: X1 = 5
        # G2: X2 = 0
        # G3: X3/(X2-X1) = 0 → X3/(0-5) = 0 → X3 = 0
        return jnp.array([5.0, 0.0, 0.0])

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        # At the optimal point, all terms are zero
        return jnp.array(0.0)
