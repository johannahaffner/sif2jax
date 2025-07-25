import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class DGOSPEC(AbstractBoundedMinimisation):
    """A global-optimization test example.

    Source: an example from the specification document for GALAHAD's DGO

    SIF input: Nick Gould, August 2021

    classification: OBR2-AN-3-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 3  # 3 variables

    @property
    def y0(self):
        # Starting point from SIF file
        return jnp.zeros(self.n)

    @property
    def args(self):
        return ()

    def objective(self, y, args):
        """Compute the objective function.

        Minimize: (x1 + x3 + 4)^2 + (x2 + x3)^2 + 1000*cos(10*x1) + (x1 + x2 + x3)
        """
        x1, x2, x3 = y

        # Parameters
        freq = 10.0
        mag = 1000.0

        # Group Q1: (x1 + x3 + 4.0)^2 with L2 type
        # The constant in SIF is on the RHS, so it's added not subtracted
        q1 = x1 + x3 + 4.0
        term1 = q1 * q1

        # Group Q2: (x2 + x3)^2 with L2 type
        q2 = x2 + x3
        term2 = q2 * q2

        # Group N1: MAG * cos(FREQ * x1)
        term3 = mag * jnp.cos(freq * x1)

        # Group L1: x1 + x2 + x3 (identity type)
        term4 = x1 + x2 + x3

        return term1 + term2 + term3 + term4

    @property
    def bounds(self):
        """Returns the bounds on the variables."""
        # From SIF file: -1.0 <= xi <= 0.5 for all variables
        lower = jnp.full(self.n, -1.0)
        upper = jnp.full(self.n, 0.5)
        return lower, upper

    @property
    def expected_result(self):
        # The optimal solution is not explicitly given in the SIF file
        return None

    @property
    def expected_objective_value(self):
        # From the SIF file comment (not specified exactly)
        return None
