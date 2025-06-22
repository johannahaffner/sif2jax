import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class FLETCHCR(AbstractUnconstrainedMinimisation):
    """The FLETCHCR function.

    The chained Rosenbrock function as given by Fletcher.

    Source: The second problem given by
    R. Fletcher, "An optimal positive definite update for sparse Hessian matrices"
    Numerical Analysis report NA/145, University of Dundee, 1992.

    SIF input: Nick Gould, Oct 1992.
    Classification: OUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 1000  # Default dimension in SIF file

    def objective(self, y, args):
        del args

        # Let me try the exact AMPL formulation without the +1 term first
        # Standard chained Rosenbrock: sum 100*(x[i+1] - x[i]^2)^2 + (x[i] - 1)^2
        term1 = jnp.sum(100 * (y[1:] - y[:-1] ** 2) ** 2)
        term2 = jnp.sum((y[:-1] - 1.0) ** 2)

        return term1 + term2

    def y0(self):
        # Starting point: all zeros
        return jnp.zeros(self.n)

    def args(self):
        return None

    def expected_result(self):
        # Minimum value is 0, achieved when all variables are 1
        return jnp.ones(self.n)

    def expected_objective_value(self):
        return jnp.array(0.0)
