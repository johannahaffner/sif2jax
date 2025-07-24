import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class EG2(AbstractUnconstrainedMinimisation):
    """The EG2 function.

    A simple nonlinear problem given as an example in Section 1.2.4 of
    the LANCELOT Manual. The problem is non convex and has several local minima.

    Source:
    A.R. Conn, N. Gould and Ph.L. Toint,
    "LANCELOT, A Fortran Package for Large-Scale Nonlinear Optimization
    (Release A)" Springer Verlag, 1992.

    Note J. Haffner --------------------------------------------------------------------
    Reference: https://doi.org/10.1007/978-3-662-12211-2_1, Chapter 1, page 11
    ------------------------------------------------------------------------------------

    SIF input: N. Gould and Ph. Toint, June 1994.

    Classification: OUR2-AN-1000-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 1000  # Problem specifies N=1000

    def objective(self, y, args):
        # From AMPL: sum {i in 1..N-1} sin(x[1] + x[i]^2 - 1.0) + 0.5*sin(x[N]^2)
        del args
        first = y[0]
        last = y[-1]

        # Sum from i=1 to N-1 of sin(x[1] + x[i]^2 - 1.0)
        # Note: in 0-indexed arrays, this is i=0 to N-2
        f1 = jnp.sum(jnp.sin(first + y[:-1] ** 2 - 1.0))

        # Add 0.5*sin(x[N]^2)
        f2 = 0.5 * jnp.sin(last**2)

        return f1 + f2

    @property
    def y0(self):
        # Initial guess: all zeros (from PyCUTEst)
        return jnp.zeros(self.n)

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        return None

    @property
    def expected_objective_value(self):
        return None
