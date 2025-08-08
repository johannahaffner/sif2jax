import jax.numpy as jnp
from jax import Array

from ..._misc import inexact_asarray
from ..._problem import AbstractConstrainedMinimisation


class DEGTRIDL(AbstractConstrainedMinimisation):
    """
    A degenerate convex quadratic program with a tri-diagonal Hessian
    and a linear equality constraint.

    The problem has the form:
    minimize    0.5 * x^T H x - c^T x
    subject to  sum(x_i) = N

    where H is a tridiagonal matrix with 1 on the diagonal and 0.5 on the
    off-diagonals, c is [0.5, -0.5, -1.0, ..., -1.0, -0.5], and N is the
    problem dimension parameter.

    SIF input: Nick Gould, August 2011

    classification QLR2-AN-V-1
    """

    _n: int = 100001  # Number of variables (N+1 where N=100000)
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self) -> int:
        """Number of variables is N+1"""
        return self._n

    def objective(self, y: Array, args) -> Array:
        """Compute the quadratic objective function"""
        n = self._n

        # Linear term coefficients (note different from DEGTRID)
        c = jnp.ones(n) * (-1.0)
        c = c.at[0].set(0.5)
        c = c.at[1].set(-0.5)
        c = c.at[-1].set(-0.5)

        # Quadratic term: 0.5 * x^T H x
        # H is tridiagonal with 1 on diagonal and 0.5 on off-diagonals

        # Diagonal contribution: sum(x_i^2)
        quad_diag = jnp.sum(y**2)

        # Off-diagonal contribution: 2 * 0.5 * sum(x_i * x_{i-1}) = sum(x_i * x_{i-1})
        quad_offdiag = jnp.sum(y[1:] * y[:-1])

        # Total quadratic term
        quad_term = 0.5 * (quad_diag + quad_offdiag)

        # Linear term
        linear_term = jnp.dot(c, y)

        return quad_term + linear_term

    def constraint(self, y: Array):
        """Linear equality constraint: sum(x_i) = N"""
        n_param = self._n - 1  # N in the SIF file

        # Equality constraint: sum(x_i) - N = 0
        eq_constraints = jnp.array([jnp.sum(y) - n_param])

        # No inequality constraints
        return eq_constraints, None

    @property
    def bounds(self):
        """No bounds for this problem"""
        return None

    @property
    def y0(self) -> Array:
        """Initial point: all components set to 2.0"""
        return inexact_asarray(jnp.full(self._n, 2.0))

    @property
    def args(self):
        return None

    @property
    def expected_result(self) -> None:
        """Solution not provided in SIF file"""
        return None

    @property
    def expected_objective_value(self) -> None:
        """Optimal value not provided in SIF file"""
        return None
