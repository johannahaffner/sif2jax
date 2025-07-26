import jax.numpy as jnp

from ..._problem import AbstractConstrainedQuadraticProblem


class CHENHARK(AbstractConstrainedQuadraticProblem):
    """A bound-constrained version of the Linear Complementarity problem.

    Find x such that w = M x + q, x and w nonnegative and x^T w = 0,
    where

    M = (  6   -4   1   0  ........ 0 )
        ( -4    6  -4   1  ........ 0 )
        (  1   -4   6  -4  ........ 0 )
        (  0    1  -4   6  ........ 0 )
           ..........................
        (  0   ........... 0  1 -4  6 )

    and q is given.

    Source:
    B. Chen and P. T. Harker,
    SIMAX 14 (1993) 1168-1190

    SDIF input: Nick Gould, November 1993.

    classification QBR2-AN-V-V
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})
    n: int = 5000  # Number of variables
    nfree: int = 2500  # Number of variables free from bounds at solution
    ndegen: int = 500  # Number of degenerate variables at solution

    @property
    def y0(self):
        """Initial guess - zeros."""
        return jnp.zeros(self.n)

    @property
    def args(self):
        return None

    def _compute_q(self):
        """Compute the q vector based on the solution structure."""
        # From the SIF file, x values are defined as:
        # x(-1) = x(0) = 0
        # x(i) = 1 for i = 1 to nfree
        # x(i) = 0 for i = nfree+1 to n+2

        # Create extended x array with boundary conditions
        x_ext = jnp.zeros(self.n + 4)  # x(-1) to x(n+2)

        # Set x(1) to x(nfree) = 1
        for i in range(1, self.nfree + 1):
            x_ext[i + 1] = 1.0  # Shift by 2 due to x(-1), x(0)

        # Compute q = -M*x
        q = jnp.zeros(self.n + 2)

        # Q(0) = x(1)
        q = q.at[0].set(x_ext[3])  # x(1) is at index 3

        # Q(1) = 2*x(1) - x(2)
        q = q.at[1].set(2.0 * x_ext[3] - x_ext[4])

        # Q(i) = x(i+1) + x(i-1) - 2*x(i) for i = 2 to n-1
        for i in range(2, self.n):
            idx = i + 1  # Shift for x indexing
            q = q.at[i].set(x_ext[idx + 2] + x_ext[idx] - 2.0 * x_ext[idx + 1])

        # Q(n) = 2*x(n) - x(n-1)
        q = q.at[self.n].set(2.0 * x_ext[self.n + 1] - x_ext[self.n])

        # Q(n+1) = x(n)
        q = q.at[self.n + 1].set(x_ext[self.n + 1])

        return -q[: self.n]  # Return negative since we defined q = -M*x

    def objective(self, y, args):
        """Quadratic objective function.

        The objective is (1/2) x^T M x + q^T x where M is the pentadiagonal matrix
        and q is computed based on the solution structure.
        """
        del args

        q = self._compute_q()

        # Linear term: q^T x
        linear_term = jnp.dot(q, y)

        # Quadratic term: (1/2) x^T M x
        # M is pentadiagonal with pattern:
        # M[i,i] = 6, M[i,i±1] = -4, M[i,i±2] = 1

        # Diagonal terms: 6 * x_i^2
        quad_term = 6.0 * jnp.sum(y**2)

        # First off-diagonal: -4 * x_i * x_{i+1}
        if self.n > 1:
            quad_term += -8.0 * jnp.sum(y[:-1] * y[1:])

        # Second off-diagonal: 1 * x_i * x_{i+2}
        if self.n > 2:
            quad_term += 2.0 * jnp.sum(y[:-2] * y[2:])

        return linear_term + 0.5 * quad_term

    @property
    def bounds(self):
        """Variable bounds: x >= 0."""
        lower = jnp.zeros(self.n)
        upper = jnp.full(self.n, jnp.inf)
        return lower, upper

    def constraint(self, y):
        """No additional constraints beyond bounds."""
        return None, None

    @property
    def expected_result(self):
        """Expected result based on problem structure."""
        x = jnp.zeros(self.n)
        # First nfree variables are 1
        x = x.at[: self.nfree].set(1.0)
        # Next ndegen variables remain at 0 (degenerate)
        # Remaining variables are 0
        return x

    @property
    def expected_objective_value(self):
        """Expected objective value not provided."""
        return None
