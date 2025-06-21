import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractConstrainedMinimisation


class CHANDHEQ(AbstractConstrainedMinimisation):
    """Chandrasekhar Radiative Transfer H equation as a constrained problem.

    Source: problem 4 in
    J.J. More',
    "A collection of nonlinear model problems"
    Proceedings of the AMS-SIAM Summer seminar on the Computational
    Solution of Nonlinear Systems of Equations, Colorado, 1988.
    Argonne National Laboratory MCS-P60-0289, 1989.

    SIF input: Ph. Toint, Dec 1989.

    Classification: NOR2-RN-V-V
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Problem parameters
    N: int = 100  # Number of discretization points
    C: float = 1.0  # Problem parameter in [0,1]

    @property
    def n(self):
        """Number of variables."""
        return self.N

    @property
    def m(self):
        """Number of constraints."""
        return self.N  # One constraint per discretization point

    def objective(self, y, args):
        """Compute the objective function.

        For a constrained least squares formulation, the objective is 0.
        """
        del args, y
        return jnp.array(0.0)

    def constraint(self, y):
        """Implement the abstract constraint method."""
        eq, ineq = self.equality_constraints(y, self.args())
        return eq, ineq

    def equality_constraints(self, y, args):
        """Compute the equality constraints.

        The H-equation is:
        H(mu) = 1 - (c/2) * mu * integral_0^1 H(nu)/(mu+nu) dnu

        Discretized as:
        H_i = 1 - (c/2) * x_i * sum_j w_j * H_j / (x_i + x_j)

        Each equation becomes a constraint.
        """
        del args
        h = y  # H values at discretization points

        # Discretization points and weights
        i_vals = jnp.arange(1, self.N + 1)
        x = i_vals / self.N  # Uniform discretization on [0,1]
        w = jnp.ones(self.N) / self.N  # Equal weights

        # Compute integral term using vectorized operations
        halfc = self.C * 0.5

        # Create matrices for x[i] + x[j]
        x_i = x[:, jnp.newaxis]  # Shape (N, 1)
        x_j = x[jnp.newaxis, :]  # Shape (1, N)
        x_sum = x_i + x_j  # Shape (N, N)

        # Compute coefficients matrix: -halfc * x[i] * w[j] / (x[i] + x[j])
        w_j = w[jnp.newaxis, :]  # Shape (1, N)
        coeffs = -halfc * x_i * w_j / x_sum  # Shape (N, N)

        # Compute integral terms: sum_j coeffs[i,j] * h[i] * h[j]
        h_i = h[:, jnp.newaxis]  # Shape (N, 1)
        h_j = h[jnp.newaxis, :]  # Shape (1, N)
        integral_terms = jnp.sum(coeffs * h_i * h_j, axis=1)  # Shape (N,)

        # Constraints: H_i - 1 + integral term = 0
        constraints = h - 1.0 + integral_terms

        return constraints, None

    def y0(self):
        """Initial guess."""
        # All components set to 1.0
        return inexact_asarray(jnp.ones(self.N))

    def bounds(self):
        """Variable bounds."""
        # From pycutest, all variables have lower bound 0
        lower = jnp.zeros(self.n)
        upper = jnp.full(self.n, jnp.inf)
        return lower, upper

    def args(self):
        """No additional arguments."""
        return None

    def expected_result(self):
        """Expected optimal solution."""
        # Not provided in SIF file
        return None

    def expected_objective_value(self):
        """Expected optimal objective value."""
        # Should be 0.0 at solution
        return jnp.array(0.0)
