import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class READING2(AbstractConstrainedMinimisation):
    """A linear optimal control problem from Nancy Nichols with a given initial
    condition. This problem arises in tide modelling.

    Source:
    S. Lyle and N.K. Nichols,
    "Numerical Methods for Optimal Control Problems with State Constraints",
    Numerical Analysis Report 8/91, Dept of Mathematics,
    University of Reading, UK.

    SIF input: Nick Gould, July 1991.
    Classification: LLR2-MN-V-V
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})
    n: int = 2000  # Number of discretized points
    pi: float = 3.141592653589
    pi_squared: float = 3.141592653589**2
    a: float = 0.07716
    h: float = 1.0 / 2000  # 1.0 / n
    two_pi: float = 2.0 * 3.141592653589  # 2.0 * pi
    h_half: float = (1.0 / 2000) / 2.0  # h / 2.0
    h_over_8pi2: float = (1.0 / 2000) / (8.0 * 3.141592653589**2)  # h / (8*pi^2)
    h_inv: float = 1.0 / (1.0 / 2000)  # 1.0 / h

    def objective(self, y, args):
        # Variables: X1_(0), X2_(0), U(0), ..., X1_(N), X2_(N), U(N)
        # Total: 3*(N+1) variables
        # Extract variables
        x1 = y[::3]  # X1_(0), X1_(1), ..., X1_(N)
        # x2 = y[1::3]  # X2_(0), X2_(1), ..., X2_(N) - not used in objective
        u = y[2::3]  # U(0), U(1), ..., U(N)

        # Vectorized linear objective function COST
        i_indices = jnp.arange(1, self.n + 1, dtype=y.dtype)
        t_i = i_indices * self.h
        t_im1 = (i_indices - 1) * self.h
        cos_2pi_ti = jnp.cos(self.two_pi * t_i)
        cos_2pi_tim1 = jnp.cos(self.two_pi * t_im1)

        # -CCTI = cos(2πt[i]) * (-h/2)
        neg_ccti = cos_2pi_ti * (-self.h_half)
        neg_cctim1 = cos_2pi_tim1 * (-self.h_half)

        # COST: X1_(I)*-CCTI + X1_(I-1)*-CCTI-1 + U(I)*H/8PI**2 + U(I-1)*H/8PI**2
        # Combine all terms into single vectorized operation
        obj = jnp.dot(x1[1:], neg_ccti) + jnp.dot(x1[:-1], neg_cctim1)
        obj += (jnp.sum(u[1:]) + jnp.sum(u[:-1])) * self.h_over_8pi2

        return obj

    @property
    def y0(self):
        # Initial guess: all zeros (SIF file defaults)
        # Vars: X1_(0), X2_(0), U(0), X1_(1), X2_(1), U(1), ..., X1_(N), X2_(N), U(N)
        # Total: 3*(N+1) = 6003 variables
        return jnp.zeros(3 * (self.n + 1))

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        # The expected result would need to be determined from running the problem
        return self.y0

    @property
    def expected_objective_value(self):
        # Expected objective value needs to be determined
        return None

    @property
    def bounds(self):
        # Vectorized bounds for interleaved variables: X1, X2, U
        n_vars = 3 * (self.n + 1)
        lower = jnp.full(n_vars, -jnp.inf)
        upper = jnp.full(n_vars, jnp.inf)

        # X1_(0) and X2_(0) are fixed at 0
        lower = lower.at[0].set(0.0)  # X1_(0)
        upper = upper.at[0].set(0.0)
        lower = lower.at[1].set(0.0)  # X2_(0)
        upper = upper.at[1].set(0.0)

        # X1_(i) for i>0 is free (already -inf, inf)
        # X2_(i) for i>0 has bounds [-0.125, 0.125]
        x2_indices = jnp.arange(4, n_vars, 3)  # 4, 7, 10, ... (X2 positions for i>0)
        lower = lower.at[x2_indices].set(-0.125)
        upper = upper.at[x2_indices].set(0.125)

        # U(i) has bounds [-1.0, 1.0] for all i
        u_indices = jnp.arange(2, n_vars, 3)  # 2, 5, 8, ... (U positions)
        lower = lower.at[u_indices].set(-1.0)
        upper = upper.at[u_indices].set(1.0)

        return lower, upper

    def constraint(self, y):
        # Extract variables
        x1 = y[::3]  # X1_(0), X1_(1), ..., X1_(N)
        x2 = y[1::3]  # X2_(0), X2_(1), ..., X2_(N)
        u = y[2::3]  # U(0), U(1), ..., U(N)

        # Vectorized constraints for i=1 to n
        # C1_(I): X1_(I)/H - X1_(I-1)/H - 0.5*X2_(I) - 0.5*X2_(I-1) = 0
        c1 = (x1[1:] - x1[:-1]) * self.h_inv - 0.5 * x2[1:] - 0.5 * x2[:-1]

        # C2_(I): X2_(I)/H - X2_(I-1)/H - 0.5*U(I) - 0.5*U(I-1) = 0
        c2 = (x2[1:] - x2[:-1]) * self.h_inv - 0.5 * u[1:] - 0.5 * u[:-1]

        # Interleave constraints: C1_(1), C2_(1), C1_(2), C2_(2), ...
        constraints = jnp.zeros(2 * self.n)
        constraints = constraints.at[::2].set(c1)
        constraints = constraints.at[1::2].set(c2)

        return constraints, None
