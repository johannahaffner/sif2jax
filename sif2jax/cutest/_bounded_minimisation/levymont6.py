import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class LEVYMONT6(AbstractBoundedMinimisation):
    """A global optimization example due to Levy & Montalvo
    This problem is one of the parameterised set LEVYMONT5-LEVYMONT10

    Source:  problem 6 in

    A. V. Levy and A. Montalvo
    "The Tunneling Algorithm for the Global Minimization of Functions"
    SIAM J. Sci. Stat. Comp. 6(1) 1985 15:29
    https://doi.org/10.1137/0906002

    SIF input: Nick Gould, August 2021

    classification SBR2-AY-3-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Parameters
    N: int = 3  # Number of variables
    A: float = 1.0
    K: float = 10.0
    L: float = 0.25
    C: float = 0.75

    @property
    def n(self):
        """Number of variables."""
        return self.N

    @property
    def y0(self):
        """Initial guess (LEVYMONTA starting point)."""
        # Default: x1 = -8.0, x2 = 8.0, others = 8.0
        return jnp.array([-8.0, 8.0, 8.0])

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def bounds(self):
        """Bounds on the variables (all [-10, 10])."""
        lw = -10.0 * jnp.ones(self.n)
        up = 10.0 * jnp.ones(self.n)
        return lw, up

    def objective(self, y, args):
        """Compute the sum of squares objective function.

        The objective consists of:
        1. Q(i) terms: Linear terms scaled by N/π
        2. N(i) terms: Nonlinear terms with sine functions
        """
        del args  # Not used

        pi = jnp.pi
        n_over_pi = self.N / pi
        sqrt_k_pi_over_n = jnp.sqrt(self.K * pi / self.N)

        # Q(i) groups: coefficient L is scaled by N/π - vectorized
        a_minus_c = self.A - self.C
        scaled_L = n_over_pi * self.L
        q_values = scaled_L * y - a_minus_c
        q_sum_of_squares = jnp.sum(q_values**2)

        # N(1) group: sin(π(L*x1 + C))
        lx1_c = self.L * y[0] + self.C
        element_1 = jnp.sin(pi * lx1_c)
        n_1_scaled = sqrt_k_pi_over_n * element_1
        n_1_contribution = n_1_scaled**2

        # N(i) for i >= 2: (L*x(i-1) + C - A) * sin(π(L*x(i) + C)) - vectorized
        if self.N > 1:
            lx_im1_c_a = self.L * y[:-1] + self.C - self.A  # y[0] to y[N-2]
            lx_i_c = self.L * y[1:] + self.C  # y[1] to y[N-1]
            elements_i = lx_im1_c_a * jnp.sin(pi * lx_i_c)
            n_i_scaled = sqrt_k_pi_over_n * elements_i
            n_i_contributions = jnp.sum(n_i_scaled**2)
        else:
            n_i_contributions = 0.0

        return q_sum_of_squares + n_1_contribution + n_i_contributions

    @property
    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        return jnp.array(0.0)
