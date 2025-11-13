import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class READING5(AbstractConstrainedMinimisation):
    """A nonlinear optimal control problem from Nancy Nichols with a given
    initial condition. This problem arises in tide modelling.

    Source: a variant upon a problem in
    S. Lyle and N.K. Nichols,
    "Numerical Methods for Optimal Control Problems with State Constraints",
    Numerical Analysis Report 8/91, Dept of Mathematics,
    University of Reading, UK.

    SIF input: Ph. Toint, Aug 1992
    Classification: OOR2-MN-V-V

    Note: This is similar to READING4 but with equality constraints instead of
    inequality constraints.
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})
    n: int = 5000  # Number of discretized points (N in SIF)
    pi: float = 3.141592653589
    a: float = 0.07716
    h: float = 1.0 / 5000  # 1.0 / n
    two_pi: float = 2.0 * 3.141592653589
    half_a_inv: float = 0.5 / 0.07716  # 1/(2a)
    two_a: float = 2.0 * 0.07716  # 2a
    two_a_over_h: float = (2.0 * 0.07716) / (1.0 / 5000)  # 2a/h

    def objective(self, y, args):
        x = y  # Only X variables in this problem

        # Vectorized computation of ENERGY elements I(1) to I(N)
        # Create time indices for i=1 to i=N
        i_indices = jnp.arange(1, self.n + 1, dtype=y.dtype)
        t_i = i_indices * self.h
        cos_2pi_ti = jnp.cos(self.two_pi * t_i)

        # Compute energy for all elements
        # ENERGY element: (F - X) * (X - XP) where F = cos(2πt)
        energy = (cos_2pi_ti - x[1:]) * (x[1:] - x[:-1])

        # Apply coefficients: -1.0 for first and last, -2.0 for middle
        coefficients = jnp.ones(self.n) * -2.0
        coefficients = coefficients.at[0].set(-1.0)  # First element
        coefficients = coefficients.at[-1].set(-1.0)  # Last element

        # The SIF file defines group J with scale 1/A
        # However, pycutest appears to divide by 1/a instead of multiplying
        # We match this behavior for consistency
        return jnp.sum(coefficients * energy) * self.a

    @property
    def y0(self):
        # Initial guess: all zeros (pycutest convention)
        # Note: SIF file specifies X(0) fixed at 0.25, but pycutest uses 0.0
        # for the initial value when drop_fixed_variables=False
        return jnp.zeros(self.n + 1)

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        return self.y0

    @property
    def expected_objective_value(self):
        return None

    @property
    def bounds(self):
        # Vectorized bounds creation
        lower = jnp.full(self.n + 1, -0.5)
        upper = jnp.full(self.n + 1, 0.5)

        # X(0) is fixed at 0.25
        lower = lower.at[0].set(0.25)
        upper = upper.at[0].set(0.25)

        return lower, upper

    def constraint(self, y):
        x = y  # Only X variables

        # Vectorized EQUALITY constraints U(1) to U(N)
        # Each has form: UC(i) * (2a/h) = 0
        # UC element: (x[i] - x[i-1]) / (cos(2πt[i]) - x[i])

        # Create time indices for i=1 to i=N
        i_indices = jnp.arange(1, self.n + 1, dtype=y.dtype)
        t_i = i_indices * self.h
        cos_2pi_ti = jnp.cos(self.two_pi * t_i)

        # UC element: (X - XP) / (F - X) where F = cos(2πt)
        numerator = x[1:] - x[:-1]
        denominator = cos_2pi_ti - x[1:]

        # UC element: (X - XP) / (F - X) where F = cos(2πt)
        # Direct division - JAX will handle edge cases
        # Note: This can produce inf/nan when denominator is exactly 0
        uc = numerator / denominator

        # Constraint value: UC * (2a/h)
        equalities = jnp.asarray(uc) * self.two_a_over_h

        # Return (equalities, inequalities=None)
        # The equalities should equal 0.0
        return equalities, None
