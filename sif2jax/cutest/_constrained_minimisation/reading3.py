import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class READING3(AbstractConstrainedMinimisation):
    """A nonlinear optimal control problem from Nancy Nichols with a periodic
    boundary condition. This problem arises in tide modelling.

    Source:
    S. Lyle and N.K. Nichols,
    "Numerical Methods for Optimal Control Problems with State Constraints",
    Numerical Analysis Report 8/91, Dept of Mathematics,
    University of Reading, UK.

    SIF input: Nick Gould, July 1991.
    Classification: OOR2-MN-V-V
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})
    n: int = 2000  # Number of discretized points
    pi: float = 3.141592653589
    a: float = 0.07716
    h: float = 1.0 / 2000  # 1.0 / n
    two_pi: float = 2.0 * 3.141592653589  # 2.0 * pi
    half_a_inv: float = 0.5 / 0.07716  # 0.5 / a
    h_inv: float = 1.0 / (1.0 / 2000)  # 1.0 / h
    h_half: float = (1.0 / 2000) / 2.0  # h / 2.0

    def objective(self, y, args):
        # Variables are interleaved: x[0], u[0], x[1], u[1], ..., x[n], u[n]
        # Extract x and u from interleaved format
        x = y[::2]  # Every even index: x[0], x[1], ..., x[n]
        u = y[1::2]  # Every odd index: u[0], u[1], ..., u[n]

        # Vectorized computation of ENERGY elements I(0) to I(N)
        i_indices = jnp.arange(self.n + 1, dtype=y.dtype)
        t_values = i_indices * self.h
        cos_2pi_t = jnp.cos(self.two_pi * t_values)

        # ENERGY element: HOVER2 * U * (X - C2PIT)^2
        energy_elements = self.h_half * u * (x - cos_2pi_t) ** 2

        # GROUP USES: Each group I(i) for i=1 to N gets -I(i) - I(i-1)
        # This is equivalent to -2 * sum(I(1:N-1)) - I(0) - I(N)
        obj = (
            -2.0 * jnp.sum(energy_elements[1 : self.n])
            - energy_elements[0]
            - energy_elements[self.n]
        )

        return obj

    @property
    def y0(self):
        # Initial guess: all zeros (SIF file defaults)
        # Variables are interleaved: x[0], u[0], x[1], u[1], ..., x[n], u[n]
        # Total: 2*(n+1) = 4002 variables
        return jnp.zeros(2 * (self.n + 1))

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
        # Bounds are interleaved: x[0], u[0], x[1], u[1], ..., x[n], u[n]
        lower = jnp.zeros(2 * (self.n + 1))
        upper = jnp.zeros(2 * (self.n + 1))

        # Set bounds for interleaved variables
        for i in range(self.n + 1):
            # x[i] bounds - all have [-0.5, 0.5]
            lower = lower.at[2 * i].set(-0.5)
            upper = upper.at[2 * i].set(0.5)

            # u[i] bounds - all have [0.0, 1.0]
            lower = lower.at[2 * i + 1].set(0.0)
            upper = upper.at[2 * i + 1].set(1.0)

        return lower, upper

    def constraint(self, y):
        # Variables are interleaved: x[0], u[0], x[1], u[1], ..., x[n], u[n]
        x = y[::2]  # Every even index: x[0], x[1], ..., x[n]
        u = y[1::2]  # Every odd index: u[0], u[1], ..., u[n]

        # Vectorized ODE constraints for i=1 to N
        i_indices = jnp.arange(1, self.n + 1, dtype=y.dtype)
        t_i = i_indices * self.h
        t_im1 = (i_indices - 1) * self.h
        cos_2pi_ti = jnp.cos(self.two_pi * t_i)
        cos_2pi_tim1 = jnp.cos(self.two_pi * t_im1)

        # Linear part: (X(I) - X(I-1))/H
        ode_constraints = (x[1:] - x[:-1]) * self.h_inv

        # CCTI * U(I) + CCTI-1 * U(I-1) where CCTI = cos(2πt[i]) * (-1/(2a))
        ode_constraints += cos_2pi_ti * (-self.half_a_inv) * u[1:]
        ode_constraints += cos_2pi_tim1 * (-self.half_a_inv) * u[:-1]

        # Nonlinear part: NC(I) + NC(I-1) where NC = P * U * X with P = 1/(2a)
        ode_constraints += self.half_a_inv * u[1:] * x[1:]
        ode_constraints += self.half_a_inv * u[:-1] * x[:-1]

        # Periodic boundary condition: X(0) - X(N) = 0 (comes last)
        period_constraint = x[0] - x[self.n]

        # Combine all constraints
        constraints = jnp.concatenate([ode_constraints, jnp.array([period_constraint])])

        # Total: N (ODE) + 1 (periodic) = 2001 constraints
        return constraints, None
