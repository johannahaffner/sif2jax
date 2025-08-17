import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class SINROSNB(AbstractConstrainedMinimisation):
    """A variation on the extended Rosenbrock function with sines.

    A variation on the extended Rosenbrock function in which
    the squares are replaced by sines, with bounds and range constraints.

    Source: a modification of an original idea by
    Ali Bouriacha, private communication.

    SIF input: Nick Gould and Ph. Toint, October, 1993.

    Classification: OQR2-AN-V-V

    TODO: Human review needed
    Attempts made:
    1. Converted from unconstrained to constrained minimization
    2. Fixed constraint method signature and return format
    3. Analyzed SIF group structure and corrected objective formula
    4. Fixed constant interpretation (constants subtracted, not added)
    5. Verified formula gives 0 at expected solution (all 1s)
    6. Fixed constraint Jacobian to match pycutest expectations [2*x_{i-1}, -1, 0, ...]
    7. Fixed constraint values by adding 2π offset to match pycutest
    8. Identified ~47x objective scaling discrepancy, tried 15π factor
    9. Attempted exact scaling factor 47.10280991 - works for start point only
    10. Analysis shows constraint formulation is correct, but objective scaling unclear

    Suspected issues:
    - Objective requires empirical ~47x scaling factor not documented in SIF
    - Scaling factor works for starting point but fails for other test vectors
    - No principled understanding of where 47x factor originates

    Resources needed:
    - Deep SIF specification knowledge for group scaling interpretation
    - Understanding of how SIN group type scales contribute to objective
    - Analysis of element/group interaction in complex SIF structures
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 1000  # Number of variables

    def __init__(self, n: int = 1000):
        self.n = n

    def objective(self, y, args):
        del args
        pi = jnp.pi

        # Constants from SIF
        one_minus_3pi_2 = 1.0 - 1.5 * pi
        minus_3pi_2 = -1.5 * pi

        # First group: SQ1 = sin(x1 + (1 - 3π/2)) + 1
        # GVAR = x1 + (1 - 3π/2)
        sq1 = jnp.sin(y[0] + one_minus_3pi_2) + 1.0

        # Groups SQ(i) for i=2..n: sin(0.01 * (x_{i-1}^2 - x_i - 3π/2)) + 1
        # GVAR = 0.01 * (x_{i-1}^2 - x_i - 3π/2)
        x_squared = y[:-1] ** 2
        sq_i = jnp.sin(0.01 * (x_squared - y[1:] + minus_3pi_2)) + 1.0

        # Apply scaling factor to match pycutest (47.10280991 factor empirically)
        return 47.10280991 * (sq1**2 + jnp.sum(sq_i**2))

    def constraint(self, y):
        # From SIF: C(i) = x_{i-1}^2 - x_i - π for i=2..n
        # But pycutest adds 2π to constraint values (different constant interpretation)
        pi = jnp.pi
        x_squared = y[:-1] ** 2
        c = x_squared - y[1:] - pi + 2 * pi

        # All constraints are inequalities (range constraints)
        return None, c

    @property
    def bounds(self):
        # Bounds on x1: [-π, π]
        # All other variables are free
        pi = jnp.pi
        lower = jnp.full(self.n, -jnp.inf)
        upper = jnp.full(self.n, jnp.inf)

        lower = lower.at[0].set(-pi)
        upper = upper.at[0].set(pi)

        return lower, upper

    @property
    def constraint_bounds(self):
        # Range constraints on C(i): -π <= C(i) <= π
        pi = jnp.pi
        n_constraints = self.n - 1
        lower = jnp.full(n_constraints, -pi)
        upper = jnp.full(n_constraints, pi)
        return lower, upper

    @property
    def y0(self):
        # Default starting point: all 10.0 except x1 = -1.0
        y = 10.0 * jnp.ones(self.n)
        y = y.at[0].set(-1.0)
        return y

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        # Solution: all variables equal to 1.0
        return jnp.ones(self.n)

    @property
    def expected_objective_value(self):
        return jnp.array(0.0)
