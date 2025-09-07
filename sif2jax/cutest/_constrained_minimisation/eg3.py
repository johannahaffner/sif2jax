import jax.numpy as jnp
from jax import Array

from ..._misc import inexact_asarray
from ..._problem import AbstractConstrainedMinimisation


class EG3(AbstractConstrainedMinimisation):
    """The generalization of example from Section 1.2.5 of the LANCELOT users' manual.

    The number of variables is a parameter N, not merely 100 as specified in the text.

    Source:
    A. R. Conn, N. I. M. Gould and Ph. L. Toint,
    "LANCELOT: a Fortran package for large-scale nonlinear optimization
     (Release A)", Springer Series in Computational Mathematics 17,
    Springer Verlag, 1992

    SIF input: A. R. Conn, Nick Gould and Ph. L. Toint, June 1994.

    Classification: OOR2-AY-V-V
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Problem parameters
    N: int = 10000  # Number of variables (default from SIF)

    @property
    def n(self):
        """Number of variables."""
        return self.N + 1  # N x variables plus y variable

    @property
    def m(self):
        """Number of constraints."""
        return 2 * self.N  # N-1 inequality + N range constraints + 1 equality

    def objective(self, y: Array, args) -> Array:
        """Compute the objective function."""
        del args
        # Variables: y followed by x(1), ..., x(N)
        y_var = y[0]
        x_vars = y[1:]

        # Objective: 0.5 * (y + (x1 - xN) * x2)^2
        x1, x2, xN = x_vars[0], x_vars[1], x_vars[-1]
        alpha = y_var + (x1 - xN) * x2
        return 0.5 * alpha * alpha

    def constraint(self, y: Array):
        """Compute the constraints."""
        y_var = y[0]
        x_vars = y[1:]

        constraints = []

        # Inequality constraints Q(i): y + x1*x(i+1) + (1 + 2/i)*xi*xN <= 0
        # for i = 1, ..., N-1
        for i in range(self.N - 1):
            i_idx = i + 1  # 1-indexed
            constraint_val = (
                y_var
                + x_vars[0] * x_vars[i_idx]
                + (1.0 + 2.0 / i_idx) * x_vars[i] * x_vars[-1]
            )
            constraints.append(constraint_val)

        # Range constraints S(i): sin^2(x(i)) with range [0, 0.5]
        # These become: sin^2(x(i)) >= 0 and sin^2(x(i)) <= 0.5
        # Since sin^2 >= 0 always, we only need sin^2(x(i)) - 0.5 <= 0
        for i in range(self.N):
            sin_squared = jnp.sin(x_vars[i]) ** 2
            constraints.append(sin_squared - 0.5)

        # Equality constraint Q: (x1 + xN)^2 = 1
        equality_constraint = (x_vars[0] + x_vars[-1]) ** 2 - 1.0

        inequality_constraints = jnp.array(constraints)
        equality_constraints = jnp.array([equality_constraint])

        return inequality_constraints, equality_constraints

    @property
    def y0(self):
        """Initial guess."""
        # From SIF: 'DEFAULT' = 0.5, Y = 0.0
        y_start = [0.0]  # y variable
        x_start = [0.5] * self.N  # x variables
        return inexact_asarray(jnp.array(y_start + x_start))

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def bounds(self):
        """Variable bounds from SIF file."""
        # y: free (no bounds)
        # x(i): -1.0 <= x(i) <= i for i = 1, ..., N
        lower_bounds = [-jnp.inf] + [-1.0] * self.N
        upper_bounds = [jnp.inf] + [float(i) for i in range(1, self.N + 1)]

        return jnp.array(lower_bounds), jnp.array(upper_bounds)

    @property
    def expected_result(self):
        """Expected optimal solution."""
        # Not provided in SIF file
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        # Not provided in SIF file
        return None
