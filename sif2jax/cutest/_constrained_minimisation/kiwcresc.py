import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractConstrainedMinimisation


class KIWCRESC(AbstractConstrainedMinimisation):
    """A nonlinear minmax problem in two variables.

    Source:
    K.C. Kiwiel,
    "Methods of Descent for Nondifferentiable Optimization"
    Lectures Notes in Mathematics 1133, Springer Verlag, 1985.

    SIF input: Ph. Toint, Nov 1993.

    classification LQR2-AN-3-2
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y: Array, args) -> Array:
        """Objective function: minimize U (the third variable)."""
        return y[2]  # U is the third variable

    def constraint(self, y: Array):
        """Constraint functions.

        From SIF:
        F1: X1^2 + X2^2 - X2 - U <= -2
        F2: -X1^2 - X2^2 + 3*X2 - U <= 2
        """
        x1, x2, u = y[0], y[1], y[2]

        # F1: X1^2 + X2^2 - X2 - U <= -2
        f1 = x1**2 + x2**2 - x2 - u + 2.0

        # F2: -X1^2 - X2^2 + 3*X2 - U <= 2
        f2 = -(x1**2) - x2**2 + 3.0 * x2 - u - 2.0

        # No equality constraints
        equality_constraints = jnp.array([])

        # Inequality constraints (should be <= 0)
        inequality_constraints = jnp.array([f1, f2])

        return equality_constraints, inequality_constraints

    @property
    def y0(self):
        """Initial guess for the optimization problem."""
        # From SIF START POINT section
        return jnp.array([-1.5, 2.0, 0.0])  # X1, X2, U

    @property
    def bounds(self):
        """Variable bounds. All variables are free."""
        return None

    @property
    def args(self):
        """Additional arguments for the objective function."""
        return None

    @property
    def expected_result(self):
        """Expected result of the optimization problem."""
        # This is a minmax problem, the solution should minimize the maximum
        # The optimal solution involves balancing the two constraint functions
        return jnp.array([0.0, 1.0, 0.0])  # Approximate solution

    @property
    def expected_objective_value(self):
        """Expected value of the objective at the solution."""
        # From SIF comment: solution around 0.0
        return jnp.array(0.0)
