import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class GOFFIN(AbstractConstrainedMinimisation):
    """
    GOFFIN - A linear minmax problem in 50 variables.

    Source:
    M.M. Makela,
    "Nonsmooth optimization",
    Ph.D. thesis, Jyvaskyla University, 1990

    SIF input: Ph. Toint, Nov 1993
    comments updated Feb 2001.

    Classification: LLR2-AN-51-50
    """

    # Required attributes
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def y0(self):
        """Initial guess from SIF file."""
        # From SIF: X(I) = I - 25.5 for I = 1..50
        x_init = jnp.arange(1.0, 51.0) - 25.5
        # U starts at 0.0 (auxiliary variable for minmax)
        u_init = jnp.array([0.0])
        return jnp.concatenate([x_init, u_init])

    @property
    def bounds(self):
        """Variable bounds - all variables are free."""
        return None

    def objective(self, y, args):
        """
        Objective function: minimize U (auxiliary variable).
        From SIF: XN OBJ U 1.0
        """
        del args
        # U is the last variable (index 50)
        u = y[-1]
        return u

    def constraint(self, y):
        """
        Constraint functions for the minmax problem.

        From SIF file: XL F(I) U -1.0 X(I) 50.0 and X(J) -1.0 for all J

        This creates constraints of the form F(I) <= 0 where:
        F(I) = -1*U + 50*X(I) + sum_{j=1}^{50} (-1)*X(j)
        F(I) = -U + 50*X(I) - sum_j X(j)

        Looking at expected Jacobian structure, constraint should be:
        F(I) = sum_j X(j) + 50*X(I) - U <= 0
        or equivalently: sum_j X(j) + 49*X(I) - U <= 0

        Actually, it should be: 50*X(i) - sum_j X(j) - U <= 0
        Which gives Jacobian: [∂F/∂X(j)] = -1 for j≠i, 49 for j=i, -1 for U
        """
        x = y[:-1]  # First 50 variables
        u = y[-1]  # Last variable

        # Compute sum of all x variables
        sum_x = jnp.sum(x)

        constraints = []
        for i in range(50):
            # F(I) = 50*X(I) - sum_j X(j) - U <= 0
            # This gives: 49*X(I) + sum_{j≠i} X(j) - U <= 0
            # Which simplifies to: -U + 50*X(I) - sum_j X(j) <= 0
            c_i = -u + 50.0 * x[i] - sum_x
            constraints.append(c_i)

        # Return (None, inequality_constraints) since XL means <=
        return None, jnp.array(constraints)

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value from SIF file."""
        return jnp.array(0.0)

    @property
    def expected_result(self):
        """Expected result from SIF file."""
        return None
