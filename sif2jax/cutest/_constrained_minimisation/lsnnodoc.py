import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class LSNNODOC(AbstractConstrainedMinimisation):
    """A small network problem from the User's Guide to LSNNO.

    Source:
    D. Tuyttens,
    "A User's Guide to LSNNO, a Fortran subroutine for large scale
    nonlinear optimization problems",
    Report 89/00, Department of Mathematics, FUNDP, Namur (Belgium), 1989.

    SIF input: J-M COLLIN, July 1990.

    Classification: ONR2-AY-5-4
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        """Compute the objective function."""
        del args

        x1, x2, x3, x4, x5 = y[0], y[1], y[2], y[3], y[4]

        # E1: EXPO element - TZ * EXP(T) where T=X1+X3, TZ=X2
        e1 = x2 * jnp.exp(x1 + x3)

        # E2: SQ2 element - X^2 * Y^2 where X=X3, Y=X4
        e2 = x3**2 * x4**2

        # E3: ISQ element - (X-Y)^2 where X=X3, Y=X5
        e3 = (x3 - x5) ** 2

        return e1 + e2 + e3

    def constraint(self, y):
        """Compute the constraints."""
        x1, x2, x3, x4, x5 = y[0], y[1], y[2], y[3], y[4]

        # Equality constraints
        # C1: X1 + X2 = 10
        c1 = x1 + x2 - 10.0

        # C2: -X1 - X3 + X4 = 0
        c2 = -x1 - x3 + x4

        # C3: -X2 + X3 + X5 = 0
        c3 = -x2 + x3 + x5

        # C4: -X4 - X5 = -10
        c4 = -x4 - x5 + 10.0

        equality_constraints = jnp.array([c1, c2, c3, c4])

        return equality_constraints, None

    @property
    def n_var(self):
        """Number of variables."""
        return 5

    @property
    def n_con(self):
        """Number of constraints."""
        return 4

    @property
    def y0(self):
        """Initial guess."""
        return jnp.array([4.0, 6.0, 2.0, 6.0, 4.0])

    @property
    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    @property
    def bounds(self):
        """Bounds on variables."""
        # From SIF:
        # X1: [2.0, 4.0]
        # X2: [6.0, 8.0]
        # X3: [0.0, 5.0]
        # X4, X5: free (no explicit bounds)
        lower = jnp.array([2.0, 6.0, 0.0, -jnp.inf, -jnp.inf])
        upper = jnp.array([4.0, 8.0, 5.0, jnp.inf, jnp.inf])
        return lower, upper

    @property
    def expected_result(self):
        """Expected optimal solution."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        return None

    @property
    def y0s(self):
        """Starting points."""
        return {0: self.y0}
