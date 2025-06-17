import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class ALLINITA(AbstractConstrainedMinimisation):
    """The ALLINITA function.

    A problem with "all in it". Intended to verify that changes to LANCELOT are safe.
    Multiple constrained version.

    Source: N. Gould: private communication.
    SIF input: Nick Gould, March 2013.

    Classification: OOR2-AY-4-4

    Note: Variable X4 is fixed at 2.0 in the original formulation.
    For compatibility with pycutest, we handle this by removing the fixed variable.
    """

    @property
    def n(self):
        """Number of variables (excluding fixed X4)."""
        return 3

    @property
    def m(self):
        """Number of constraints."""
        return 4

    def objective(self, y, args):
        del args
        x1, x2, x3 = y
        x4 = 2.0  # Fixed value

        # FT3: x1^2
        ft3 = x1**2

        # FT4: x2^2 + (x3 + x4)^2
        ft4 = x2**2 + (x3 + x4) ** 2

        # FT5: -3 + x4 + sin(x3)^2 + (x1 * x2)^2
        ft5 = -3 + x4 + jnp.sin(x3) ** 2 + (x1 * x2) ** 2

        # FT6: sin(x3)^2
        ft6 = jnp.sin(x3) ** 2

        # FT2: 1 + x3
        ft2 = 1 + x3

        # FNT1: x4^2
        fnt1 = x4**2

        # FNT2: (1 + x4)^2
        fnt2 = (1 + x4) ** 2

        # FNT3: x2^2 + x2^4
        fnt3 = x2**2 + x2**4

        # FNT4: x3^2 + (x4 + x1)^2 + x3^4 + ((x4 + x1)**2)**2
        fnt4 = x3**2 + (x4 + x1) ** 2
        fnt4_squared = fnt4**2

        # FNT5: 4 + x1 + sin(x4)^2 + (x2 * x3)^2
        fnt5 = 4 + x1 + jnp.sin(x4) ** 2 + (x2 * x3) ** 2

        # FNT6: sin(x4)^2
        fnt6 = jnp.sin(x4) ** 2

        return (
            ft2
            + ft3
            + ft4
            + ft5
            + ft6
            + fnt1
            + fnt2
            + fnt3
            + fnt4_squared
            + fnt5
            + fnt6
        )

    def y0(self):
        # Starting point not given in SIF, using zeros for free variables
        return jnp.zeros(3)

    def args(self):
        return None

    def constraint(self, y):
        x1, x2, x3 = y

        # Equality constraints:
        # C1: x1^2 + x2^2 = 1
        # L2: x1 + x3 = 0.25
        eq_constraints = jnp.array(
            [
                x1**2 + x2**2 - 1.0,  # C1
                x1 + x3 - 0.25,  # L2
            ]
        )

        # Inequality constraints (raw values as pycutest returns them):
        # C2 (G type): x2^2 + x3^2 >= 0
        # L1 (L type): x1 + x2 + x3 <= 1.5
        ineq_constraints = jnp.array(
            [
                x2**2 + x3**2,  # C2: raw value
                x1 + x2 + x3 - 1.5,  # L1: raw value
            ]
        )

        return eq_constraints, ineq_constraints

    def equality_constraints(self):
        """Mark which constraints are equalities."""
        return jnp.array([True, True, False, False])

    def bounds(self):
        # X1: free
        # X2 >= 1.0
        # X3: -1e10 <= x3 <= 1.0
        lower = jnp.array([-jnp.inf, 1.0, -1e10])
        upper = jnp.array([jnp.inf, jnp.inf, 1.0])
        return lower, upper

    def expected_result(self):
        """Expected optimal solution."""
        return None

    def expected_objective_value(self):
        """Expected optimal objective value."""
        return None
