import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS86(AbstractConstrainedMinimisation):
    """Hock and Schittkowski problem 86.

    Source: problem 86 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    SIF input: Nick Gould, August 1991.

    classification OLR2-AN-5-10
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return jnp.array(5)

    @property
    def y0(self):
        """Initial guess."""
        # From START POINT: default 0.0, X5=1.0
        return jnp.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=jnp.float64)

    @property
    def args(self):
        return None

    def objective(self, y, args):
        """Objective function."""
        del args
        x1, x2, x3, x4, x5 = y[0], y[1], y[2], y[3], y[4]
        x = jnp.array([x1, x2, x3, x4, x5])

        # From SIF GROUPS: ZN OBJ X(J) E(J) - linear terms
        e = jnp.array([-15.0, -27.0, -36.0, -18.0, -12.0])
        linear_part = jnp.dot(e, x)

        # From SIF GROUP USES: ZE OBJ D(J) D(J) where D(J) is CUBE element XJ³
        d = jnp.array([4.0, 8.0, 10.0, 6.0, 2.0])
        cubic_part = jnp.dot(d, x**3)

        # From SIF GROUP USES: ZE OBJ C(I,J) C(I,J) where C(I,J) is PROD element XI*XJ
        # C(I,J) coefficients from the SIF file
        C = jnp.array(
            [
                [30.0, -20.0, -10.0, 32.0, -10.0],
                [-20.0, 39.0, -6.0, -31.0, 32.0],
                [-10.0, -6.0, 10.0, -6.0, -10.0],
                [32.0, -31.0, -6.0, 39.0, -20.0],
                [-10.0, 32.0, -10.0, -20.0, 30.0],
            ]
        )

        # Quadratic terms: sum over I,J of C(I,J) * XI * XJ
        quadratic_part = jnp.dot(x, jnp.dot(C, x))

        return linear_part + cubic_part + quadratic_part

    @property
    def bounds(self):
        """Variable bounds."""
        # From test error: pycutest has lower bounds of 0.0
        lower = jnp.zeros(5, dtype=jnp.float64)
        upper = jnp.full(5, jnp.inf, dtype=jnp.float64)
        return lower, upper

    def constraint(self, y):
        """Constraint functions."""
        x1, x2, x3, x4, x5 = y[0], y[1], y[2], y[3], y[4]
        x = jnp.array([x1, x2, x3, x4, x5])

        # Coefficient matrix A(I,J) from SIF file
        A = jnp.array(
            [
                [-16.0, 0.0, -3.5, 0.0, 0.0, 2.0, -1.0, -1.0, 1.0, 1.0],
                [2.0, -2.0, 0.0, -2.0, -9.0, 0.0, -1.0, -2.0, 2.0, 1.0],
                [0.0, 0.0, 2.0, 0.0, -2.0, -4.0, -1.0, -3.0, 3.0, 1.0],
                [1.0, 4.0, 0.0, -4.0, 1.0, 0.0, -1.0, -2.0, 4.0, 1.0],
                [0.0, 2.0, 0.0, -1.0, -2.8, 0.0, -1.0, -1.0, 5.0, 1.0],
            ]
        ).T  # Transpose since A(I,J) where I is constraint, J is variable

        # Constants B(I)
        b = jnp.array([-40.0, -2.0, -0.25, -4.0, -4.0, -1.0, -40.0, -60.0, 5.0, 1.0])

        # Linear constraints: A @ x - b = 0 for equality constraints
        # But looking at the SIF structure, these appear to be inequality constraints
        # The constraint is A @ x >= b, so we return A @ x - b >= 0
        inequalities = A @ x - b

        return None, inequalities

    @property
    def expected_result(self):
        """Expected result from SIF file."""
        # From HS86SOL section
        return jnp.array(
            [0.3, 0.33346761, 0.4, 0.42831010, 0.22396487], dtype=jnp.float64
        )

    @property
    def expected_objective_value(self):
        """Expected objective value not provided in SIF file."""
        return None
