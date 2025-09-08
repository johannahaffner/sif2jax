import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS111LNP(AbstractConstrainedMinimisation):
    """Hock and Schittkowski problem 111 modified by Todd Plantenga.

    This problem is a chemical equilibrium problem involving 3 linear
    equality constraints.

    Source: problem 111 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.
    This problem has been modified on 20 Oct 92 by Todd Plantenga as follows.
    The bound constraints, which are inactive at the solution,
    are removed.

    SIF input: Nick Gould, August 1991 and T. Plantenga, October 1992.

    classification OOR2-AN-10-3
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return jnp.array(10)

    @property
    def y0(self):
        """Initial guess."""
        # From START POINT: 'DEFAULT' -2.3
        return jnp.full(10, -2.3, dtype=jnp.float64)

    @property
    def args(self):
        return None

    def objective(self, y, args):
        """Objective function."""
        del args

        # Constants C1-C10 from SIF file
        c = jnp.array(
            [
                -6.089,
                -17.164,
                -34.054,
                -5.914,
                -24.721,
                -14.986,
                -24.100,
                -10.708,
                -26.662,
                -22.179,
            ],
            dtype=jnp.float64,
        )

        # The objective is sum over i of exp(x_i) * (c_i + x_i - log(sum of exp(x_j)))
        # This comes from the OBJ element type definition
        exp_x = jnp.exp(y)
        sum_exp = jnp.sum(exp_x)
        log_sum = jnp.log(sum_exp)

        # Each O(i) element contributes exp(x_i) * (c_i + x_i - log_sum)
        obj = jnp.sum(exp_x * (c + y - log_sum))

        return obj

    @property
    def bounds(self):
        """Variable bounds."""
        # From BOUNDS: FR (free/unbounded) - return None for unbounded problems
        return None

    def constraint(self, y):
        """Constraint functions."""
        # Constraints use exp(X_i) values
        exp_x = jnp.exp(y)

        # From GROUP USES:
        # CON1: E1 + 2*E2 + 2*E3 + E6 + E10 = 2.0
        con1 = exp_x[0] + 2.0 * exp_x[1] + 2.0 * exp_x[2] + exp_x[5] + exp_x[9] - 2.0

        # CON2: E4 + 2*E5 + E6 + E7 = 1.0
        con2 = exp_x[3] + 2.0 * exp_x[4] + exp_x[5] + exp_x[6] - 1.0

        # CON3: E3 + E7 + E8 + 2*E9 + E10 = 1.0
        con3 = exp_x[2] + exp_x[6] + exp_x[7] + 2.0 * exp_x[8] + exp_x[9] - 1.0

        # All constraints are equality constraints (type E in SIF)
        equalities = jnp.array([con1, con2, con3])

        return equalities, None

    @property
    def expected_result(self):
        """Expected result not provided in SIF file."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value from SIF file."""
        # From OBJECT BOUND section
        return jnp.array(-47.707579)
