import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class ELATTAR(AbstractConstrainedMinimisation):
    """A nonlinear minmax problem in six variables.

    The problem is nonconvex and has several local minima.

    Source:
    R.A. El-Attar, M. Vidyasagar and S.R.K. Dutta,
    "An algorithm for l_1-approximation",
    SINUM 16, pp.70-86, 1979.

    SIF input: Ph. Toint, Nov 1993.

    Classification: LOR2-AN-7-102
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return 7  # 6 x variables + 1 u variable

    def objective(self, y, args):
        """Compute the objective function: minimize u."""
        del args
        # The objective is simply u (the 7th variable)
        return y[6]

    def constraint(self, y):
        """Returns the inequality constraints.

        Constraints are:
        u >= f_i(x) - y_i for all i (51 constraints)
        u >= -(f_i(x) - y_i) for all i (51 constraints)

        Rewritten as:
        f_i(x) - y_i - u <= 0
        -(f_i(x) - y_i) - u <= 0
        """
        # Extract variables
        x = y[:6]
        u = y[6]

        # Generate data points
        t_values = jnp.arange(51, dtype=jnp.float64) * 0.1

        # Compute y_i values
        y_i = (
            0.5 * jnp.exp(t_values)
            - jnp.exp(-2.0 * t_values)
            + 0.5 * jnp.exp(-3.0 * t_values)
            + 1.5 * jnp.exp(-1.5 * t_values) * jnp.sin(7.0 * t_values)
            + jnp.exp(-2.5 * t_values) * jnp.sin(5.0 * t_values)
        )

        # Compute f_i(x) = x1 * exp(-x2*t) * cos(x3*t + x4) + x5 * exp(-x6*t)
        # First element type: x1 * exp(-x2*t) * cos(x3*t + x4)
        a1 = -x[1] * t_values  # -x2 * t
        b1 = x[2] * t_values + x[3]  # x3 * t + x4
        et1 = x[0] * jnp.exp(a1) * jnp.cos(b1)

        # Second element type: x5 * exp(-x6*t)
        a2 = -x[5] * t_values  # -x6 * t
        et2 = x[4] * jnp.exp(a2)

        # f_i(x) = et1 + et2
        f_i = et1 + et2

        # Residuals
        residuals = f_i - y_i

        # Inequality constraints
        # The SIF file has F(I) and MF(I) groups
        # F(I): u - (f_i - y_i) >= 0  => f_i - y_i - u <= 0
        # MF(I): u - (-(f_i - y_i)) >= 0  => -(f_i - y_i) - u <= 0

        # Interleave the constraints as they appear in the SIF file
        constraints = []
        for i in range(51):
            constraints.append(residuals[i] - u)  # F(I)
            constraints.append(-residuals[i] - u)  # MF(I)

        # Return only inequality constraints
        return None, jnp.array(constraints)

    @property
    def y0(self):
        """Initial guess."""
        # From SIF file, the more interesting starting point
        return jnp.array([-2.0, -2.0, 7.0, 0.0, -2.0, 1.0, 0.0])

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def bounds(self):
        """No bounds on variables."""
        return None

    @property
    def expected_result(self):
        """Expected optimal solution (approximate)."""
        # The SIF file doesn't provide the solution
        return jnp.zeros(7)  # Placeholder

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        # From SIF file: either 0.1427066255 or 74.206179244
        # The smaller value seems more likely for a well-solved problem
        return jnp.array(0.1427066255)
