import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class BDVALUES(AbstractNonlinearEquations):
    """BDVALUES problem.

    The Boundary Value problem.
    This is a nonlinear equations problems with the original
    starting point scaled by the factor X0SCALE.
    See BDVALUE for the original formulation, corresponding to X0SCALE = 1.0.

    Source:  a variant of problem 28 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    SIF input: Ph. Toint, June 2003.

    classification NOR2-MN-V-V
    """

    # Default parameters
    NDP: int = 10002  # Number of discretization points
    X0SCALE: float = 1000.0  # Starting point scaling

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        """Compute the objective function."""
        del args, y
        # For nonlinear equations problems, the objective is typically constant
        return jnp.array(0.0)

    def residual(self, y):
        """Compute the residuals for the system.

        y contains all NDP variables including fixed boundary conditions.
        x(1) = 0 and x(NDP) = 0 are fixed boundary conditions.
        """
        ndp = self.NDP
        x = y  # y already contains all variables

        # Useful parameters
        h = 1.0 / (ndp - 1)
        h2 = h * h
        halfh2 = 0.5 * h2

        # Vectorized computation of interior residuals for i = 1 to NDP-2
        # Basic finite difference part: -x(i-1) + 2*x(i) - x(i+1)
        residuals = -x[:-2] + 2.0 * x[1:-1] - x[2:]

        # Nonlinear part
        # For constraint i (1-indexed), the element parameter B = i*h + 1
        # The element computes (V + B)**3 where V is x(i)
        i_vals = jnp.arange(1, ndp - 1, dtype=jnp.float64)
        ih_vals = i_vals * h  # IH = i*h
        b_vals = ih_vals + 1.0  # B = IH + 1 (from line 107 in SIF)
        vplusb = x[1:-1] + b_vals
        residuals += halfh2 * (vplusb**3)

        return residuals

    def constraint(self, y):
        """Return the constraint values as required by the abstract base class."""
        # For nonlinear equations, all residuals are equality constraints
        residuals = self.residual(y)
        return (residuals, None)

    @property
    def y0(self):
        """Initial guess for all variables."""
        ndp = self.NDP
        h = 1.0 / (ndp - 1)
        x0scale = self.X0SCALE

        # All variables including fixed boundary values
        y = jnp.zeros(ndp)

        # Set all values (boundary values remain 0)
        for i in range(1, ndp - 1):
            # From SIF: TI = IH * (IH - 1) where IH = i * h
            ih = float(i) * h
            ti = ih * (ih - 1.0)
            y = y.at[i].set(ti * x0scale)

        return y

    @property
    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    @property
    def expected_result(self):
        """Expected optimal solution."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        # From SIF file: SOLTN = 0.0
        return jnp.array(0.0)

    @property
    def n(self):
        """Number of variables (including fixed ones)."""
        return self.NDP

    @property
    def m(self):
        """Number of equations/residuals."""
        return self.NDP - 2

    @property
    def bounds(self):
        """Returns the bounds on the variable y."""
        # X(1) and X(NDP) are fixed at 0
        lower = jnp.full(self.NDP, -jnp.inf)
        upper = jnp.full(self.NDP, jnp.inf)

        # Fix boundary values
        lower = lower.at[0].set(0.0)
        upper = upper.at[0].set(0.0)
        lower = lower.at[-1].set(0.0)
        upper = upper.at[-1].set(0.0)

        return lower, upper
