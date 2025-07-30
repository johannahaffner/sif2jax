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

        y contains only the free variables x(2) to x(NDP-1).
        x(1) = 0 and x(NDP) = 0 are fixed boundary conditions.
        """
        ndp = self.NDP

        # Construct full x vector with boundary conditions
        x = jnp.zeros(ndp)
        # x[0] = 0.0 (already zero)
        x = x.at[1 : ndp - 1].set(y)  # Set free variables
        # x[ndp-1] = 0.0 (already zero)

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
        """Initial guess for free variables only."""
        ndp = self.NDP
        h = 1.0 / (ndp - 1)
        x0scale = self.X0SCALE

        # Only the free variables
        y = jnp.zeros(ndp - 2)

        # Set interior values
        for i in range(1, ndp - 1):
            # From SIF: TI = IH * (IH - 1) where IH = i * h
            ih = float(i) * h
            ti = ih * (ih - 1.0)
            y = y.at[i - 1].set(ti * x0scale)  # Note: y[i-1] corresponds to x[i]

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
        """Number of free variables."""
        return self.NDP - 2

    @property
    def m(self):
        """Number of equations/residuals."""
        return self.NDP - 2

    @property
    def bounds(self):
        """Returns the bounds on the variable y."""
        # No bounds for this problem
        return None
