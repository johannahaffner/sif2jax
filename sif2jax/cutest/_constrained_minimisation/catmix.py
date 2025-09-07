# TODO: Human review needed
# Test failures in constraints, bounds, and runtime performance
# Attempts made: 1
# Suspected issues: Constraint formulation or bound specification issues
# Resources needed: Review COPS problem 14 formulation and boundary conditions

import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class CATMIX(AbstractConstrainedMinimisation):
    """Optimal mixing policy of two catalysts in tubular plug flow reactor.

    Determine the optimal mixing policy of two catalysts along the
    length of a tubular plug flow reactor involving several reactions.

    This is problem 14 in the COPS (Version 2) collection of
    E. Dolan and J. More', "Benchmarking Optimization Software with COPS",
    Argonne National Labs Technical Report ANL/MCS-246 (2000).

    SIF input: Nick Gould, November 2000.

    Classification: OOR2-AN-V-V
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Parameters from SIF file
    nh: int = 800  # Number of subintervals
    tf: float = 1.0  # Final time
    x1_0: float = 1.0  # Initial condition for x1
    x2_0: float = 0.0  # Initial condition for x2
    alpha: float = 0.0  # Smoothing parameter

    # Computed parameters
    h: float = tf / nh  # Uniform interval length
    alphah: float = alpha * h
    h_half: float = h * 0.5

    @property
    def n(self):
        """Number of variables: 3*(NH+1) = U(0..NH), X1(0..NH), X2(0..NH)."""
        return 3 * (self.nh + 1)

    def objective(self, y, args):
        """Objective: X1[NH] + X2[NH] - 1 + ALPHA*H*sum((U[i+1] - U[i])^2)."""
        del args

        n_points = self.nh + 1  # NH+1 points (0 to NH)

        # Variables: U(0)...U(NH), X1(0)...X1(NH), X2(0)...X2(NH)
        u = y[:n_points]  # Control variables
        x1 = y[n_points : 2 * n_points]  # State variable 1
        x2 = y[2 * n_points : 3 * n_points]  # State variable 2

        # Linear part: x1[nh] + x2[nh] - 1
        obj = x1[self.nh] + x2[self.nh] - 1.0

        # Smoothing term: ALPHA*H*sum((U[i+1] - U[i])^2) for i=0 to NH-1
        if self.alphah > 0:
            u_diff = u[1:] - u[:-1]  # u[i+1] - u[i] for i=0 to NH-1
            obj += self.alphah * jnp.sum(u_diff**2)

        return obj

    def constraint(self, y):
        """ODE discretization constraints."""
        n_points = self.nh + 1  # NH+1 points

        # Variables: U(0)...U(NH), X1(0)...X1(NH), X2(0)...X2(NH)
        u = y[:n_points]  # Control variables
        x1 = y[n_points : 2 * n_points]  # State variable 1
        x2 = y[2 * n_points : 3 * n_points]  # State variable 2

        equality_constraints = []

        # ODE constraints for i=0 to NH-1:
        # ODE1: x1[i] - x1[i+1] + (h/2)*(u[i]*(10*x2[i] - x1[i]) +
        #                                u[i+1]*(10*x2[i+1] - x1[i+1])) = 0
        # ODE2: x2[i] - x2[i+1] + (h/2)*((u[i]-1)*x2[i] + u[i]*(x1[i] - 10*x2[i]) +
        #       (u[i+1]-1)*x2[i+1] + u[i+1]*(x1[i+1] - 10*x2[i+1]))

        for i in range(self.nh):  # i = 0 to NH-1
            # ODE1 constraint
            ode1 = x1[i] - x1[i + 1]
            ode1 += self.h_half * (
                u[i] * (10 * x2[i] - x1[i]) + u[i + 1] * (10 * x2[i + 1] - x1[i + 1])
            )
            equality_constraints.append(ode1)

            # ODE2 constraint
            ode2 = x2[i] - x2[i + 1]
            ode2 += self.h_half * ((u[i] - 1) * x2[i] + u[i] * (x1[i] - 10 * x2[i]))
            ode2 += self.h_half * (
                (u[i + 1] - 1) * x2[i + 1] + u[i + 1] * (x1[i + 1] - 10 * x2[i + 1])
            )
            equality_constraints.append(ode2)

        equality_constraints = jnp.array(equality_constraints)

        return equality_constraints, None  # No inequality constraints

    @property
    def y0(self):
        """Initial guess from SIF file."""
        n_points = self.nh + 1

        # Initialize all variables to their SIF starting values
        u = jnp.zeros(n_points)  # U(i) = 0.0 for all i
        x1 = jnp.ones(n_points)  # X1(i) = 1.0 for all i
        x2 = jnp.zeros(n_points)  # X2(i) = 0.0 for all i

        return jnp.concatenate([u, x1, x2])

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def bounds(self):
        """Variable bounds."""
        n_points = self.nh + 1
        n_vars = 3 * n_points

        # Default: all variables unbounded
        lower = jnp.full(n_vars, -jnp.inf)
        upper = jnp.full(n_vars, jnp.inf)

        # Control bounds: U(i) ∈ [0, 1] for all i
        lower = lower.at[:n_points].set(0.0)
        upper = upper.at[:n_points].set(1.0)

        # Initial conditions: X1(0) = X1_0, X2(0) = X2_0
        lower = lower.at[n_points].set(self.x1_0)  # X1(0)
        upper = upper.at[n_points].set(self.x1_0)
        lower = lower.at[2 * n_points].set(self.x2_0)  # X2(0)
        upper = upper.at[2 * n_points].set(self.x2_0)

        return lower, upper

    @property
    def expected_result(self):
        """Expected solution (not provided in SIF)."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value from SIF."""
        # From SIF: -4.7185D-02 for NH=800
        return jnp.array(-0.047185)
