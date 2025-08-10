"""
Problem OPTMASS: Optimal control of a particle on a frictionless plane.

A particle of unit mass moves on a frictionless plane under the action of a
controlling force whose magnitude may not exceed unity. At time=0, the particle
moves through the origin in the direction of the positive x-axis with speed SPEED.
The cost function maximizes final distance from origin while minimizing final speed.

Variables (for j=1,2 and i=0 to N+1):
- X(j,i): position coordinates at time i/N
- V(j,i): velocity components at time i/N
- F(j,i): force components at time i/N (only for i=0 to N)

Source:
M. Gawande and J. Dunn,
"A Projected Newton Method in a Cartesian Product of Balls",
JOTA 59(1): 59-69, 1988.

SIF input: Ph. Toint, June 1990.

classification QQR2-AN-V-V
"""

import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class OPTMASS(AbstractConstrainedMinimisation):
    """OPTMASS: Optimal control of particle on frictionless plane.

    Problem size depends on discretization parameter N:
    - Variables: 6*(N+2) - 2 total (X, V, F for each time step)
    - Constraints: 4*(N+1) equality + (N+1) inequality constraints

    Variable ordering (from SIF file):
    For i=0 to N: X(1,i), V(1,i), F(1,i), X(2,i), V(2,i), F(2,i)
    For i=N+1: X(1,N+1), V(1,N+1), X(2,N+1), V(2,N+1)
    """

    # Problem parameters
    N: int = 5000  # Default from SIF file (n=30010)
    SPEED: float = 0.01  # Initial speed
    PEN: float = 0.335  # Penalty parameter for final speed

    @property
    def name(self) -> str:
        return "OPTMASS"

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self) -> int:
        # For i=0 to N: 6 variables each (X1, X2, V1, V2, F1, F2)
        # For i=N+1: 4 variables (X1, X2, V1, V2)
        return 6 * (self.N + 1) + 4

    @property
    def m(self) -> int:
        # A(j,i): 2*(N+1) equations
        # B(j,i): 2*(N+1) equations
        # C(i): (N+1) inequalities
        return 4 * (self.N + 1) + (self.N + 1)

    def _get_variable(self, var_type, j, i):
        """Get index of variable X, V, or F at position (j,i).

        var_type: 'X', 'V', or 'F'
        j: component (1 or 2)
        i: time step (0 to N+1)
        """
        if i <= self.N:
            # For i=0 to N: each timestep has 6 variables
            # Order: X(1,i), V(1,i), F(1,i), X(2,i), V(2,i), F(2,i)
            base = 6 * i
            if j == 1:
                if var_type == "X":
                    return base + 0
                elif var_type == "V":
                    return base + 1
                else:  # F
                    return base + 2
            else:  # j == 2
                if var_type == "X":
                    return base + 3
                elif var_type == "V":
                    return base + 4
                else:  # F
                    return base + 5
        else:  # i == N+1
            # For i=N+1: 4 variables
            # Order: X(1,N+1), V(1,N+1), X(2,N+1), V(2,N+1)
            base = 6 * (self.N + 1)
            if j == 1:
                if var_type == "X":
                    return base + 0
                else:  # V
                    return base + 1
            else:  # j == 2
                if var_type == "X":
                    return base + 2
                else:  # V
                    return base + 3

    @property
    def y0(self):
        # Initial point: all zeros except V(1,0) = SPEED
        y0_vals = jnp.zeros(self.n)
        # V(1,0) is at index 2 (after X(1,0), X(2,0))
        v1_0_idx = self._get_variable("V", 1, 0)
        y0_vals = y0_vals.at[v1_0_idx].set(self.SPEED)
        return y0_vals

    @property
    def args(self):
        return None

    def objective(self, y, args):
        """Maximize distance, minimize speed:
        -X(1,N+1)^2 - X(2,N+1)^2 + PEN*(V(1,N+1)^2 + V(2,N+1)^2).
        """
        N = self.N

        # Final position and velocity
        x1_final = y[self._get_variable("X", 1, N + 1)]
        x2_final = y[self._get_variable("X", 2, N + 1)]
        v1_final = y[self._get_variable("V", 1, N + 1)]
        v2_final = y[self._get_variable("V", 2, N + 1)]

        # Objective: minimize negative distance + penalty on speed
        return -(x1_final**2 + x2_final**2) + self.PEN * (v1_final**2 + v2_final**2)

    def constraint(self, y):
        """State equations and force magnitude constraints (vectorized).

        Equality constraints (SIF order: A(1,1), B(1,1), A(2,1), B(2,1), ...):
        A(j,i): X(j,i) - X(j,i-1) - V(j,i-1)/N - F(j,i-1)/(2N^2) = 0
        B(j,i): V(j,i) - V(j,i-1) - F(j,i-1)/N = 0

        Inequality constraints:
        C(i): F(1,i)^2 + F(2,i)^2 <= 1
        """
        N = self.N
        h = 1.0 / N  # time step
        h2_half = h * h / 2.0

        # Extract all variables efficiently
        # For i=0 to N: reshape to (N+1, 6) where each row is [X1, V1, F1, X2, V2, F2]
        vars_main = y[: 6 * (N + 1)].reshape((N + 1, 6))
        X_main = vars_main[:, [0, 3]]  # Shape: (N+1, 2) - X(j,i) for j=1,2 and i=0 to N
        V_main = vars_main[:, [1, 4]]  # Shape: (N+1, 2) - V(j,i) for j=1,2 and i=0 to N
        F_main = vars_main[:, [2, 5]]  # Shape: (N+1, 2) - F(j,i) for j=1,2 and i=0 to N

        # For i=N+1: X(j,N+1), V(j,N+1) for j=1,2
        vars_final = y[6 * (N + 1) :].reshape((1, 4))
        X_final = vars_final[:, [0, 2]]  # Shape: (1, 2) - X(j,N+1) for j=1,2
        V_final = vars_final[:, [1, 3]]  # Shape: (1, 2) - V(j,N+1) for j=1,2

        # Combine position and velocity arrays for full time series
        X_all = jnp.vstack([X_main, X_final])  # Shape: (N+2, 2)
        V_all = jnp.vstack([V_main, V_final])  # Shape: (N+2, 2)

        # State equations for i=1 to N+1 (vectorized)
        i_indices = jnp.arange(1, N + 2)  # i from 1 to N+1

        # A(j,i): position equations for all j and i
        # X(j,i) - X(j,i-1) - h*V(j,i-1) - h2_half*F(j,i-1) = 0
        A_eq = (
            X_all[i_indices]
            - X_all[i_indices - 1]
            - h * V_all[i_indices - 1]
            - h2_half * F_main[i_indices - 1]
        )

        # B(j,i): velocity equations for all j and i
        # V(j,i) - V(j,i-1) - h*F(j,i-1) = 0
        B_eq = V_all[i_indices] - V_all[i_indices - 1] - h * F_main[i_indices - 1]

        # Combine constraints in SIF order: A(1,i), B(1,i), A(2,i), B(2,i) for each i
        equality_constraints = jnp.concatenate(
            [
                jnp.stack(
                    [A_eq[:, 0], B_eq[:, 0], A_eq[:, 1], B_eq[:, 1]], axis=1
                ).flatten()
            ]
        )

        # Force magnitude constraints for i=0 to N (vectorized)
        # F(1,i)^2 + F(2,i)^2 <= 1 converted to pycutest form:
        # F(1,i)^2 + F(2,i)^2 - 1 <= 0
        F_squared_sum = jnp.sum(F_main**2, axis=1)  # Shape: (N+1,)
        inequality_constraints = F_squared_sum - 1.0

        return equality_constraints, inequality_constraints

    @property
    def bounds(self):
        """Variable bounds from SIF file.

        Fixed values:
        - X(1,0) = 0, X(2,0) = 0 (start at origin)
        - V(1,0) = SPEED, V(2,0) = 0 (initial velocity)
        All other variables are free.
        """
        lower = jnp.full(self.n, -jnp.inf)
        upper = jnp.full(self.n, jnp.inf)

        # X(1,0) = 0.0
        idx = self._get_variable("X", 1, 0)
        lower = lower.at[idx].set(0.0)
        upper = upper.at[idx].set(0.0)

        # X(2,0) = 0.0
        idx = self._get_variable("X", 2, 0)
        lower = lower.at[idx].set(0.0)
        upper = upper.at[idx].set(0.0)

        # V(1,0) = SPEED
        idx = self._get_variable("V", 1, 0)
        lower = lower.at[idx].set(self.SPEED)
        upper = upper.at[idx].set(self.SPEED)

        # V(2,0) = 0.0
        idx = self._get_variable("V", 2, 0)
        lower = lower.at[idx].set(0.0)
        upper = upper.at[idx].set(0.0)

        return lower, upper

    @property
    def expected_result(self):
        """Solution not provided in SIF file."""
        return None

    @property
    def expected_objective_value(self):
        """Solution not provided in SIF file."""
        return None
