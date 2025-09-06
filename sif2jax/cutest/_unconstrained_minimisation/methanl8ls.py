import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# TODO: Human review needed
# Issues to resolve:
# 1. Starting value mismatch with pycutest
# 2. Objective function formulation differs significantly (discrepancy ~961k)
# 3. Complex SIF element structure needs detailed analysis
# 4. Constraint formulation requires deeper understanding of collocation method
class METHANL8LS(AbstractUnconstrainedMinimisation):
    """METHANL8LS problem - The methanol-8 problem by Fletcher (least squares version).

    Source: Problem 2c in
    J.J. More',"A collection of nonlinear model problems"
    Proceedings of the AMS-SIAM Summer Seminar on the Computational
    Solution of Nonlinear Systems of Equations, Colorado, 1988.
    Argonne National Laboratory MCS-P60-0289, 1989.

    SIF input: N. Gould, Dec 1989.
    Least-squares version of METHANL8.SIF, Nick Gould, Jan 2020.

    Classification: SUR2-MN-31-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Problem parameters
    N = 8  # Number of stages
    M = 2  # Number of components
    K = 2  # Control stage
    NC = 3  # Number of collocation points

    # Constants from SIF file
    A = jnp.array([18.5751, 18.3443])
    B = jnp.array([-3632.649, -3841.2203])
    C = jnp.array([239.2, 228.0])

    AL = jnp.array([0.0, 0.0])
    AL_PRIME = jnp.array([15.97, 18.1])
    AL_DOUBLE_PRIME = jnp.array([0.0422, 0.0])

    BE = jnp.array([9566.67, 10834.67])
    BE_PRIME = jnp.array([-1.59, 8.74])
    BE_DOUBLE_PRIME = jnp.array([0.0422, 0.0])

    FL = jnp.array([451.25, 684.25])
    FV = jnp.array([0.0, 0.0])

    TF = 89.0
    B_CONST = 693.37
    D = 442.13
    Q = 8386200.0

    PI = jnp.array([1210.0, 1200.0, 1190.0, 1180.0, 1170.0, 1160.0, 1150.0, 1140.0])

    # Collocation points (roots of 3rd degree Legendre polynomial)
    RHO = jnp.array([0.5, 0.8872983346, 0.1127016654])

    @property
    def n(self):
        """Number of variables: T (8) + X (16) + V (7) = 31."""
        return self.N + self.N * self.M + (self.N - 1)

    def _unpack_variables(self, y):
        """Unpack optimization variables."""
        idx = 0

        # Temperature variables T[0:7] (8 variables)
        T = y[idx : idx + self.N]
        idx += self.N

        # Composition variables X[i,j] for i=0:7, j=1:2 (16 variables)
        X = y[idx : idx + self.N * self.M].reshape(self.N, self.M)
        idx += self.N * self.M

        # Volume variables V[0:6] (7 variables)
        V = y[idx : idx + (self.N - 1)]

        return T, X, V

    def objective(self, y, args):
        """Objective function: sum of squared residuals."""
        del args
        T, X, V = self._unpack_variables(y)

        residuals = []

        # Constraint 2.1-(j): X(0,j) * B = given values
        for j in range(self.M):
            residual = X[0, j] * self.B_CONST - (self.FL[j] if j == 0 else self.FL[j])
            residuals.append(residual)

        # Constraint 2.2-(i,j): Material balance equations
        for i in range(1, self.N - 1):
            for j in range(self.M):
                # Simplified material balance (placeholder)
                residual = X[i, j] - X[i - 1, j]  # Simplified constraint
                residuals.append(residual)

        # Constraint 2.3-(j): Final composition
        for j in range(self.M):
            residual = X[self.N - 1, j]  # Should match some target
            residuals.append(residual)

        # Constraint 2.7-(i): Vapor-liquid equilibrium
        for i in range(self.N):
            # VLE constraint involving exponential terms
            sum_term = 0.0
            for j in range(self.M):
                # Simplified VLE constraint
                exp_term = jnp.exp(self.A[j] + self.B[j] / (T[i] + self.C[j]))
                sum_term += X[i, j] * exp_term / self.PI[i]
            residual = sum_term - 1.0
            residuals.append(residual)

        # Additional constraints (2.8, 2.9) - simplified
        # Energy balance
        energy_residual = jnp.sum(T) - self.Q / 1000  # Scaled for numerical stability
        residuals.append(energy_residual)

        # Volume constraints
        for i in range(self.N - 1):
            if i < len(V):  # Ensure we don't go out of bounds
                vol_residual = V[i] - 900.0  # Target volume around 900
                residuals.append(vol_residual)

        # Return sum of squared residuals
        residuals_array = jnp.array(residuals)
        return jnp.sum(residuals_array**2)

    @property
    def y0(self):
        """Initial point from SIF file."""
        # From START POINT section of SIF file
        y0 = jnp.zeros(self.n)
        idx = 0

        # Temperature initial values
        T_init = jnp.array([120.0, 110.0, 100.0, 88.0, 86.0, 84.0, 80.0, 76.0])
        y0 = y0.at[idx : idx + self.N].set(T_init)
        idx += self.N

        # Composition initial values
        X_init = jnp.array(
            [
                [0.09203, 0.908],
                [0.1819, 0.8181],
                [0.284, 0.716],
                [0.3051, 0.6949],
                [0.3566, 0.6434],
                [0.468, 0.532],
                [0.6579, 0.3421],
                [0.8763, 0.1237],
            ]
        )
        y0 = y0.at[idx : idx + self.N * self.M].set(X_init.flatten())
        idx += self.N * self.M

        # Volume initial values
        V_init = jnp.array([886.37, 910.01, 922.52, 926.46, 935.56, 952.83, 975.73])
        y0 = y0.at[idx : idx + (self.N - 1)].set(V_init)

        return y0

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        return None  # No known analytical solution

    @property
    def expected_objective_value(self):
        return jnp.array(0.0)  # Least square problems aim for zero residual
