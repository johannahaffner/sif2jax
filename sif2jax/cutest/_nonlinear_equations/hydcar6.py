import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class HYDCAR6(AbstractNonlinearEquations):
    """The hydrocarbon-6 problem by Fletcher.

    Source: Problem 2a in
    J.J. More',"A collection of nonlinear model problems"
    Proceedings of the AMS-SIAM Summer Seminar on the Computational
    Solution of Nonlinear Systems of Equations, Colorado, 1988.
    Argonne National Laboratory MCS-P60-0289, 1989.

    SIF input : N. Gould, Dec 1989

    Classification: NOR2-AN-29-29
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Problem parameters
    N: int = 6
    M: int = 3
    K: int = 2

    # Constants from SIF file
    A = jnp.array([9.647, 9.953, 9.466])
    B_param = jnp.array([-2998.00, -3448.10, -3347.25])
    C = jnp.array([230.66, 235.88, 215.31])

    AL = jnp.array([0.0, 0.0, 0.0])
    AL_prime = jnp.array([37.6, 48.2, 45.4])
    AL_double_prime = jnp.array([0.0, 0.0, 0.0])

    BE = jnp.array([8425.0, 9395.0, 10466.0])
    BE_prime = jnp.array([24.2, 35.6, 31.9])
    BE_double_prime = jnp.array([0.0, 0.0, 0.0])

    FL = jnp.array([30.0, 30.0, 40.0])
    FV = jnp.array([0.0, 0.0, 0.0])

    TF = 100.0
    B_const = 40.0
    D = 60.0
    Q = 2500000.0

    @property
    def n(self):
        """Number of variables: T(0..5), X(0..5,1..3), V(0..4)"""
        return self.N + self.N * self.M + (self.N - 1)

    def num_residuals(self):
        """Number of residual equations."""
        return 29  # From classification

    def residual(self, y, args):
        """Compute the residuals for the hydrocarbon equations."""
        del args

        # Extract variables - pycutest interleaves T and X by row
        # Pattern: T(0), X(0,1), X(0,2), X(0,3), T(1), X(1,1), ...
        T = jnp.zeros(self.N)
        X = jnp.zeros((self.N, self.M))

        idx = 0
        for i in range(self.N):
            T = T.at[i].set(y[idx])
            idx += 1
            for j in range(self.M):
                X = X.at[i, j].set(y[idx])
                idx += 1

        # V(0..4) come after all T and X
        V = y[idx:]

        # Precompute PI values
        PI = jnp.ones(self.N)
        INVPI = 1.0 / PI

        residuals = []

        # Equations 2.1-(J) for J=1,2,3
        for j in range(self.M):
            eq = self.B_const * X[0, j] - V[0] * X[1, j]
            eq += (
                V[0]
                * X[0, j]
                * INVPI[0]
                * jnp.exp(self.A[j] + self.B_param[j] / (T[0] + self.C[j]))
            )
            residuals.append(eq * 100.0)  # Scaled by 100

        # Equations 2.2-(I,J) for I=1..4, J=1,2,3
        for i in range(1, self.N - 1):
            for j in range(self.M):
                if i < self.K:
                    p2_21 = self.B_const
                    p2_23 = self.B_const
                elif i == self.K:
                    p2_21 = -self.D
                    p2_23 = self.B_const
                else:
                    p2_21 = -self.D
                    p2_23 = -self.D

                eq = -V[i] * X[i + 1, j] + p2_21 * V[i]
                eq -= (
                    V[i - 1]
                    * X[i - 1, j]
                    * INVPI[i - 1]
                    * jnp.exp(self.A[j] + self.B_param[j] / (T[i - 1] + self.C[j]))
                )
                eq += V[i - 1] * X[i, j] + p2_23 * V[i - 1]
                eq += (
                    V[i]
                    * X[i, j]
                    * INVPI[i]
                    * jnp.exp(self.A[j] + self.B_param[j] / (T[i] + self.C[j]))
                )
                residuals.append(eq * 100.0)  # Scaled by 100

        # Equations 2.3-(J) for J=1,2,3
        for j in range(self.M):
            eq = -X[self.N - 1, j]
            eq += (
                X[self.N - 2, j]
                * INVPI[self.N - 2]
                * jnp.exp(self.A[j] + self.B_param[j] / (T[self.N - 2] + self.C[j]))
            )
            residuals.append(eq)

        # Equations 2.7-(I) for I=0..5
        for i in range(self.N):
            eq = -1.0
            for j in range(self.M):
                eq += (
                    X[i, j]
                    * INVPI[i]
                    * jnp.exp(self.A[j] + self.B_param[j] / (T[i] + self.C[j]))
                )
            residuals.append(eq)

        # Equation 2.8
        eq = -self.Q
        # Compute SMALLHF and BIGHF
        SMALLHF = 0.0
        BIGHF = 0.0
        for j in range(self.M):
            TEMP1 = self.TF * self.TF * self.AL_double_prime[j]
            TEMP2 = self.TF * self.AL_prime[j]
            TEMP1 = TEMP1 + TEMP2 + self.AL[j]
            TEMP1 = TEMP1 * self.FL[j]
            SMALLHF += TEMP1

            TEMP1 = self.TF * self.TF * self.BE_double_prime[j]
            TEMP2 = self.TF * self.BE_prime[j]
            TEMP1 = TEMP1 + TEMP2 + self.BE[j]
            TEMP1 = TEMP1 * self.FV[j]
            BIGHF += TEMP1

        # E81-(J), E82-(J) and E83-(J) elements
        for j in range(self.M):
            # E81-(J): EXP4PROD
            poly_be = (
                self.BE[j]
                + self.BE_prime[j] * T[0]
                + self.BE_double_prime[j] * T[0] * T[0]
            )
            exprod = INVPI[0] * jnp.exp(
                self.A[j] + self.B_param[j] / (T[0] + self.C[j])
            )
            term = V[0] * X[0, j] * exprod * poly_be
            eq += term

            # E82-(J): POLY1PRD
            poly_al = (
                self.AL[j]
                + self.AL_prime[j] * T[0]
                + self.AL_double_prime[j] * T[0] * T[0]
            )
            term = self.B_const * X[0, j] * poly_al
            eq += term

            # E83-(J): POLY2PRD
            poly_al = (
                self.AL[j]
                + self.AL_prime[j] * T[1]
                + self.AL_double_prime[j] * T[1] * T[1]
            )
            term = -1.0 * X[1, j] * (self.B_const + V[0]) * poly_al
            eq += term

        residuals.append(eq * 1e5)  # Scaled by 1e5

        # Equations 2.9-(I) for I=1..4
        for i in range(1, self.N - 1):
            eq = -SMALLHF if i < self.K else -BIGHF

            for j in range(self.M):
                # E91-(I,J): EXP3PROD
                exprod = INVPI[i - 1] * jnp.exp(
                    self.A[j] + self.B_param[j] / (T[i - 1] + self.C[j])
                )
                term = V[i - 1] * X[i - 1, j] * exprod
                poly = (
                    self.AL[j]
                    + self.AL_prime[j] * T[i - 1]
                    + self.AL_double_prime[j] * T[i - 1] * T[i - 1]
                )
                eq += poly * term

                # E92-(I,J): POLY2PRD
                poly = (
                    self.AL[j]
                    + self.AL_prime[j] * T[i - 1]
                    + self.AL_double_prime[j] * T[i - 1] * T[i - 1]
                )
                p2 = (
                    self.B_const
                    if i < self.K
                    else -self.D
                    if i > self.K
                    else self.B_const
                )
                term = -1.0 * X[i - 1, j] * (p2 + V[i - 1]) * poly
                eq += term

                # E93-(I,J): EXP3PROD
                exprod = INVPI[i] * jnp.exp(
                    self.A[j] + self.B_param[j] / (T[i] + self.C[j])
                )
                term = V[i] * X[i, j] * exprod
                poly = (
                    self.AL[j]
                    + self.AL_prime[j] * T[i]
                    + self.AL_double_prime[j] * T[i] * T[i]
                )
                eq += poly * term

                # E94-(I,J): POLY2PRD
                poly = (
                    self.AL[j]
                    + self.AL_prime[j] * T[i]
                    + self.AL_double_prime[j] * T[i] * T[i]
                )
                p2 = self.B_const if i < self.K else -self.D
                term = -1.0 * X[i, j] * (p2 + V[i]) * poly
                eq += term

            residuals.append(eq * 1e5)  # Scaled by 1e5

        return jnp.array(residuals)

    @property
    def y0(self):
        """Initial guess."""
        y0 = jnp.zeros(self.n)

        # X values from SIF file
        X_init = jnp.array(
            [
                [0.0, 0.2, 0.9],
                [0.0, 0.2, 0.8],
                [0.05, 0.3, 0.8],
                [0.1, 0.3, 0.6],
                [0.3, 0.5, 0.3],
                [0.6, 0.6, 0.0],
            ]
        )

        # Interleave T and X values
        idx = 0
        for i in range(self.N):
            y0 = y0.at[idx].set(100.0)  # T(i)
            idx += 1
            for j in range(self.M):
                y0 = y0.at[idx].set(X_init[i, j])  # X(i,j)
                idx += 1

        # V values
        for i in range(self.N - 1):
            y0 = y0.at[idx].set(300.0)
            idx += 1

        return y0

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return jnp.zeros(self.n)  # Placeholder

    @property
    def expected_objective_value(self):
        """Expected optimal objective value (0 for constrained formulation)."""
        return jnp.array(0.0)

    def constraint(self, y):
        """Returns the residuals as equality constraints."""
        return self.residual(y, self.args), None

    @property
    def bounds(self) -> tuple[jnp.ndarray, jnp.ndarray] | None:
        """No bounds for this problem."""
        return None
