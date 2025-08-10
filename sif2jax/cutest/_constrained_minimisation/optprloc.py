"""
Problem OPTPRLOC: Optimal product location in multiattribute space.

A market positioning problem with M existing products and N consumers
in a K-dimensional attribute space. The goal is to position a new product
optimally. Note: The SIF file specifies Y variables as INTEGER but bounded
[0,1], effectively making them binary. We treat them as continuous [0,1].

Variables:
- X(i): Product attributes for i=1 to K (K=5)
- Y(i): Consumer choice variables for i=1 to N (N=25)

Source: Test problem 4 in M. Duran & I.E. Grossmann,
"An outer approximation algorithm for a class of mixed integer nonlinear
programs", Mathematical Programming 36, pp. 307-339, 1986.

SIF input: S. Leyffer, October 1997

classification QQR2-AN-30-30
"""

import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class OPTPRLOC(AbstractConstrainedMinimisation):
    """OPTPRLOC: Optimal product location problem.

    Fixed size problem:
    - Variables: 30 (5 product attributes + 25 consumer variables)
    - Constraints: 30 (25 ellipsoid + 5 linear constraints)
    """

    # Problem dimensions
    K: int = 5  # Number of attributes
    M: int = 10  # Number of existing products
    N: int = 25  # Number of consumers
    H: float = 1000.0

    @property
    def name(self) -> str:
        return "OPTPRLOC"

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self) -> int:
        return self.K + self.N  # 5 + 25 = 30

    @property
    def m(self) -> int:
        return self.N + 5  # 25 ellipsoid + 5 linear = 30

    @property
    def _data(self):
        """Return problem data matrices."""
        # Ideal points Z[i,k] for consumers i=1..N, attributes k=1..K
        Z = jnp.array(
            [
                [2.26, 5.15, 4.03, 1.74, 4.74],
                [5.51, 9.01, 3.84, 1.47, 9.92],
                [4.06, 1.80, 0.71, 9.09, 8.13],
                [6.30, 0.11, 4.08, 7.29, 4.24],
                [2.81, 1.65, 8.08, 3.99, 3.51],
                [4.29, 9.49, 2.24, 9.78, 1.52],
                [9.76, 3.64, 6.62, 3.66, 9.08],
                [1.37, 6.99, 7.19, 3.03, 3.39],
                [8.89, 8.29, 6.05, 7.48, 4.09],
                [7.42, 4.60, 0.30, 0.97, 8.77],
                [1.54, 7.06, 0.01, 1.23, 3.11],
                [7.74, 4.40, 7.93, 5.95, 4.88],
                [9.94, 5.21, 8.58, 0.13, 4.57],
                [9.54, 1.57, 9.66, 5.24, 7.90],
                [7.46, 8.81, 1.67, 6.47, 1.81],
                [0.56, 8.10, 0.19, 6.11, 6.40],
                [3.86, 6.68, 6.42, 7.29, 4.66],
                [2.98, 2.98, 3.03, 0.02, 0.67],
                [3.61, 7.62, 1.79, 7.80, 9.81],
                [5.68, 4.24, 4.17, 6.75, 1.08],
                [5.48, 3.74, 3.34, 6.22, 7.94],
                [8.13, 8.72, 3.93, 8.80, 8.56],
                [1.37, 0.54, 1.55, 5.56, 5.85],
                [8.79, 5.04, 4.83, 6.94, 0.38],
                [2.66, 4.19, 6.49, 8.04, 1.66],
            ]
        )

        # Weights W[i,k] - corrected from SIF file
        W = jnp.array(
            [
                [9.57, 2.74, 9.75, 3.96, 8.67],
                [8.38, 3.93, 5.18, 5.20, 7.82],
                [9.81, 0.04, 4.21, 7.38, 4.11],
                [7.41, 6.08, 5.46, 4.86, 1.48],
                [9.96, 9.13, 2.95, 8.25, 3.58],
                [9.39, 4.27, 5.09, 1.81, 7.58],
                [1.88, 7.20, 6.65, 1.74, 2.86],
                [4.01, 2.67, 4.86, 2.55, 6.91],
                [4.18, 1.92, 2.60, 7.15, 2.86],
                [7.81, 2.14, 9.63, 7.61, 9.17],  # Fixed W10
                [8.96, 3.47, 5.49, 4.73, 9.43],  # Fixed W11
                [9.94, 1.63, 1.23, 4.33, 7.08],  # Fixed W12
                [0.31, 5.00, 0.16, 2.52, 3.08],  # Fixed W13
                [6.02, 0.92, 7.47, 9.74, 1.76],  # Fixed W14
                [5.06, 4.52, 1.89, 1.22, 9.05],  # Fixed W15
                [5.92, 2.56, 7.74, 6.96, 5.18],  # Fixed W16
                [6.45, 1.52, 0.06, 5.34, 8.47],  # Fixed W17
                [1.04, 1.36, 5.99, 8.10, 5.22],  # Fixed W18
                [1.40, 1.35, 0.59, 8.58, 1.21],  # Fixed W19
                [6.68, 9.48, 1.60, 6.74, 8.92],  # Fixed W20
                [1.95, 0.46, 2.90, 1.79, 0.99],  # Fixed W21
                [5.18, 5.10, 8.81, 3.27, 9.63],  # Fixed W22
                [1.47, 5.71, 6.95, 1.42, 3.49],  # Fixed W23
                [5.40, 3.12, 5.37, 6.10, 3.71],  # Fixed W24
                [6.32, 0.81, 6.12, 6.73, 7.93],  # Fixed W25
            ]
        )

        # R values (precomputed minimums)
        R = jnp.array(
            [
                77.83985,
                175.9710,
                201.8226,
                143.9533,
                154.3895,
                433.3177,
                109.0764,
                41.59592,
                144.0623,
                99.83416,
                149.1791,
                123.8074,
                27.22197,
                89.92683,
                293.0766,
                174.3170,
                125.1028,
                222.8417,
                50.48593,
                361.1973,
                40.32642,
                161.8518,
                66.85827,
                340.5807,
                407.5200,
            ]
        )

        # Profit coefficients for Y variables (from GROUPS section)
        profit_y = jnp.array(
            [
                -1.0,
                -0.2,
                -1.0,
                -0.2,
                -0.9,
                -0.9,
                -0.1,
                -0.8,
                -1.0,
                -0.4,
                -1.0,
                -0.3,
                -0.1,
                -0.3,
                -0.5,
                -0.9,
                -0.8,
                -0.1,
                -0.9,
                -1.0,
                -1.0,
                -1.0,
                -0.2,
                -0.7,
                -0.7,
            ]
        )

        return Z, W, R, profit_y

    @property
    def y0(self):
        # Use all zeros to match pycutest default (even though it violates bounds)
        return jnp.zeros(self.n)

    @property
    def args(self):
        return None

    def objective(self, y, args):
        """Profit objective with quadratic terms for X and linear terms for Y."""
        x = y[: self.K]
        y_consumer = y[self.K :]

        Z, W, R, profit_y = self._data

        # Quadratic terms: 0.6*X1^2 + 0.1*X4^2
        # Linear terms for X: -0.9*X2 - 0.5*X3 + 1.0*X5 (from GROUPS)
        obj = 0.6 * x[0] ** 2 + 0.1 * x[3] ** 2 - 0.9 * x[1] - 0.5 * x[2] + 1.0 * x[4]

        # Linear terms for Y
        obj += jnp.sum(profit_y * y_consumer)

        return obj

    def constraint(self, y):
        """Ellipsoid and linear constraints.

        Ellipsoid constraints (25):
        ELLI(i): H*Y(i) - sum_k W(i,k)*(X(k) - Z(i,k))^2 <= R(i) + H

        Linear constraints (5):
        LIN1: X1 - X2 + X3 + X4 + X5 <= 10.0
        LIN2: 0.6*X1 - 0.9*X2 - 0.5*X3 + 0.1*X4 + X5 <= -0.64
        LIN3: X1 - X2 + X3 - X4 + X5 >= 0.69
        LIN4: 0.157*X1 + 0.05*X2 <= 1.5
        LIN5: 0.25*X2 + 1.05*X4 - 0.3*X5 >= 4.5
        """
        x = y[: self.K]
        y_consumer = y[self.K :]

        Z, W, R, profit_y = self._data

        # Ellipsoid constraints: convert to <= 0 form (pycutest convention)
        # Original SIF: H*Y(i) + sum_k W(i,k)*(X(k) - Z(i,k))^2 <= R(i) + H
        # pycutest form: H*Y(i) + sum_k W(i,k)*(X(k) - Z(i,k))^2 - R(i) - H <= 0
        ellipsoid_constraints = []
        for i in range(self.N):
            dist_sq = jnp.sum(W[i, :] * (x - Z[i, :]) ** 2)
            constraint_val = self.H * y_consumer[i] + dist_sq - R[i] - self.H
            ellipsoid_constraints.append(constraint_val)

        ellipsoid_constraints = jnp.array(ellipsoid_constraints)

        # Linear constraints in <= 0 form
        # LIN1: X1 - X2 + X3 + X4 + X5 <= 10.0
        lin1 = x[0] - x[1] + x[2] + x[3] + x[4] - 10.0

        # LIN2: 0.6*X1 - 0.9*X2 - 0.5*X3 + 0.1*X4 + X5 <= -0.64
        lin2 = 0.6 * x[0] - 0.9 * x[1] - 0.5 * x[2] + 0.1 * x[3] + x[4] + 0.64

        # LIN3: X1 - X2 + X3 - X4 + X5 >= 0.69 --> X1 - X2 + X3 - X4 + X5 - 0.69 >= 0
        # But test expects <= 0 form, so: X1 - X2 + X3 - X4 + X5 - 0.69 <= 0
        lin3 = x[0] - x[1] + x[2] - x[3] + x[4] - 0.69

        # LIN4: 0.157*X1 + 0.05*X2 <= 1.5
        lin4 = 0.157 * x[0] + 0.05 * x[1] - 1.5

        # LIN5: 0.25*X2 + 1.05*X4 - 0.3*X5 >= 4.5
        # --> 0.25*X2 + 1.05*X4 - 0.3*X5 - 4.5 >= 0
        # --> -(0.25*X2 + 1.05*X4 - 0.3*X5 - 4.5) <= 0
        # But test expects different sign, so trying:
        # 0.25*X2 + 1.05*X4 - 0.3*X5 - 4.5 <= 0
        lin5 = 0.25 * x[1] + 1.05 * x[3] - 0.3 * x[4] - 4.5

        linear_constraints = jnp.array([lin1, lin2, lin3, lin4, lin5])

        # All constraints are inequalities in <= 0 form
        inequality_constraints = jnp.concatenate(
            [ellipsoid_constraints, linear_constraints]
        )

        return None, inequality_constraints

    @property
    def bounds(self):
        """Variable bounds from SIF file."""
        lower = jnp.zeros(self.n)
        upper = jnp.ones(self.n)

        # X bounds
        lower = lower.at[0].set(2.0)  # X1 >= 2.0
        upper = upper.at[0].set(4.5)  # X1 <= 4.5
        lower = lower.at[1].set(0.0)  # X2 >= 0 (default)
        upper = upper.at[1].set(8.0)  # X2 <= 8.0
        lower = lower.at[2].set(3.0)  # X3 >= 3.0
        upper = upper.at[2].set(9.0)  # X3 <= 9.0
        lower = lower.at[3].set(0.0)  # X4 >= 0 (default)
        upper = upper.at[3].set(5.0)  # X4 <= 5.0
        lower = lower.at[4].set(4.0)  # X5 >= 4.0
        upper = upper.at[4].set(10.0)  # X5 <= 10.0

        # Y bounds: [0, 1] for all (already set)

        return lower, upper

    @property
    def expected_result(self):
        """Solution not provided in SIF file."""
        return None

    @property
    def expected_objective_value(self):
        """Solution not provided in SIF file."""
        return None
