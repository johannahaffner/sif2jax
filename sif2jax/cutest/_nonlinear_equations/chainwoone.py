import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class CHAINWOONE(AbstractNonlinearEquations):
    """
    The chained Woods problem, a variant on Woods function
    This is a nonlinear equation variant of CHAINWOO

    This problem is a sum of n/2 sets of 6 terms, each of which is
    assigned its own group.  For a given set i, the groups are
    A(i), B(i), C(i), D(i), E(i) and F(i). Groups A(i) and C(i) contain 1
    nonlinear element each, denoted Y(i) and Z(i).

    The problem dimension is defined from the number of these sets.
    The number of problem variables is then 2 times + 2 as large

    This version uses a slightly unorthodox expression of Woods
    function as a sum of squares (see Buckley)

    Source:  problem 8 in
    A.R.Conn,N.I.M.Gould and Ph.L.Toint,
    "Testing a class of methods for solving minimization
    problems with simple bounds on their variables,
    Mathematics of Computation 50, pp 399-430, 1988.

    SIF input: Nick Gould and Ph. Toint, Dec 1995.
              Nick Gould (nonlinear equation version), Jan 2019

    classification NOR2-AN-V-V

    Note: This problem has a constant objective value of -1.0 due to the CONST group
    in the SIF formulation. This doesn't affect the solution of the nonlinear equations.
    """

    ns: int = 1999  # Number of sets
    n: int = 4000  # n = 2 * ns + 2
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def starting_point(self) -> Array:
        x = jnp.full(self.n, -2.0, dtype=jnp.float64)
        # Special starting values for first 4 variables
        x = x.at[0].set(-3.0)  # X1
        x = x.at[1].set(-1.0)  # X2
        x = x.at[2].set(-3.0)  # X3
        x = x.at[3].set(-1.0)  # X4
        return x

    def num_residuals(self) -> int:
        # 6 groups per set (A, B, C, D, E, F)
        return 6 * self.ns

    def residual(self, y: Array, args) -> Array:
        """Compute the residuals of the chained Woods problem"""
        ns = self.ns

        # Precompute constants
        root10 = jnp.sqrt(10.0)
        rootp1 = jnp.sqrt(0.1)
        r90 = jnp.sqrt(90.0)

        residuals = []

        # Process each set i = 1, ..., ns
        j = 3  # Starting from 4 in Fortran (0-indexed: 3)
        for i in range(ns):
            # Group A(i): 0.1 * (y[j-2] - y[j-3]^2)
            a_i = 0.1 * (y[j - 2] - y[j - 3] ** 2)
            residuals.append(a_i)

            # Group B(i): -y[j-3] - 1.0
            b_i = -y[j - 3] - 1.0
            residuals.append(b_i)

            # Group C(i): (1/sqrt(90)) * (y[j] - y[j-1]^2)
            c_i = (1.0 / r90) * (y[j] - y[j - 1] ** 2)
            residuals.append(c_i)

            # Group D(i): -y[j-1] - 1.0
            d_i = -y[j - 1] - 1.0
            residuals.append(d_i)

            # Group E(i): sqrt(0.1) * (y[j-2] + y[j]) - 2.0
            e_i = rootp1 * (y[j - 2] + y[j]) - 2.0
            residuals.append(e_i)

            # Group F(i): sqrt(10) * (y[j-2] - y[j])
            f_i = root10 * (y[j - 2] - y[j])
            residuals.append(f_i)

            j += 2

        return jnp.array(residuals)

    def objective(self, y: Array, args) -> Array:
        """Returns the constant objective value of -1.0."""
        return jnp.array(-1.0)

    def y0(self) -> Array:
        """Initial guess for the optimization problem."""
        return self.starting_point()

    def args(self):
        """Additional arguments for the residual function."""
        return None

    def expected_result(self) -> Array | None:
        """Expected result of the optimization problem."""
        # The solution has all variables equal to 1
        return jnp.ones(self.n, dtype=jnp.float64)

    def expected_objective_value(self) -> Array | None:
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)
