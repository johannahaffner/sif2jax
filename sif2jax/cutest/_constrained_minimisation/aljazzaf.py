import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class ALJAZZAF(AbstractConstrainedMinimisation):
    """The ALJAZZAF problem from the CUTEst collection.

    Source:
    M. Aljazzaf,
    "Multiplier methods with partial elimination of constraints for
    nonlinear programming",
    PhD Thesis, North Carolina State University, Raleigh, 1990.

    SDIF input: Ph. Toint, May 1990.

    Classification: QQR2-AN-V-V

    Problem: A quadratic programming problem with a single equality constraint.
    The problem has variable size controlled by parameters N and N1.
    Default size: N=1000, N1=500
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Problem parameters (from SIF file defaults)
    n: int = 1000  # Number of variables
    n1: int = 500  # Upper range of active lower bounds (must be < n)
    biga: float = 100.0  # Problem parameter A

    def objective(self, y, args):
        """Compute the objective function using fully vectorized operations."""
        del args

        # Compute A(i) coefficients
        asq = self.biga * self.biga
        asq_minus_1 = asq - 1.0
        f = asq_minus_1 / (self.n - 1)
        f2 = f / self.biga

        i_indices = jnp.arange(self.n, dtype=y.dtype)
        a_coeffs = self.biga - i_indices * f2

        # Compute objective shifts in a vectorized manner
        # x1 (index 0): shift = 0.5
        # x2 to xn1 (indices 1 to n1-1): shift = -1.0
        # xn1+1 to xn (indices n1 to n-1): shift = 1.0
        indices = jnp.arange(self.n)
        obj_shifts = jnp.where(
            indices == 0, 0.5, jnp.where(indices < self.n1, -1.0, 1.0)
        )

        # Single vectorized operation: sum of a_i * (x_i - shift_i)^2
        return jnp.sum(a_coeffs * jnp.square(y - obj_shifts))

    def constraint(self, y):
        """Compute the constraint function using vectorized operations."""
        # Compute B(i) coefficients
        asq = self.biga * self.biga
        asq_minus_1 = asq - 1.0
        f = asq_minus_1 / (self.n - 1)

        i_indices = jnp.arange(self.n, dtype=y.dtype)
        b_coeffs = i_indices * f + 1.0
        minus_b1 = -b_coeffs[0]

        # Linear term for first variable
        linear_term = minus_b1 * y[0]

        # Squared terms for remaining variables
        # x2 to xn1: shift = 0.0 (no shift)
        # xn1+1 to xn: shift = 1.0
        indices = jnp.arange(self.n)
        con_shifts = jnp.where(indices < self.n1, 0.0, 1.0)

        # Mask to exclude first element from squared terms
        mask = indices > 0
        shifted_y = y - con_shifts
        squared_terms = jnp.sum(jnp.where(mask, b_coeffs * jnp.square(shifted_y), 0.0))

        con = linear_term + squared_terms - minus_b1

        # Return as tuple (equality constraints, inequality constraints)
        return jnp.array([con]), None

    @property
    def bounds(self):
        """Returns the bounds on variables.

        Variables have lower bounds of 0 (non-negative).
        """
        lower = jnp.zeros(self.n)
        upper = jnp.full(self.n, jnp.inf)
        return (lower, upper)

    @property
    def y0(self):
        """Initial point: all zeros."""
        return jnp.zeros(self.n)

    @property
    def args(self):
        return None

    @property
    def n_var(self):
        """Number of variables."""
        return self.n

    @property
    def n_con(self):
        """Number of constraints (1 equality constraint)."""
        return 1

    @property
    def expected_result(self):
        """The optimal solution is not provided in the SIF file."""
        return None

    @property
    def expected_objective_value(self):
        """From SIF file: *LO SOLTN 75.004996 (for default size)."""
        if self.n == 1000 and self.n1 == 500:
            return jnp.array(75.004996)
        return None
