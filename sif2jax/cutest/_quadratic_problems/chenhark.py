"""CHENHARK problem."""

# TODO: Human review needed
# Attempts made: [close objective values but gradient mismatch, complex
#                 pentadiagonal structure]
# Suspected issues: [boundary condition handling in pentadiagonal matrix,
#                    possible indexing issues]
# Additional resources needed: [clarification on exact SIF formulation,
#                               verification of matrix structure]

import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class CHENHARK(AbstractBoundedMinimisation):
    """A bound-constrained version of the Linear Complementarity problem.

    Find x such that w = M x + q, x and w nonnegative and x^T w = 0,
    where

    M = (  6   -4   1   0  ........ 0 )
        ( -4    6  -4   1  ........ 0 )
        (  1   -4   6  -4  ........ 0 )
        (  0    1  -4   6  ........ 0 )
           ..........................
        (  0   ........... 0  1 -4  6 )

    and q is given.

    Source:
    B. Chen and P. T. Harker,
    SIMAX 14 (1993) 1168-1190

    SDIF input: Nick Gould, November 1993.

    classification QBR2-AN-V-V
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})
    n: int = 5000  # Number of variables
    nfree: int = 2500  # Number of variables free from bounds at solution
    ndegen: int = 500  # Number of degenerate variables at solution

    @property
    def y0(self):
        """Initial guess - 0.5 everywhere."""
        return jnp.full(self.n, 0.5)

    @property
    def args(self):
        return None

    def objective(self, y, args):
        """Quadratic objective function."""
        del args

        # Create padded array for boundary conditions
        # x(-1) = x(0) = 0, then x(1) to x(n), then x(n+1) = x(n+2) = 0
        x_pad = jnp.concatenate([jnp.zeros(2), y, jnp.zeros(2)])

        # Compute the quadratic groups Q(i) for i = 0 to n+1
        # Each group has the HALFL2 type: 0.5 * value^2

        # Q(0) = x(1)
        q0 = 0.5 * x_pad[3] ** 2  # x(1) is at index 3

        # Q(1) = 2*x(1) - x(2)
        q1 = 0.5 * (2.0 * x_pad[3] - x_pad[4]) ** 2

        # Q(i) = x(i+1) + x(i-1) - 2*x(i) for i = 2 to n-1
        # Vectorized computation
        i_indices = jnp.arange(2, self.n)
        # x_pad indices: i+2 for x(i), i+3 for x(i+1), i+1 for x(i-1)
        q_middle = (
            0.5
            * (x_pad[i_indices + 3] + x_pad[i_indices + 1] - 2.0 * x_pad[i_indices + 2])
            ** 2
        )

        # Q(n) = 2*x(n) - x(n-1)
        qn = 0.5 * (2.0 * x_pad[self.n + 1] - x_pad[self.n]) ** 2

        # Q(n+1) = x(n)
        qn1 = 0.5 * x_pad[self.n + 1] ** 2

        # Sum all quadratic terms
        quad_sum = q0 + q1 + jnp.sum(q_middle) + qn + qn1

        # Compute linear terms L (vectorized)
        # Based on the SIF file, for each i in 1 to NF+ND (nfree + ndegen):
        # Q = -6*x(i) + 4*x(i+1) + 4*x(i-1) - x(i+2) - x(i-2)
        # L contributes x(i) * Q

        # For i in NF+ND+1 to N:
        # Q = -6*x(i) + 4*x(i+1) + 4*x(i-1) - x(i+2) - x(i-2) + 1
        # L contributes x(i) * Q

        nf_nd = self.nfree + self.ndegen

        # Vectorized computation for all variables
        # Create coefficient array for pentadiagonal matrix multiplication
        q_coeffs = jnp.zeros(self.n)

        # For each variable i (1 to n), compute:
        # -6*x(i) + 4*x(i+1) + 4*x(i-1) - x(i+2) - x(i-2)

        # Main diagonal: -6
        q_coeffs = -6.0 * y

        # First off-diagonal: +4 from both sides
        q_coeffs = q_coeffs.at[:-1].add(4.0 * y[1:])  # x(i+1) contribution to x(i)
        q_coeffs = q_coeffs.at[1:].add(4.0 * y[:-1])  # x(i-1) contribution to x(i)

        # Second off-diagonal: -1 from both sides
        if self.n > 2:
            q_coeffs = q_coeffs.at[:-2].add(-1.0 * y[2:])  # x(i+2) contribution to x(i)
            q_coeffs = q_coeffs.at[2:].add(-1.0 * y[:-2])  # x(i-2) contribution to x(i)

        # Handle boundary terms from padding
        # x(1) gets contribution from x(-1)=0 and x(0)=0
        # x(2) gets contribution from x(0)=0
        # x(n-1) gets contribution from x(n+1)=0
        # x(n) gets contribution from x(n+1)=0 and x(n+2)=0

        # Add constant +1 for indices > nf_nd
        if nf_nd < self.n:
            q_coeffs = q_coeffs.at[nf_nd:].add(1.0)

        # Linear sum: x(i) * q_coeff(i)
        linear_sum = jnp.dot(y, q_coeffs)

        return quad_sum + linear_sum

    @property
    def bounds(self):
        """Variable bounds: x >= 0."""
        lower = jnp.zeros(self.n)
        upper = jnp.full(self.n, jnp.inf)
        return lower, upper

    @property
    def expected_result(self):
        """Expected result based on problem structure."""
        x = jnp.zeros(self.n)
        # First nfree variables are 1
        x = x.at[: self.nfree].set(1.0)
        # Next ndegen variables remain at 0 (degenerate)
        # Remaining variables are 0
        return x

    @property
    def expected_objective_value(self):
        """Expected objective value not provided."""
        return None
