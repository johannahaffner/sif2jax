import jax.numpy as jnp
from jax import Array

from ..._misc import inexact_asarray
from ..._problem import AbstractNonlinearEquations


class CYCLOOCF(AbstractNonlinearEquations):
    """
    The cyclooctane molecule configuration problem (no fixed variables).

    The cyclooctane molecule is comprised of eight carbon atoms aligned
    in an equally spaced ring. When they take a position of minimum
    potential energy so that next-neighbours are equally spaced.

    Given positions v_1, ..., v_p in R^3 (with p = 8 for cyclooctane),
    and given a spacing c^2 we have that

       ||v_i - v_i+1,mod p||^2 = c^2 for i = 1,..,p, and
       ||v_i - v_i+2,mod p||^2 = 2p/(p-2) c^2

    This is a version of CYCLOOCT.SIF without the fixed variables.

    Source:
    an extension of the cyclooctane molecule configuration space as
    described in (for example)

     E. Coutsias, S. Martin, A. Thompson & J. Watson
     "Topology of cyclooctane energy landscape"
     J. Chem. Phys. 132-234115 (2010)

    SIF input: Nick Gould, Feb 2020.

    classification  NQR2-MN-V-V
    """

    p: int = 10000  # Number of molecules
    c: float = 1.0  # Radius
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self) -> int:
        """Number of variables: Y2, Z2, and X3..XP, Y3..YP, Z3..ZP
        Total: 2 + 3*(P-2) = 3*P - 4
        """
        return 3 * self.p - 4

    def _get_positions(self, y: Array) -> Array:
        """Convert variable vector to full position array (p, 3)"""
        p = self.p
        positions = jnp.zeros((p, 3))

        # v_1 = (0, 0, 0) - fixed
        # v_2 = (0, Y2, Z2) where X2=0 is fixed
        positions = positions.at[1, 1].set(y[0])  # Y2
        positions = positions.at[1, 2].set(y[1])  # Z2

        # v_3 to v_p: X(i), Y(i), Z(i) for i = 3 to P
        # Variables y[2:] contain [X3, Y3, Z3, X4, Y4, Z4, ...]
        remaining_positions = y[2:].reshape(p - 2, 3)
        positions = positions.at[2:].set(remaining_positions)

        return positions

    def constraint(self, y: Array):
        """Compute the constraint equations for cyclooctane configuration"""
        p = self.p
        c2 = self.c**2
        sc2 = (2.0 * p / (p - 2)) * c2

        positions = self._get_positions(y)

        # Nearest neighbor constraints: ||v_i - v_{i+1}||^2 = c^2
        next_indices = (jnp.arange(p) + 1) % p
        diff_next = positions - positions[next_indices]
        dist_sq_next = jnp.sum(diff_next**2, axis=1)
        constraints_a = dist_sq_next - c2

        # Next-next neighbor constraints: ||v_i - v_{i+2}||^2 = 2p/(p-2)*c^2
        next2_indices = (jnp.arange(p) + 2) % p
        diff_next2 = positions - positions[next2_indices]
        dist_sq_next2 = jnp.sum(diff_next2**2, axis=1)
        constraints_b = dist_sq_next2 - sc2

        # Interleave constraints as A(1), B(1), A(2), B(2), ..., A(P), B(P)
        # This matches the SIF file constraint ordering
        eq_constraints = jnp.stack([constraints_a, constraints_b], axis=1).flatten()

        # No inequality constraints for this problem
        return eq_constraints, None

    @property
    def y0(self) -> Array:
        """Initial point: Y2, Z2 = 0, then i/P for molecule i"""
        p = self.p
        n = self.n

        # Initialize all variables
        initial = jnp.zeros(n)

        # Y2 and Z2 start at 0 (per SIF file START POINT section)
        # initial[0] and initial[1] are already 0

        # Vectorized initialization for X3..XP, Y3..YP, Z3..ZP
        # Create values for molecules 3 to p
        molecule_indices = jnp.arange(3, p + 1)
        values = molecule_indices / p

        # Each molecule has 3 coordinates (X, Y, Z) all set to the same value
        remaining_values = jnp.repeat(values, 3)

        # Set all remaining positions at once
        initial = initial.at[2:].set(remaining_values)

        return inexact_asarray(initial)

    @property
    def args(self):
        return None

    @property
    def expected_result(self) -> None:
        """Solution represents minimum energy configuration"""
        return None

    @property
    def bounds(self):
        """No bounds for this problem"""
        return None

    @property
    def expected_objective_value(self) -> Array:
        """Expected optimal objective value: 0"""
        return inexact_asarray(0.0)
