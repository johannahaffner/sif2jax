"""Compatibility wrapper for converting between constraint conventions (DRAFT).

This module provides a wrapper class that converts constraints between different
sign conventions used by PyCUTEst and Optimistix:

- PyCUTEst: Uses mixed conventions (g(x) >= 0, g(x) <= 0, and range constraints)
- Optimistix: Uses consistent g(x) >= 0 convention for all inequality constraints

The wrapper allows problems implemented in Optimistix convention to be tested
against PyCUTEst reference implementations by converting constraints appropriately.
"""

from typing import Any

from jaxtyping import Array, ArrayLike

from sif2jax import AbstractConstrainedMinimisation


class PyCUTEstCompatibilityWrapper(AbstractConstrainedMinimisation):
    """Wrapper that converts Optimistix-convention constraints to PyCUTEst format.

    This wrapper takes a problem implemented with Optimistix's g(x) >= 0 convention
    and converts its constraints to match PyCUTEst's mixed convention for testing.

    The wrapper:
    1. Passes through all problem methods unchanged except constraint()
    2. Converts inequality constraints based on their types:
       - 'G' constraints: passed through (already g(x) >= 0)
       - 'L' constraints: negated (from -g(x) >= 0 to g(x) <= 0)
       - 'R' constraints: reconstructed from two inequalities to single ranged form
       - 'E' constraints: passed through unchanged

    Constraint type information is expected to be available as attributes on the
    wrapped problem instance.

    Attributes:
        problem: The underlying problem instance with Optimistix conventions
    """

    problem: AbstractConstrainedMinimisation

    @property
    def y0_iD(self) -> int:
        """Pass through y0_iD from wrapped problem."""
        return self.problem.y0_iD

    @property
    def provided_y0s(self) -> frozenset:
        """Pass through provided_y0s from wrapped problem."""
        return self.problem.provided_y0s

    # Pass through all methods to the underlying problem
    def objective(self, y: ArrayLike, args: Any) -> Array:
        return self.problem.objective(y, args)

    def y0(self) -> Array:
        return self.problem.y0()

    def args(self) -> Any:
        return self.problem.args()

    def expected_result(self) -> Array | None:
        return self.problem.expected_result()

    def expected_objective_value(self) -> Array | None:
        return self.problem.expected_objective_value()

    def bounds(self) -> tuple[Array, Array] | None:
        return self.problem.bounds()

    @property
    def name(self) -> str:
        return self.problem.name

    def constraint(self, y: ArrayLike) -> tuple[Array | None, Array | None]:
        """Convert constraints from Optimistix to PyCUTEst format.

        This method transforms constraints from consistent g(x) >= 0 format
        to PyCUTEst's mixed format based on constraint_types metadata.

        Args:
            y: Point at which to evaluate constraints

        Returns:
            Tuple of (equality_constraints, inequality_constraints) in PyCUTEst format

        Raises:
            NotImplementedError: Always (to be implemented based on feedback)
        """
        raise NotImplementedError("Constraint conversion logic to be implemented")
