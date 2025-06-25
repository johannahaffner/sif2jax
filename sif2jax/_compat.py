"""Compatibility wrapper for pycutest problems."""

from dataclasses import dataclass
from typing import Any

from jaxtyping import Array, ArrayLike, Int, PyTree, Scalar

from ._problem import AbstractConstrainedMinimisation


@dataclass(frozen=True)
class ConstraintConventions:
    """Immutable container for constraint type information.

    Attributes:
        inequality_types: Tuple of constraint types ('L', 'G', 'R') for each inequality.
                        'R' indicates a ranged constraint that pycutest has split into
                        two. When 'R' appears, it represents both constraints from the
                        range.
    """

    inequality_types: tuple[str, ...]


class PycutestCompatWrapper(AbstractConstrainedMinimisation):
    """Compatibility wrapper that transforms constraints from Optimistix convention
    (where all inequality constraints satisfy g(x) >= 0) to SIF/pycutest conventions.

    This wrapper handles the following transformations:
    - 'L' type constraints: Negates to convert g(x) >= 0 to -g(x) <= 0 (i.e., c(x) <= b)
    - 'G' type constraints: Keeps as-is since g(x) >= 0 already matches c(x) >= b
    - 'E' type constraints: Passes through unchanged
    - 'R' type constraints: Handles ranged constraints that pycutest splits into pairs

    The problem must have a `constraint_conventions` attribute that specifies the
    type of each constraint. This wrapper is primarily used for testing against pycutest
    and as a convenience for users familiar with SIF conventions.
    """

    problem: AbstractConstrainedMinimisation

    @property
    def y0_iD(self) -> int:
        return self.problem.y0_iD

    @property
    def provided_y0s(self) -> frozenset:
        return self.problem.provided_y0s

    @property
    def name(self) -> str:
        return self.problem.name

    def objective(self, y: PyTree[ArrayLike], args: PyTree[Any]) -> Scalar:
        return self.problem.objective(y, args)

    def y0(self) -> PyTree[ArrayLike]:
        return self.problem.y0()

    def args(self) -> PyTree[Any]:
        return self.problem.args()

    def expected_result(self) -> PyTree[ArrayLike]:
        return self.problem.expected_result()

    def expected_objective_value(self) -> Scalar | None:
        return self.problem.expected_objective_value()

    def bounds(self) -> PyTree[ArrayLike] | None:
        return self.problem.bounds()

    def constraint(self, y: PyTree[ArrayLike]) -> tuple[Array, Array]:
        # TODO return ConstraintOut
        raise NotImplementedError

    def num_constraints(self) -> tuple[Int, Int, Int]:
        # No adjustment needed - pycutest already counts split ranged constraints
        return self.problem.num_constraints()
