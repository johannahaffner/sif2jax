import abc
from typing import Any

import equinox as eqx
import jax.flatten_util as jfu
import jax.tree_util as jtu
from jax import numpy as jnp
from jaxtyping import ArrayLike, Int, PyTree, Scalar


_Out = Scalar | PyTree[ArrayLike] | None
_ConstraintOut = (
    tuple[None, PyTree[ArrayLike]]
    | tuple[PyTree[ArrayLike], None]
    | tuple[PyTree[ArrayLike], PyTree[ArrayLike]]
)


class AbstractProblem(eqx.Module):
    """Abstract base class for benchmark problems."""

    y0_iD: eqx.AbstractVar[int]
    provided_y0s: eqx.AbstractVar[frozenset]

    def __check_init__(self):
        if self.y0_iD not in self.provided_y0s:
            raise ValueError(
                f"y0_iD {self.y0_iD} is not one of the accepted values for problem "
                f"{self.name}. Accepted values are {sorted(self.provided_y0s)}."
            )

    @property
    def name(self):
        """Returns the name of the benchmark problem, which should be the same as the
        name of the class that implements it. For CUTEST problems, this is the name of
        the problem used in the SIF file: e.g. "BT1" or "AIRCRAFTB".
        """
        return self.__class__.__name__

    @abc.abstractmethod
    def objective(self, y, args) -> _Out:
        """Objective function to be minimized. Can return a single scalar value (for a
        minimisation problem) or a PyTree of arrays (for a least-squares problem).
        """

    @abc.abstractmethod
    def y0(self) -> PyTree[ArrayLike]:
        """Initial guess for the optimization problem.

        If the problem provides multiple initial values (indicated by provided_y0s
        having more than one element), this method should return the initial value
        corresponding to the current y0_iD.
        """

    @abc.abstractmethod
    def args(self) -> PyTree[Any]:
        """Additional arguments for the objective function."""

    @abc.abstractmethod
    def expected_result(self) -> PyTree[ArrayLike]:
        """Expected result of the optimization problem. Should be a PyTree of arrays
        with the same structure as `y0`."""

    @abc.abstractmethod
    def expected_objective_value(self) -> _Out:
        """Expected value of the objective function at the optimal solution. For a
        minimisation function, this is a scalar value.
        For a least-squares problem, this is a PyTree of residuals.
        """

    def num_variables(self) -> int:
        """Returns the number of variables in the problem. This is the total number of
        elements in the PyTree returned by `y0`.
        """
        flattened_y0, _ = jfu.ravel_pytree(self.y0())
        return flattened_y0.size

    @abc.abstractmethod
    def num_constraints(self) -> tuple[Int, Int, Int]:
        """Returns the number of constraints in the problem. The first element is the
        number of equality constraints, the second is the number of inequality
        constraints, and the third is the number of bound constraints.
        """


class AbstractUnconstrainedMinimisation(AbstractProblem):
    """Abstract base class for unconstrained minimisation problems. The objective
    function for these problems returns a single scalar value, and they have neither
    bounds on the variable `y` nor any other constraints.
    """

    @abc.abstractmethod
    def objective(self, y, args) -> Scalar:
        """Objective function to be minimized. Must return a scalar value."""

    @abc.abstractmethod
    def expected_objective_value(self) -> Scalar | None:
        """Expected value of the objective function at the optimal solution. For a
        minimisation function, this is a scalar value.
        """

    def num_constraints(self) -> tuple[Int, Int, Int]:
        return 0, 0, 0


class AbstractBoundedMinimisation(AbstractProblem):
    """Abstract base class for bounded minimisation problems. The objective
    function for these problems returns a single scalar value, they specify bounds on
    the variable `y` but no other constraints.
    """

    @abc.abstractmethod
    def objective(self, y, args) -> Scalar:
        """Objective function to be minimized. Must return a scalar value."""

    @abc.abstractmethod
    def expected_objective_value(self) -> Scalar | None:
        """Expected value of the objective function at the optimal solution. For a
        minimisation function, this is a scalar value.
        """

    @abc.abstractmethod
    def bounds(self) -> PyTree[ArrayLike]:
        """Returns the bounds on the variable `y`. Should be a tuple (`lower`, `upper`)
        where `lower` and `upper` are PyTrees of arrays with the same structure as `y0`.
        """

    def num_constraints(self) -> tuple[Int, Int, Int]:
        num_bounds = jtu.tree_map(jnp.isfinite, self.bounds())
        num_bounds, _ = jfu.ravel_pytree(num_bounds)
        return 0, 0, jnp.sum(num_bounds)


class AbstractConstrainedMinimisation(AbstractProblem):
    """Abstract base class for constrained minimisation problems. These can have both
    equality or inequality constraints, and they may also have bounds on `y`. We do not
    differentiate between bounded constrained problems and constrained optimisation
    problems without bounds, as we do expect our solvers to do the right thing in each
    of these cases.
    """

    @abc.abstractmethod
    def objective(self, y, args) -> Scalar:
        """Objective function to be minimized. Must return a scalar value."""

    @abc.abstractmethod
    def expected_objective_value(self) -> Scalar | None:
        """Expected value of the objective function at the optimal solution. For a
        minimisation function, this is a scalar value.
        """

    @abc.abstractmethod
    def bounds(self) -> PyTree[ArrayLike] | None:
        """Returns the bounds on the variable `y`, if specified.
        Should be a tuple (`lower`, `upper`) where `lower` and `upper` are PyTrees of
        arrays with the same structure as `y0`.
        """

    @abc.abstractmethod
    def constraint(self, y) -> _ConstraintOut:
        """Returns the constraints on the variable `y`. The constraints can be either
        equality, inequality constraints, or both. This method returns a tuple, with the
        equality constraint in the first argument and the inequality constraint values
        in the second argument. If there are no equality constraints, the first element
        should be `None`. If there are no inequality constraints, the second element
        should be `None`. (None, None) is not allowed as an output - in that case the
        problem has no constraints and should not be classified as a constrained
        minimisation problem.

        All constraints are assumed to be satisfied when the value is
        equal to zero for equality constraints and greater than or equal to zero for
        inequality constraints. Each element of each returned pytree of arrays will be
        treated as the output of a constraint function (in other words: each constraint
        function returns a scalar value, a collection of which may be arranged in a
        pytree.)

        Example:
        ```python
        def constraint(self, y):
            x1, x2, x3 = y
            # Equality constraints
            c1 = x1 * x2 + x3
            # Inequality constraints
            c2 = x1 + x2
            c3 = x3 - x3
            return c1, (c2, c3)
        ```
        """

    def num_constraints(self) -> tuple[Int, Int, Int]:
        equality_out, inequality_out = self.constraint(self.y0())
        if equality_out is None:
            num_equalities = 0
        else:
            equalities, _ = jfu.ravel_pytree(jtu.tree_map(jnp.isfinite, equality_out))
            num_equalities = jnp.sum(equalities)
        if inequality_out is None:
            num_inequalities = 0
        else:
            inequalities, _ = jfu.ravel_pytree(
                jtu.tree_map(jnp.isfinite, inequality_out)
            )
            num_inequalities = jnp.sum(inequalities)
        bounds = self.bounds()
        if bounds is None:
            num_bounds = 0
        else:
            num_bounds, _ = jfu.ravel_pytree(jtu.tree_map(jnp.isfinite, bounds))
            num_bounds = jnp.sum(num_bounds)
        return num_equalities, num_inequalities, num_bounds
