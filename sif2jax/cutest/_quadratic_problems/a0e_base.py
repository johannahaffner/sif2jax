"""Base class for A0E series quadratic programming problems.

A0E series problems are quadratic programming reformulations of linear
complementarity problems provided by Michael Ferris. All problems follow
the same structure with different sizes.

Common structure:
- 3N variables total (N = 15002 for A0E series)
- N constraints
- Variables split into 3 groups:
  - X1 to XN: Free variables
  - X(N+1) to X2N: Variables with bounds [0, large_value]
  - X(2N+1) to X3N: Variables with quadratic terms

Quadratic pattern:
- X(N+1+i) * X(i) with coefficient 1.0
- X(2N+1+i) * X(i) with coefficient -1.0

Classification: QLR2-AN-45006-15002
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse

from ..._problem import AbstractConstrainedQuadraticProblem


class A0EBase(AbstractConstrainedQuadraticProblem, ABC):
    """Base class for A0E series problems.

    Provides common functionality for all A0E problems which follow the same
    mathematical structure but with different constraint coefficients.
    """

    @property
    @abstractmethod
    def problem_name(self) -> str:
        """Return the specific problem name (e.g., 'A0ENDNDL')."""
        pass

    @property
    def n(self):
        """Number of variables: 45,006 for all A0E problems."""
        return 45006

    @property
    def constraint_size(self):
        """Number of constraints: 15,002 for all A0E problems."""
        return 15002

    @property
    def y0(self):
        """Initial guess - all zeros from SIF."""
        return jnp.zeros(self.n, dtype=jnp.float64)

    @property
    def args(self):
        return None

    def _load_problem_data(self):
        """Load problem-specific data from NPZ file."""
        data_file = (
            Path(__file__).parent / "data" / f"{self.problem_name.lower()}_full.npz"
        )
        return np.load(data_file)

    def objective(self, y, args):
        """Compute quadratic objective function.

        The quadratic form follows the pattern:
        - X(N+1+i) * X(i) with coefficient 1.0
        - X(2N+1+i) * X(i) with coefficient -1.0

        Stored in upper triangular format requiring symmetry doubling.
        """
        del args

        # Load quadratic terms (cached at class level)
        if not hasattr(self, "_quad_data"):
            problem_data = self._load_problem_data()
            self._quad_i = jnp.array(problem_data["quad_i"], dtype=jnp.int32)
            self._quad_j = jnp.array(problem_data["quad_j"], dtype=jnp.int32)
            self._quad_vals = jnp.array(problem_data["quad_vals"], dtype=jnp.float64)
            self._quad_data = True

        # Upper triangular representation - double for symmetry
        obj = jnp.sum(2.0 * y[self._quad_i] * y[self._quad_j] * self._quad_vals)
        return 0.5 * obj

    def constraint(self, y):
        """Compute linear constraint values."""
        # Load constraint data (cached at class level)
        if not hasattr(self, "_constraint_data"):
            problem_data = self._load_problem_data()
            n_con = int(problem_data["n_con"])
            n_var = int(problem_data["n_var"])
            indices = jnp.array(
                np.stack([problem_data["con_rows"], problem_data["con_cols"]], axis=1),
                dtype=jnp.int32,
            )
            values = jnp.array(problem_data["con_vals"], dtype=jnp.float64)
            self._A_sparse = sparse.BCOO((values, indices), shape=(n_con, n_var))
            self._rhs = jnp.array(problem_data["rhs"], dtype=jnp.float64)
            self._constraint_data = True

        eq_constraint = self._A_sparse @ y - self._rhs
        ineq_constraint = None
        return eq_constraint, ineq_constraint

    @property
    def bounds(self):
        """Return the variable bounds."""
        # Load bounds data (cached at class level)
        if not hasattr(self, "_bounds_data"):
            problem_data = self._load_problem_data()
            # Convert large finite values to infinity during loading
            lower_raw = problem_data["lower"]
            upper_raw = problem_data["upper"]
            self._lower = jnp.where(lower_raw == -1e21, -jnp.inf, lower_raw)
            self._upper = jnp.where(upper_raw == 1e21, jnp.inf, upper_raw)
            self._bounds_data = True

        return self._lower, self._upper

    @property
    def expected_result(self):
        """Expected result not provided in SIF files."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value not provided in SIF files."""
        return None
