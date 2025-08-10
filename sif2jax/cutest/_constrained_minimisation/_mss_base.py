"""Base class for MSS (Maximum Stable Set) problems."""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..._problem import AbstractConstrainedMinimisation


class MSSBase(AbstractConstrainedMinimisation):
    """Base class for rank-two relaxation of maximum stable set problems.

    These are quadratic optimization problems arising from graph theory,
    specifically from the maximum stable set problem.

    The general formulation is:
    - Variables: x_i, y_i for i=1,...,n_vertices
    - Objective: maximize sum(x_i) + sum(y_i), reformulated as -sum(x_i)^2 - sum(y_i)^2
    - Constraints:
      - Spherical: sum(x_i^2 + y_i^2) = 1
      - Edge: for each edge (i,j), x_i*x_j + y_i*y_j = 0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})
    n_vertices: int = eqx.field(static=True, default=0)  # Will be set by subclass
    n_edges: int = eqx.field(static=True, default=0)  # Will be set by subclass
    edges: tuple = eqx.field(static=True, default=())  # Will be set by subclass

    def __init__(self):
        self.n_vertices = self._get_n_vertices()
        self.n_edges = self._get_n_edges()
        self.edges = self._get_edges()

    def _get_n_vertices(self) -> int:
        """Return the number of vertices in the graph."""
        raise NotImplementedError

    def _get_n_edges(self) -> int:
        """Return the number of edges in the graph."""
        raise NotImplementedError

    def _get_edges(self) -> tuple:
        """Return the edges of the graph as a tuple of vertex pairs."""
        raise NotImplementedError

    @property
    def n(self) -> int:
        """Number of variables (2 * number of vertices)."""
        return 2 * self.n_vertices

    @property
    def y0(self) -> Array:
        """Initial guess - all ones as per SIF file."""
        return jnp.ones(self.n)

    @property
    def args(self):
        """Additional arguments."""
        return None

    def objective(self, y: Array, args) -> Float[Array, ""]:
        """Compute the objective function -sum(x_i)^2 - sum(y_i)^2."""
        del args
        x = y[: self.n_vertices]
        y_vars = y[self.n_vertices :]

        # The objective is to maximize sum(x_i) + sum(y_i)
        # But with the group type -L2, it becomes -sum(x_i)^2 - sum(y_i)^2
        sum_x = jnp.sum(x)
        sum_y = jnp.sum(y_vars)
        return -(sum_x**2) - (sum_y**2)

    def constraint(self, y: Array):
        """Compute the constraints."""
        x = y[: self.n_vertices]
        y_vars = y[self.n_vertices :]

        # Spherical constraint: sum(x_i^2 + y_i^2) = 1
        spherical = jnp.sum(x**2 + y_vars**2) - 1.0

        # Edge constraints: for each edge (i,j), x_i*x_j + y_i*y_j = 0
        # Vectorized computation
        edges_array = jnp.array(self.edges)
        i_indices = edges_array[:, 0]
        j_indices = edges_array[:, 1]
        edge_constraints = (
            x[i_indices] * x[j_indices] + y_vars[i_indices] * y_vars[j_indices]
        )

        # All constraints are equality constraints
        equality_constraints = jnp.concatenate(
            [jnp.array([spherical]), edge_constraints]
        )

        return equality_constraints, None

    @property
    def bounds(self) -> tuple[Array, Array] | None:
        """No bounds for this problem."""
        return None

    @property
    def expected_result(self):
        """Expected result of the optimization problem."""
        # The SIF files don't provide exact solutions
        return None

    @property
    def expected_objective_value(self):
        """Expected value of the objective at the solution."""
        # The SIF files don't provide exact objective values
        return None
