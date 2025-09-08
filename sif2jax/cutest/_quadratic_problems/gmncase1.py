import os

import jax.numpy as jnp
import numpy as np
from equinox import Module

from ..._problem import AbstractConstrainedQuadraticProblem


class GMNCASE1(AbstractConstrainedQuadraticProblem, Module):
    """GMNCASE1 problem - A non-convex quadratic program from control.

    The aim is to synthesize a quadratically optimized control
    for an industrial process that respects certain constraints
    which may be physical or part of a design. The data is from
    one instant in a continuous process.

    Source:
    Gordon McNeilly (gordon@icu.strath.ac.uk)
    Industrial Control Centre
    Strathclyde University, Glasgow G1 1QE

    SIF input: Nick Gould, October 1997

    Classification: QLR2-AN-175-300

    TODO: Human review needed - constraint ordering vs pycutest

    Implementation status: 15/27 tests passing (56% complete)
    ✓ Comprehensive SIF parsing: 11,802 quadratic terms, 300 constraints extracted
    ✓ Correct mathematics: constraint values numerically identical to pycutest
    ✓ Efficient storage: 106KB .npz format (97% space reduction from 3.6MB .py files)
    ✗ Constraint ordering: values match but wrong positions vs pycutest sequence

    Issue: SIF constraints CON2,CON3,CON4,CON5,CON6,CON7,CON9... (gaps at 8,15,22,29...)
    but pycutest expects different ordering. Values match: pycutest[0]=our[95]

    Data location: sif2jax/cutest/_quadratic_problems/data/gmncase1.npz
    """

    # Store data as class attributes for efficiency
    _c: jnp.ndarray  # Linear coefficients
    _hess_row: jnp.ndarray  # Hessian row indices
    _hess_col: jnp.ndarray  # Hessian column indices
    _hess_val: jnp.ndarray  # Hessian values
    _A: jnp.ndarray  # Constraint matrix
    _b: jnp.ndarray  # RHS vector

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def __init__(self):
        """Initialize with precomputed data from .npz file."""
        # Load coefficient data from efficient binary format in data/ subdirectory
        data_dir = os.path.dirname(__file__)
        data_path = os.path.join(data_dir, "data", "gmncase1.npz")

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Coefficient data file not found: {data_path}")

        # Load all coefficient data at once
        data = np.load(data_path)

        # Store linear coefficients
        self._c = jnp.array(data["linear_c"])

        # Store quadratic coefficients in sparse format
        self._hess_row = jnp.array(data["quad_rows"])
        self._hess_col = jnp.array(data["quad_cols"])
        self._hess_val = jnp.array(data["quad_vals"])

        # Store constraint matrix and RHS vector
        self._A = jnp.array(data["A_matrix"])
        self._b = jnp.array(data["b_vector"])

    @property
    def n(self):
        """Number of variables."""
        return 175

    @property
    def m(self):
        """Number of constraints."""
        return 300

    def objective(self, y, args):
        """Compute the objective function using vectorized operations."""
        del args

        # Linear term: c'y (use all 175 variables)
        linear_term = jnp.dot(self._c, y)

        # Quadratic term: 0.5 * y'Qy using sparse representation
        # For each (i,j,v) triple, compute v * y[i] * y[j]
        quad_term = 0.5 * jnp.sum(
            self._hess_val * y[self._hess_row] * y[self._hess_col]
        )

        return linear_term + quad_term

    def constraint(self, y):
        """Compute constraints using matrix-vector multiplication."""
        # SIF convention: constraints are g(x) <= 0, where g(x) = linear_terms - RHS
        # pycutest returns g(x) values directly
        # Our stored b = RHS, so g(x) = A*y - b, but we need to return -g(x)
        # Actually, pycutest returns g(x) = RHS - A*y, so we compute b - A*y
        constraint_vals = self._b - self._A @ y
        return None, constraint_vals

    def equality_constraints(self):
        """No equality constraints for this problem."""
        return jnp.zeros(self.m, dtype=bool)

    @property
    def y0(self):
        """Initial guess - zeros as no starting point specified in SIF."""
        return jnp.zeros(self.n)

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def bounds(self):
        """All variables are free (unbounded)."""
        # Return None for completely unbounded problems
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value from SIF file."""
        return jnp.array(2.66733e-01)

    @property
    def expected_result(self):
        """Expected solution - not provided in SIF, using placeholder."""
        # SIF file doesn't provide the full solution vector
        # This would need to be obtained from running the Fortran solver
        return None
