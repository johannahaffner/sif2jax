#!/usr/bin/env python3
"""Extract DUAL problem data from SIF files and generate JAX code."""

import re
import sys
from pathlib import Path


def extract_dual_data(filepath):
    """Extract linear coefficients and quadratic matrix from DUAL SIF file."""

    with open(filepath) as f:
        lines = f.readlines()

    # Extract problem name
    problem_name = None
    for line in lines:
        if line.startswith("NAME"):
            problem_name = line.split()[1]
            break

    # Extract dimensions from classification
    n_vars = None
    n_constraints = None
    for line in lines:
        if "classification" in line:
            # Parse QLR2-MN-XX-YY format
            match = re.search(r"QLR2-MN-(\d+)-(\d+)", line)
            if match:
                n_vars = int(match.group(1))
                n_constraints = int(match.group(2))
            break

    # Extract linear coefficients
    linear_coeffs = {}
    for line in lines:
        match = re.match(r"\s+x(\d+)\s+obj\s+([\d.]+)", line)
        if match:
            idx = int(match.group(1)) - 1  # Convert to 0-based
            val = float(match.group(2))
            linear_coeffs[idx] = val

    # Build full coefficient array
    c_array = []
    if n_vars:
        for i in range(n_vars):
            c_array.append(linear_coeffs.get(i, 0.0))
    else:
        # Default to empty if dimensions not found
        c_array = []

    # Extract quadratic matrix entries
    matrix_entries = []
    in_group_uses = False
    for line in lines:
        if "GROUP USES" in line:
            in_group_uses = True
            continue
        if in_group_uses:
            match = re.match(r"\s*E\s+obj\s+x(\d+),(\d+)\s+([-\d.]+)", line)
            if match:
                i = int(match.group(1)) - 1  # Convert to 0-based
                j = int(match.group(2)) - 1  # Convert to 0-based
                val = float(match.group(3))
                matrix_entries.append((i, j, val))
            elif line.strip() and not line.startswith(" "):
                break

    return {
        "name": problem_name,
        "n": n_vars,
        "m": n_constraints,
        "c": c_array,
        "Q_entries": matrix_entries,
    }


def generate_dual_implementation(data):
    """Generate JAX implementation for DUAL problem."""

    # Build sparse matrix representation
    rows = []
    cols = []
    vals = []

    for i, j, v in data["Q_entries"]:
        rows.append(i)
        cols.append(j)
        vals.append(v)
        # Add symmetric entry if off-diagonal
        if i != j:
            rows.append(j)
            cols.append(i)
            vals.append(v)

    implementation = f'''import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractConstrainedQuadraticProblem


class {data["name"]}(AbstractConstrainedQuadraticProblem):
    """A dual quadratic program from Antonio Frangioni.
    
    This is the dual of PRIMAL{data["name"][4:]}.SIF
    
    References:
    - Problem provided by Antonio Frangioni (frangio@DI.UniPi.IT)
    - SIF input: Irv Lustig and Nick Gould, June 1996
    
    Classification: QLR2-MN-{data["n"]}-{data["m"]}
    - QLR2: Quadratic objective, linear constraints
    - MN: General constraints
    - {data["n"]} variables, {data["m"]} constraint(s)
    """
    
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({{0}})
    
    @property
    def n(self):
        """Number of variables."""
        return {data["n"]}
    
    @property
    def m(self):
        """Number of constraints."""
        return {data["m"]}
    
    # Linear objective coefficients
    c = jnp.array({data["c"]})
    
    # Quadratic matrix in COO format (row, col, value)
    # Total non-zero entries: {len(vals)}
    Q_row = jnp.array({rows}, dtype=jnp.int32)
    Q_col = jnp.array({cols}, dtype=jnp.int32)
    Q_val = jnp.array({vals})
    
    def objective(self, y, args):
        """Quadratic objective: 0.5 * y^T * Q * y + c^T * y"""
        del args
        
        # Linear term
        linear_term = jnp.dot(self.c, y)
        
        # Quadratic term using sparse representation
        quad_term = jnp.sum(self.Q_val * y[self.Q_row] * y[self.Q_col])
        
        return 0.5 * quad_term + linear_term
    
    def constraint(self, y):
        """Linear equality constraint: sum(y) = 1"""
        constraints = jnp.array([jnp.sum(y) - 1.0])
        return constraints, None
    
    def equality_constraints(self):
        """All constraints are equalities."""
        return jnp.ones(self.m, dtype=bool)
    
    @property
    def bounds(self):
        """Bound constraints: 0 <= y <= 1"""
        lbs = jnp.zeros(self.n)
        ubs = jnp.ones(self.n)
        return lbs, ubs
    
    @property
    def y0(self):
        """Initial point: uniform distribution"""
        return inexact_asarray(jnp.ones(self.n) / self.n)
    
    @property
    def args(self):
        return None
    
    @property
    def expected_result(self):
        """Expected optimal solution (not known analytically)."""
        return None
    
    @property
    def expected_objective_value(self):
        """Expected optimal objective value (not known analytically)."""
        return None
'''

    return implementation


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_dual_data.py <SIF_FILE>")
        sys.exit(1)

    filepath = Path(sys.argv[1])
    data = extract_dual_data(filepath)

    print(f"Problem: {data['name']}")
    print(f"Variables: {data['n']}")
    print(f"Constraints: {data['m']}")
    print(f"Quadratic entries: {len(data['Q_entries'])}")

    # Generate implementation
    impl = generate_dual_implementation(data)

    # Write to file
    output_file = f"{data['name'].lower()}.py"
    with open(output_file, "w") as f:
        f.write(impl)

    print(f"\nImplementation written to {output_file}")
