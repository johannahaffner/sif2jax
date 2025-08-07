#!/usr/bin/env python3
"""Parse DUAL SIF files to extract quadratic matrix coefficients."""

import re
import sys
from pathlib import Path


def parse_dual_sif(filepath):
    """Parse a DUAL SIF file and extract the quadratic matrix."""

    with open(filepath) as f:
        lines = f.readlines()

    # Find the GROUP USES section
    in_group_uses = False
    matrix_entries = []

    for line in lines:
        if "GROUP USES" in line:
            in_group_uses = True
            continue

        if in_group_uses:
            # Look for lines like: E  obj       x1,2                 8
            match = re.match(r"\s*E\s+obj\s+x(\d+),(\d+)\s+([-\d.]+)", line)
            if match:
                i = int(match.group(1)) - 1  # Convert to 0-based
                j = int(match.group(2)) - 1  # Convert to 0-based
                val = float(match.group(3))
                matrix_entries.append((i, j, val))
            elif line.strip() and not line.startswith(" "):
                # End of GROUP USES section
                break

    return matrix_entries


def generate_matrix_code(entries, n):
    """Generate Python code for the matrix."""

    # Group by chunks for readability
    rows = []
    cols = []
    vals = []

    for i, j, v in entries:
        rows.append(i)
        cols.append(j)
        vals.append(v)
        # Add symmetric entry if off-diagonal
        if i != j:
            rows.append(j)
            cols.append(i)
            vals.append(v)

    # Generate code
    code = f"""        # Quadratic matrix entries (row, col, value)
        # Total non-zero entries: {len(rows)}
        rows = {rows}
        cols = {cols}
        vals = {vals}
        
        self.Q_row = jnp.array(rows, dtype=jnp.int32)
        self.Q_col = jnp.array(cols, dtype=jnp.int32)
        self.Q_val = jnp.array(vals)"""

    return code


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python parse_dual_matrix.py <SIF_FILE>")
        sys.exit(1)

    filepath = Path(sys.argv[1])
    entries = parse_dual_sif(filepath)

    print(f"Found {len(entries)} matrix entries")

    # Determine matrix size
    if entries:
        max_idx = max(max(i, j) for i, j, _ in entries)
        n = max_idx + 1
        print(f"Matrix size: {n}x{n}")

        # Generate code
        code = generate_matrix_code(entries, n)
        print("\nGenerated code:")
        print(code)
