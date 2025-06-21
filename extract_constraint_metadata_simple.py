"""Simple extraction of constraint metadata from SIF files."""

import json
import re
from pathlib import Path


def extract_constraint_info(problem_name: str) -> dict | None:
    """Extract constraint information for a problem from its SIF file.

    Returns dict with constraint information or None if SIF file not found.
    """
    sif_path = Path(f"/workspaces/sif2jax/archive/mastsif/{problem_name.upper()}.SIF")
    if not sif_path.exists():
        return None

    with open(sif_path) as f:
        content = f.read()

    # Check if RANGES section exists
    has_ranges = "RANGES" in content

    # Extract constraint lines from GROUPS section
    lines = content.split("\n")
    in_groups = False
    constraints = []

    for line in lines:
        line = line.strip()

        if line == "GROUPS":
            in_groups = True
            continue
        elif line in [
            "BOUNDS",
            "START POINT",
            "ELEMENT TYPE",
            "VARIABLES",
            "OBJECT BOUND",
            "GROUP USES",
            "CONSTANTS",
            "RANGES",
        ]:
            in_groups = False
            continue

        if in_groups and line:
            # Skip DO/ND loop markers and other metadata lines
            if any(line.startswith(x) for x in ["DO ", "ND", "IM ", "IA ", "*"]):
                continue

            # Match constraint definitions like "G C1" or "XG A(K)"
            match = re.match(r"^(X?[ELG])\s+(\S+)", line)
            if match:
                ctype = match.group(1)
                cname = match.group(2)

                # Normalize type (remove X prefix if present)
                if ctype.startswith("X"):
                    ctype = ctype[1]

                # Skip objective function and already seen constraints
                if cname not in ["OBJ", "OBJECTIVE"] and not any(
                    c[0] == cname for c in constraints
                ):
                    constraints.append((cname, ctype))

    # For problems with ranges, we need to check which constraints have them
    ranged_constraints = set()
    if has_ranges:
        in_ranges = False
        for line in lines:
            line = line.strip()

            if line == "RANGES":
                in_ranges = True
                continue
            elif line in [
                "BOUNDS",
                "START POINT",
                "ELEMENT TYPE",
                "VARIABLES",
                "OBJECT BOUND",
                "GROUP USES",
                "CONSTANTS",
            ]:
                in_ranges = False
                continue

            if in_ranges and line and not line.startswith("*"):
                # Skip DO/ND loop markers
                if any(line.strip().startswith(x) for x in ["DO ", "ND"]):
                    continue

                # Extract constraint name from ranges section
                parts = line.split()
                if len(parts) >= 3:  # Format: [X] problem_name constraint_name value
                    # Skip the optional X prefix
                    if parts[0] == "X":
                        cname = parts[2] if len(parts) >= 3 else parts[1]
                    else:
                        cname = parts[1]
                    # Remove parameter notation for matching
                    base_name = cname.split("(")[0] if "(" in cname else cname
                    ranged_constraints.add(base_name)

    return {
        "constraints": constraints,
        "ranged_constraints": list(ranged_constraints),
        "has_ranges": has_ranges,
    }


def main():
    """Extract constraint metadata for all HS problems."""
    # Get all HS problems
    hs_problems = []
    const_dir = Path("/workspaces/sif2jax/sif2jax/cutest/_constrained_minimisation")
    for py_file in sorted(const_dir.glob("hs*.py")):
        problem_name = py_file.stem.upper()
        hs_problems.append(problem_name)

    # Process each problem
    results = {}

    for problem in hs_problems:
        info = extract_constraint_info(problem)
        if info:
            # Count constraint types
            eq_count = sum(1 for _, ctype in info["constraints"] if ctype == "E")
            ineq_types = []

            for cname, ctype in info["constraints"]:
                if ctype != "E":
                    # Check if this constraint has a range
                    base_name = cname.split("(")[0] if "(" in cname else cname
                    # Check both the full name and base name for ranges
                    if (
                        cname in info["ranged_constraints"]
                        or base_name in info["ranged_constraints"]
                    ):
                        ineq_types.append("R")
                    else:
                        ineq_types.append(ctype)

            results[problem] = {
                "equality_count": eq_count,
                "inequality_types": ineq_types,
                "raw_constraints": info["constraints"],  # For debugging
            }

            ineq_str = ", ".join(ineq_types) if ineq_types else "none"
            print(
                f"{problem}: {eq_count} equality, "
                f"{len(ineq_types)} inequality ({ineq_str})"
            )
        else:
            print(f"{problem}: No SIF file found")

    # Save results
    with open("hs_constraint_metadata_simple.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nProcessed {len(results)} problems")

    # Show some examples
    print("\nExamples:")
    for problem in ["HS10", "HS93", "HS83", "HS116", "HS118"]:
        if problem in results:
            r = results[problem]
            print(
                f"{problem}: equality={r['equality_count']}, "
                f"inequality={r['inequality_types']}"
            )


if __name__ == "__main__":
    main()
