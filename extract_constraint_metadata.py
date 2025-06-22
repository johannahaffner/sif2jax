"""Extract constraint metadata from SIF files."""

from pathlib import Path


def parse_sif_file(sif_path: Path) -> dict:
    """Parse a SIF file to extract constraint information.

    Returns:
        Dict with:
        - 'constraints': List of (name, type) tuples where type is 'E', 'L', 'G'
        - 'has_ranges': Boolean indicating if RANGES section exists
    """
    with open(sif_path) as f:
        content = f.read()

    constraints = []
    seen_constraints = set()

    # Check if RANGES section exists
    has_ranges = "RANGES" in content

    # Extract constraints from GROUPS section
    lines = content.split("\n")
    in_groups = False

    for line in lines:
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith("*"):
            continue

        # Check for section headers
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

        # Parse groups section for constraints
        if in_groups:
            parts = line.split()
            if len(parts) >= 2:
                # Skip DO/ND loop markers and other non-constraint lines
                if parts[0] in ["DO", "ND"] or parts[0].startswith("I"):
                    continue

                # Handle expanded constraint types like XG, XE, XL
                if parts[0].startswith("X") and len(parts[0]) == 2:
                    constraint_type = parts[0][1]
                else:
                    constraint_type = parts[0]

                constraint_name = parts[1]

                if constraint_type in ["E", "L", "G"] and constraint_name not in [
                    "OBJ",
                    "OBJECTIVE",
                ]:
                    # For parametric constraints, just note that this type exists
                    # We'll handle the actual expansion based on the problem dimensions
                    if "(" in constraint_name:
                        # This is a parametric constraint like A(K)
                        # We'll note it exists but not expand it here
                        base_name = constraint_name[: constraint_name.index("(")]
                        # Add a marker that this is parametric
                        if f"{base_name}_param" not in seen_constraints:
                            constraints.append((f"{base_name}_param", constraint_type))
                            seen_constraints.add(f"{base_name}_param")
                    else:
                        # Regular constraint
                        if constraint_name not in seen_constraints:
                            constraints.append((constraint_name, constraint_type))
                            seen_constraints.add(constraint_name)

    return {"constraints": constraints, "has_ranges": has_ranges}


def get_constraint_metadata(sif_path: Path) -> tuple[list[str], list[str]]:
    """Extract constraint metadata for a problem.

    Returns:
        Tuple of (equality_types, inequality_types)
        where each list contains 'E', 'L', 'G', or 'R' for each constraint
    """
    info = parse_sif_file(sif_path)

    equality_types = []
    inequality_types = []

    for name, ctype in info["constraints"]:
        if ctype == "E":
            equality_types.append("E")
        else:
            # Check if this constraint has a range
            if name in info["ranges"]:
                inequality_types.append("R")
            else:
                inequality_types.append(ctype)

    return equality_types, inequality_types


def generate_constraint_info(problem_name: str) -> str | None:
    """Generate constraint info code for a problem."""
    # Find the SIF file
    sif_path = Path(f"/workspaces/sif2jax/archive/mastsif/{problem_name.upper()}.SIF")
    if not sif_path.exists():
        return None

    eq_types, ineq_types = get_constraint_metadata(sif_path)

    # Count equality constraints
    n_eq = len(eq_types)

    # Format inequality types
    if ineq_types:
        ineq_str = "(" + ", ".join(f"'{t}'" for t in ineq_types) + ",)"
    else:
        ineq_str = "()"

    return (
        f"constraint_info = ConstraintInfo(inequality_types={ineq_str}, "
        f"equality_count={n_eq})"
    )


# Test with some examples
if __name__ == "__main__":
    import json

    # Get all HS problems from the constrained minimisation directory
    hs_problems = []
    const_dir = Path("/workspaces/sif2jax/sif2jax/cutest/_constrained_minimisation")
    for py_file in sorted(const_dir.glob("hs*.py")):
        problem_name = py_file.stem.upper()
        hs_problems.append(problem_name)

    # Extract metadata for all HS problems
    metadata = {}

    for problem in hs_problems:
        sif_path = Path(f"/workspaces/sif2jax/archive/mastsif/{problem}.SIF")
        if sif_path.exists():
            eq_types, ineq_types = get_constraint_metadata(sif_path)
            metadata[problem] = {
                "equality_count": len(eq_types),
                "inequality_types": ineq_types,
            }
            ineq_str = ", ".join(ineq_types) if ineq_types else "none"
            print(
                f"{problem}: {len(eq_types)} equality, "
                f"{len(ineq_types)} inequality ({ineq_str})"
            )
        else:
            print(f"{problem}: No SIF file found")

    # Save to JSON
    with open("hs_constraint_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nExtracted metadata for {len(metadata)} HS problems")

    # Show a few examples
    print("\nGenerated code examples:")
    for problem in ["HS10", "HS93", "HS83", "HS116", "HS118"]:
        if problem in metadata:
            code = generate_constraint_info(problem)
            print(f"{problem}: {code}")
