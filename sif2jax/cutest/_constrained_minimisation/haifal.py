import json
import os

import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


# Load data at module level to avoid tracer leaks
def _load_haifal_data():
    """Load and preprocess HAIFAL data."""
    data_path = os.path.join(os.path.dirname(__file__), "data", "haifal_data.json")

    with open(data_path) as f:
        data = json.load(f)

    # Load the constraint mapping
    mapping_path = os.path.join(
        os.path.dirname(__file__), "data", "haifal_constraint_mapping.json"
    )

    with open(mapping_path) as f:
        mapping_data = json.load(f)

    # Convert to JAX arrays
    element_x_indices = jnp.array(data["element_x_indices"], dtype=jnp.int32)
    element_y_indices = jnp.array(data["element_y_indices"], dtype=jnp.int32)
    constraint_id_order = jnp.array(
        mapping_data["constraint_id_order"], dtype=jnp.int32
    )

    # Build vectorized sparse representation
    # Flatten all element indices and coefficients
    element_indices_flat = []
    coefficients_flat = []
    constraint_indices_flat = []

    constraint_element_lists = data["constraint_element_lists"]
    constraint_coeff_lists = data["constraint_coeff_lists"]

    for constraint_idx, (elements, coeffs) in enumerate(
        zip(constraint_element_lists, constraint_coeff_lists)
    ):
        if elements:  # Only process non-empty constraints
            constraint_id = (
                constraint_id_order[constraint_idx] - 1
            )  # Convert to 0-based
            for elem_idx, coeff in zip(elements, coeffs):
                element_indices_flat.append(elem_idx - 1)  # Convert to 0-based
                coefficients_flat.append(coeff)
                constraint_indices_flat.append(constraint_id)

    # Convert to JAX arrays
    element_indices_flat = jnp.array(element_indices_flat, dtype=jnp.int32)
    coefficients_flat = jnp.array(coefficients_flat)
    constraint_indices_flat = jnp.array(constraint_indices_flat, dtype=jnp.int32)

    # Also return the constraint_id_order for extracting constraints in correct order
    return (
        element_x_indices,
        element_y_indices,
        element_indices_flat,
        coefficients_flat,
        constraint_indices_flat,
        constraint_id_order - 1,  # Convert to 0-based for indexing
    )


# Load data once at module level
_HAIFAL_DATA = _load_haifal_data()


class HAIFAL(AbstractConstrainedMinimisation):
    """Truss Topology Design problem HAIFAL (171-8940).

    A truss topology design optimization problem with 343 variables and 8940
    constraints. This is a quadratic minimization problem with inequality
    constraints derived from structural engineering applications.

    Variables:
    - z: objective variable to be minimized
    - x(1), ..., x(342): design variables

    Objective: minimize z

    Constraints: 8940 inequality constraints of the form C(i) - z - x(333) ≤ 0,
    where each C(i) contains quadratic terms in the design variables.

    Source: M. Tsibulevsky, Optimization Laboratory,
    Faculty of Industrial Engineering, Technion,
    Haifa, 32000, Israel.

    SIF input: Conn, Gould and Toint, May, 1992

    Classification: LQR2-AN-343-8940
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        # Minimize z (the first variable)
        return y[0]

    @property
    def y0(self):
        # Starting point: all variables initialized to 0 (matching pycutest)
        return jnp.zeros(343)

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        # Solution not provided in SIF file
        return None

    @property
    def expected_objective_value(self):
        # Lower bound is 0.0 according to SIF file
        return None

    @property
    def bounds(self):
        # All variables are real with no explicit bounds
        return None

    def constraint(self, y):
        z = y[0]
        x = y[1:343]  # x[0] corresponds to X(1) in SIF, etc.
        x333 = x[332]  # X(333) - appears in all constraints

        # Use preloaded vectorized data
        (
            element_x_indices,
            element_y_indices,
            element_indices_flat,
            coefficients_flat,
            constraint_indices_flat,
            constraint_id_order,
        ) = _HAIFAL_DATA

        # Vectorized element computation: E(i) = 0.5 * X(idx1) * X(idx2)
        # All 74,423 elements computed at once
        element_values = 0.5 * x[element_x_indices] * x[element_y_indices]

        # Initialize all constraints with base formula: -z - x333
        # We need 8959 positions (0-8958) to handle constraint IDs from 0 to 8958
        constraints = jnp.full(8959, -z - x333)

        # Fully vectorized constraint computation using scatter-add
        # Select relevant element values and multiply by coefficients
        relevant_elements = element_values[element_indices_flat]
        weighted_contributions = coefficients_flat * relevant_elements

        # Use scatter-add to accumulate contributions to each constraint
        constraints = constraints.at[constraint_indices_flat].add(
            weighted_contributions
        )

        # Return constraints starting from index 1
        # Constraint IDs start at 1 in pycutest
        # But we computed them with 0-based indexing
        # The mapping starts at C2, so constraints[1] is C2
        # pycutest expects all 8958 constraints starting from C1
        # C1 stays at its initialized value of -z - x333
        return None, constraints[1:]  # Return indices 1-8958
