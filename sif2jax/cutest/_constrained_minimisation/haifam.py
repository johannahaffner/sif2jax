import json
import os

import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HAIFAM(AbstractConstrainedMinimisation):
    """Truss Topology Design problem HAIFAM (t49-150).

    A truss topology design optimization problem with 99 variables and 150 constraints.
    This is a quadratic minimization problem with inequality constraints derived
    from structural engineering applications.

    Variables:
    - z: objective variable to be minimized
    - x(1), ..., x(98): design variables

    Objective: minimize z

    Constraints: 150 inequality constraints of the form C(i) - 100*z - x(92) ≤ 0,
    where each C(i) contains quadratic terms in the design variables.

    Source: M. Tsibulevsky, Optimization Laboratory,
    Faculty of Industrial Engineering, Technion,
    Haifa, 32000, Israel.

    SIF input: Conn, Gould and Toint, May, 1992
    minor correction by Ph. Shott, Jan 1995.

    Classification: LQR2-AN-99-150
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        # Minimize 100*z (as specified in SIF file)
        return 100.0 * y[0]

    @property
    def y0(self):
        # Starting point: all variables initialized to 1 (matching pycutest)
        return jnp.ones(99)

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        # Solution not provided in SIF file
        return None

    @property
    def expected_objective_value(self):
        # Lower bound not specified in SIF file
        return None

    @property
    def bounds(self):
        # All variables are real with no explicit bounds
        return None

    def constraint(self, y):
        z = y[0]
        x = y[1:99]  # x[0] corresponds to X(1) in SIF, etc.

        # Extract X(92) - appears in all constraints (0-indexed: x[91])
        x92 = x[91]

        # Exact implementation using parsed SIF element mappings
        # Based on 856 elements and 150 constraints from HAIFAM.SIF

        # Import the exact constraint computation from parsed data
        # This uses the pre-computed element and constraint mappings
        try:
            # Load precomputed HAIFAM data if available
            data_path = os.path.join(
                os.path.dirname(__file__), "data", "haifam_data.json"
            )

            with open(data_path) as f:
                data = json.load(f)

            # Vectorized computation of element values E(i) = 0.5 * X(idx1) * X(idx2)
            element_x_indices = jnp.array(data["element_x_indices"])
            element_y_indices = jnp.array(data["element_y_indices"])

            # Fully vectorized element computation
            element_values = 0.5 * x[element_x_indices] * x[element_y_indices]

            # Vectorized constraint computation using advanced indexing
            constraint_element_lists = data["constraint_element_lists"]
            constraint_coeff_lists = data["constraint_coeff_lists"]

            # Convert to flat arrays for vectorized operations
            max_elements_per_constraint = max(
                len(elem_list) for elem_list in constraint_element_lists
            )

            # Pad element indices and coefficients to uniform length
            padded_element_indices = jnp.full(
                (150, max_elements_per_constraint), -1, dtype=jnp.int32
            )
            padded_coefficients = jnp.zeros(
                (150, max_elements_per_constraint), dtype=y.dtype
            )

            for j in range(150):
                elem_list = constraint_element_lists[j]
                coeff_list = constraint_coeff_lists[j]
                n_elements = len(elem_list)

                # Convert to 0-based indexing (SIF uses 1-based) and ensure dtype
                elem_array = jnp.array(elem_list, dtype=jnp.int32) - 1
                coeff_array = jnp.array(coeff_list, dtype=y.dtype)

                padded_element_indices = padded_element_indices.at[j, :n_elements].set(
                    elem_array
                )
                padded_coefficients = padded_coefficients.at[j, :n_elements].set(
                    coeff_array
                )

            # Create mask for valid indices
            valid_mask = padded_element_indices >= 0

            # Use advanced indexing to gather element values
            # Set invalid indices to 0 to avoid out-of-bounds
            safe_indices = jnp.where(valid_mask, padded_element_indices, 0)
            gathered_elements = element_values[safe_indices]

            # Apply mask and compute weighted sums
            masked_products = jnp.where(
                valid_mask, gathered_elements * padded_coefficients, 0.0
            )
            constraint_sums = jnp.sum(jnp.asarray(masked_products), axis=1)

            # Apply the constraint formula: C(j) - 100*z - x(92) <= 0
            constraints = constraint_sums - 100.0 * z - x92

            return None, constraints

        except (FileNotFoundError, json.JSONDecodeError):
            # If data file not available, raise error
            raise NotImplementedError(
                "HAIFAM requires haifam_data.json with exact SIF element mappings. "
                "Run parse_haifam.py to generate this file from HAIFAM.SIF."
            )
