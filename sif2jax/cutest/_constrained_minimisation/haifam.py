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

            # Compute all element values E(i) = 0.5 * X(idx1) * X(idx2)
            element_x_indices = data["element_x_indices"]
            element_y_indices = data["element_y_indices"]

            # Vectorized computation of all elements
            element_values = []
            for i in range(len(element_x_indices)):
                x_idx = element_x_indices[i]
                y_idx = element_y_indices[i]
                element_val = 0.5 * x[x_idx] * x[y_idx]
                element_values.append(element_val)

            element_values = jnp.array(element_values)

            # Compute constraints: C(j) = sum(coeff * E(i)) - 100*z - x(92)
            constraint_element_lists = data["constraint_element_lists"]
            constraint_coeff_lists = data["constraint_coeff_lists"]

            constraints = []
            for j in range(150):
                element_indices = constraint_element_lists[j]
                coefficients = constraint_coeff_lists[j]

                # Sum weighted elements for this constraint
                constraint_sum = 0.0
                for elem_idx, coeff in zip(element_indices, coefficients):
                    # Element indices in SIF are 1-based, our array is 0-based
                    constraint_sum += coeff * element_values[elem_idx - 1]

                # Apply the constraint formula: C(j) - 100*z - x(92) <= 0
                constraint_value = constraint_sum - 100.0 * z - x92
                constraints.append(constraint_value)

            return None, jnp.array(constraints)

        except (FileNotFoundError, json.JSONDecodeError):
            # If data file not available, raise error
            raise NotImplementedError(
                "HAIFAM requires haifam_data.json with exact SIF element mappings. "
                "Run parse_haifam.py to generate this file from HAIFAM.SIF."
            )
