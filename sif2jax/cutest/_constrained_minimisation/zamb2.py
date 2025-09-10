"""
ZAMB2 - Zambezi hydropower system optimization (1 year, 1960 start)

Optimal control of Zambezi hydropower system.

Reference: Arnold, Tatjewski, and Wolochowicz. "Two methods for large-scale
nonlinear optimization and their comparison on a case study of hydropower
optimization." JOTA 81(1994)2,221-248.

Classification: OOR2-RN-66-24
"""

import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


# Module-level private variables for ZAMB2 problem data
_phi = 1e-10  # 1e-5 * 1e-5
_psi = 1e-10  # 1e-5 * 100.0 * 1e-5
_phi3 = 2e-9  # 1e-5 * 20 * 1e-5 * 100.0
_ve = 1e-5
_alfa = 1.2285e-5  # 1.2285 * 1e-5

# Problem configuration: 30 years starting from 1931
_ny = 30  # number of years
_start_year = 1931  # starting year

# Initial reservoir volumes (from SIF file)
_v10, _v20, _v30, _v40 = 50.0, 4.0, 2.0, 40.0

# Reference values (from SIF file)
_v1_ref, _v2_ref, _v3_ref, _v4_ref = 70.97, 5.75, 3.0, 61.8
_q1_ref, _f2_ref, _q3_ref, _q4_ref = 3.8, 0.6, 0.544, 4.78

# Monthly evaporation data [12 months, 2 types: E1 and E4]
_evaporation_data = jnp.array(
    [
        [0.1672, 0.1710],  # Oct (month 0 in SIF indexing)
        [0.1158, 0.1242],  # Nov
        [0.0, 0.0],  # Dec
        [0.0, 0.0],  # Jan
        [0.0, 0.0],  # Feb
        [0.0074, 0.0404],  # Mar
        [0.0910, 0.0989],  # Apr
        [0.1113, 0.1119],  # May
        [0.0994, 0.1011],  # Jun
        [0.0865, 0.0866],  # Jul
        [0.1113, 0.1128],  # Aug
        [0.1464, 0.1471],  # Sep
    ]
)

# Inflow data for 1960 (months 360-371 in SIF indexing)
_inflow_data = jnp.array(
    [
        # [IN1, IN2, IN4] for each month (Oct 1960 - Sep 1961)
        [0.8620, 0.1320, 0.0391],  # Oct 1960
        [0.7240, 0.0860, 0.0391],  # Nov 1960
        [2.1330, 0.0540, 1.4933],  # Dec 1960
        [2.3570, 0.0470, 5.3319],  # Jan 1961
        [2.3010, 0.1060, 5.9693],  # Feb 1961
        [3.2110, 0.4820, 2.5110],  # Mar 1961
        [6.5460, 1.3740, 1.9396],  # Apr 1961
        [8.5840, 1.7390, 2.6549],  # May 1961
        [5.2770, 1.1290, 2.5089],  # Jun 1961
        [2.9270, 0.5820, 2.0481],  # Jul 1961
        [1.6640, 0.3070, 1.3877],  # Aug 1961
        [1.0920, 0.1990, 0.9490],  # Sep 1961
    ]
)


class ZAMB2(AbstractConstrainedMinimisation):
    """ZAMB2: Zambezi hydropower system optimization problem.

    TODO: Human review needed
    Attempts made: [1 - Basic 1-year implementation, 2 - Found needs 30-year horizon]
    Suspected issues: [Requires full 30-year historical dataset (1931-1961),
                      3966 variables, 1440 constraints]
    Resources needed: [Complete inflow/evaporation data extraction from SIF,
                      large-scale vectorization]

    Expected: 3966 variables, 1440 constraints
    (30 years: 11*(12*30)+6 vars, 4*(12*30) constraints)
    Currently: 138 variables, 48 constraints
    (1 year implementation for framework validation)
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def _unpack_variables(self, y):
        """Unpack decision variables into reservoir volumes, outflows, spillages."""
        idx = 0

        # Reservoir volumes V1(t), V2(t), V3(t), V4(t) for t=0..12
        v1 = y[idx : idx + 13]
        idx += 13
        v2 = y[idx : idx + 13]
        idx += 13
        v3 = y[idx : idx + 13]
        idx += 13
        v4 = y[idx : idx + 13]
        idx += 13

        # Outflows Q1(t), Q3(t), Q4(t) for t=0..11
        q1 = y[idx : idx + 12]
        idx += 12
        q3 = y[idx : idx + 12]
        idx += 12
        q4 = y[idx : idx + 12]
        idx += 12

        # Spillages F1(t), F2(t), F3(t), F4(t) for t=0..11
        f1 = y[idx : idx + 12]
        idx += 12
        f2 = y[idx : idx + 12]
        idx += 12
        f3 = y[idx : idx + 12]
        idx += 12
        f4 = y[idx : idx + 12]
        idx += 12

        # Delayed F2 values
        f2_delay1 = y[idx]
        f2_delay2 = y[idx + 1]

        return {
            "v1": v1,
            "v2": v2,
            "v3": v3,
            "v4": v4,
            "q1": q1,
            "q3": q3,
            "q4": q4,
            "f1": f1,
            "f2": f2,
            "f3": f3,
            "f4": f4,
            "f2_delay1": f2_delay1,
            "f2_delay2": f2_delay2,
        }

    def _compute_evaporation_terms(self, vars_dict):
        """Compute evaporation losses for each reservoir at each timestep."""
        months = jnp.arange(12) % 12

        # Get evaporation rates for each reservoir
        e1_vals = _evaporation_data[months, 0]  # E1 data
        e4_vals = _evaporation_data[months, 1]  # E4 data

        # Compute evaporation losses: A*V^2 + B*V + C for each reservoir
        evap1 = self._compute_reservoir_evaporation(vars_dict["v1"][:-1], e1_vals, 1)
        evap2 = self._compute_reservoir_evaporation(vars_dict["v2"][:-1], e1_vals, 2)
        evap3 = self._compute_reservoir_evaporation(vars_dict["v3"][:-1], e1_vals, 3)
        evap4 = self._compute_reservoir_evaporation(vars_dict["v4"][:-1], e4_vals, 4)

        return evap1, evap2, evap3, evap4

    def _compute_reservoir_evaporation(self, volumes, evap_rates, reservoir_idx):
        """Compute evaporation A*V^2 + B*V + C for a reservoir."""
        # Coefficients from SIF element FA definitions
        if reservoir_idx == 1:
            a, b, c = -0.0000553, 0.02232, 4.361
        elif reservoir_idx == 2:
            a, b, c = -0.0026236, 0.07353, 0.036
        elif reservoir_idx == 3:
            a, b, c = -0.2653036, 1.42945, 0.214
        elif reservoir_idx == 4:
            a, b, c = -0.0001706, 0.04366, 0.861
        else:
            raise ValueError(f"Invalid reservoir index: {reservoir_idx}")

        return evap_rates * (a * volumes**2 + b * volumes + c)

    def _compute_energy_terms(self, vars_dict):
        """Compute energy production terms for objective function."""
        # Energy coefficients from SIF FHQ elements
        energy_coeffs = {
            1: (-0.000372, 0.22463, 97.559),  # Q1
            3: (-0.399511, 2.16826, 393.914),  # Q3
            4: (-0.005591, 0.88079, 90.576),  # Q4
        }

        energy_total = 0.0

        for reservoir in [1, 3, 4]:
            a, b, c = energy_coeffs[reservoir]
            if reservoir == 1:
                q_vals = vars_dict["q1"]
                v_curr = vars_dict["v1"][:-1]
                v_next = vars_dict["v1"][1:]
            elif reservoir == 3:
                q_vals = vars_dict["q3"]
                v_curr = vars_dict["v3"][:-1]
                v_next = vars_dict["v3"][1:]
            else:  # reservoir == 4
                q_vals = vars_dict["q4"]
                v_curr = vars_dict["v4"][:-1]
                v_next = vars_dict["v4"][1:]

            # Energy production: Q * (A*V^2 + B*V + C + A*V_next^2 + B*V_next + C)
            head_terms = a * v_curr**2 + b * v_curr + c + a * v_next**2 + b * v_next + c
            energy_total += jnp.sum(q_vals * head_terms)

        return -_alfa * energy_total  # Negative since we minimize

    def _compute_final_energy_terms(self, vars_dict):
        """Compute final energy terms based on final reservoir volumes."""
        # Final energy coefficients from SIF FE elements
        final_coeffs = [
            (1.16, 668.74, -161.32),  # V1 final
            (2.03, 1690.16, -156.01),  # V2 final
            (2.03, 1690.16, -156.01),  # V3 final
            (0.82, 332.09, -154.73),  # V4 final
        ]

        final_volumes = [
            vars_dict["v1"][-1],
            vars_dict["v2"][-1],
            vars_dict["v3"][-1],
            vars_dict["v4"][-1],
        ]

        final_energy = 0.0
        for i, (a, b, c) in enumerate(final_coeffs):
            v = final_volumes[i]
            final_energy += a * v**2 + b * v + c

        return -_ve * final_energy  # Negative since we minimize

    def _compute_reference_penalties(self, vars_dict):
        """Compute penalty terms for deviations from reference values."""
        # Outflow and spillage penalties
        outflow_penalty = (
            jnp.sum((vars_dict["q1"] - _q1_ref) ** 2)
            + jnp.sum((vars_dict["f2"] - _f2_ref) ** 2)
            + jnp.sum((vars_dict["q3"] - _q3_ref) ** 2)
            + jnp.sum((vars_dict["q4"] - _q4_ref) ** 2)
        )

        # Final volume penalties
        final_penalty = (
            (vars_dict["v1"][-1] - _v1_ref) ** 2
            + (vars_dict["v2"][-1] - _v2_ref) ** 2
            + (vars_dict["v3"][-1] - _v3_ref) ** 2 * (_phi3 / _phi)
            + (vars_dict["v4"][-1] - _v4_ref) ** 2
        )

        return _psi * outflow_penalty + _phi * final_penalty

    def objective(self, y, args):
        """Compute objective function: energy production + penalty terms."""
        del args
        vars_dict = self._unpack_variables(y)

        energy_terms = self._compute_energy_terms(vars_dict)
        final_energy_terms = self._compute_final_energy_terms(vars_dict)
        penalty_terms = self._compute_reference_penalties(vars_dict)

        return energy_terms + final_energy_terms + penalty_terms

    def constraint(self, y):
        """Compute reservoir balance constraints."""
        vars_dict = self._unpack_variables(y)

        # Get inflow data for each timestep (start from year 1960 = index 30*12 = 360)
        inflow1 = _inflow_data[:, 0]  # IN1
        inflow2 = _inflow_data[:, 1]  # IN2
        inflow3 = jnp.full(12, -0.0388)  # IN3 (constant)
        inflow4 = _inflow_data[:, 2]  # IN4

        # Compute evaporation losses
        evap1, evap2, evap3, evap4 = self._compute_evaporation_terms(vars_dict)

        # Build F2 delayed series: [f2_delay2, f2_delay1, f2[0], f2[1], ...]
        f2_extended = jnp.concatenate(
            [
                jnp.array([vars_dict["f2_delay2"], vars_dict["f2_delay1"]]),
                vars_dict["f2"],
            ]
        )

        # Reservoir balance equations
        constraints = []

        # Reservoir 1: V1(t) + inflow - Q1 - F1 - evap = V1(t+1)
        cons1 = (
            vars_dict["v1"][:-1]
            + inflow1
            - vars_dict["q1"]
            - vars_dict["f1"]
            - evap1
            - vars_dict["v1"][1:]
        )
        constraints.append(cons1)

        # Reservoir 2: V2(t) + inflow - F2 - evap = V2(t+1)
        cons2 = (
            vars_dict["v2"][:-1]
            + inflow2
            - vars_dict["f2"]
            - evap2
            - vars_dict["v2"][1:]
        )
        constraints.append(cons2)

        # Reservoir 3: V3(t) + inflow - F2(t-2) - Q3 - F3 - evap = V3(t+1)
        f2_delayed = f2_extended[:-2]  # F2(t-2) for each timestep t
        cons3 = (
            vars_dict["v3"][:-1]
            + inflow3
            - f2_delayed
            - vars_dict["q3"]
            - vars_dict["f3"]
            - evap3
            - vars_dict["v3"][1:]
        )
        constraints.append(cons3)

        # Reservoir 4: V4(t) + inflow - (Q1+F1) - (Q3+F3) + Q4 + F4 - evap = V4(t+1)
        cons4 = (
            vars_dict["v4"][:-1]
            + inflow4
            - vars_dict["q1"]
            - vars_dict["f1"]
            - vars_dict["q3"]
            - vars_dict["f3"]
            + vars_dict["q4"]
            + vars_dict["f4"]
            - evap4
            - vars_dict["v4"][1:]
        )
        constraints.append(cons4)

        return jnp.concatenate(constraints), None  # Only equality constraints

    @property
    def y0(self):
        """Starting point for ZAMB2."""
        x0 = []

        # Initial reservoir volumes (V1, V2, V3, V4 for t=0..12)
        for _ in range(13):
            x0.append(_v10)  # V1(t) = V10
        for _ in range(13):
            x0.append(_v20)  # V2(t) = V20
        for _ in range(13):
            x0.append(_v30)  # V3(t) = V30
        for _ in range(13):
            x0.append(_v40)  # V4(t) = V40

        # Outflows (Q1, Q3, Q4 for t=0..11)
        for t in range(12):
            x0.append(_inflow_data[t, 0])  # Q1(t) = IN1(t)
        for t in range(12):
            x0.append(_inflow_data[t, 1])  # Q3(t) = IN2(t)
        for t in range(12):
            x0.append(_inflow_data[t, 0])  # Q4(t) = IN1(t)

        # Spillages (F1, F2, F3, F4 for t=0..11)
        for _ in range(12):
            x0.append(0.0)  # F1(t) = 0
        for t in range(12):
            x0.append(_inflow_data[t, 1])  # F2(t) = IN2(t)
        for _ in range(12):
            x0.append(0.0)  # F3(t) = 0
        for _ in range(12):
            x0.append(0.0)  # F4(t) = 0

        # Delayed F2 values
        x0.append(_f2_ref)  # F2-1 = F2REF
        x0.append(_f2_ref)  # F2-2 = F2REF

        return jnp.array(x0)

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        # We don't know the exact solution, return None
        return None

    @property
    def expected_objective_value(self):
        # Expected objective value from SIF file comment
        return jnp.array(-1.11614e-5)  # Scaled by 1e-5

    @property
    def bounds(self):
        """Build variable bounds for ZAMB2."""
        lower_bounds = []
        upper_bounds = []

        # Reservoir volume bounds V1(t), t=0..12
        for t in range(13):
            if t == 0:
                lower_bounds.append(_v10)  # V1(0) = v10 (initial)
                upper_bounds.append(_v10)
            else:
                lower_bounds.append(0.05)  # V1(t), t=1..12
                upper_bounds.append(70.97)

        # Reservoir volume bounds V2(t), t=0..12
        for t in range(13):
            if t == 0:
                lower_bounds.append(_v20)  # V2(0) = v20 (initial)
                upper_bounds.append(_v20)
            else:
                lower_bounds.append(0.7)  # V2(t), t=1..12
                upper_bounds.append(5.75)

        # Reservoir volume bounds V3(t), t=0..12
        for t in range(13):
            if t == 0:
                lower_bounds.append(_v30)  # V3(0) = v30 (initial)
                upper_bounds.append(_v30)
            else:
                lower_bounds.append(0.02)  # V3(t), t=1..12
                upper_bounds.append(3.0)

        # Reservoir volume bounds V4(t), t=0..12
        for t in range(13):
            if t == 0:
                lower_bounds.append(_v40)  # V4(0) = v40 (initial)
                upper_bounds.append(_v40)
            else:
                lower_bounds.append(0.0)  # V4(t), t=1..12
                upper_bounds.append(61.8)

        # Outflow bounds Q1(t), Q3(t), Q4(t), t=0..11
        for _ in range(12):
            lower_bounds.append(2.1)  # Q1(t)
            upper_bounds.append(3.8)
        for _ in range(12):
            lower_bounds.append(0.3)  # Q3(t)
            upper_bounds.append(0.544)
        for _ in range(12):
            lower_bounds.append(3.59)  # Q4(t)
            upper_bounds.append(4.78)

        # Spillage bounds F1(t), F2(t), F3(t), F4(t), t=0..11
        for _ in range(12):
            lower_bounds.append(0.0)  # F1(t)
            upper_bounds.append(8.1)

        # F2(t) bounds with seasonal variations
        for t in range(12):
            if t in [0, 1, 2, 3]:  # Oct-Jan
                lower_bounds.append(0.37)
            elif t == 4:  # Feb
                lower_bounds.append(0.555)
            elif t == 5:  # Mar
                lower_bounds.append(0.8325)
            elif t == 6:  # Apr
                lower_bounds.append(0.555)
            elif t in [7, 8, 9, 10, 11]:  # May-Sep
                lower_bounds.append(0.37)
            upper_bounds.append(10.9)  # F2(t)

        for _ in range(12):
            lower_bounds.append(0.0)  # F3(t)
            upper_bounds.append(10.9)
        for _ in range(12):
            lower_bounds.append(0.0)  # F4(t)
            upper_bounds.append(36.1)

        # Delayed F2 values
        lower_bounds.append(_f2_ref)  # F2-1 = F2REF
        upper_bounds.append(_f2_ref)  # F2-1 = F2REF
        lower_bounds.append(_f2_ref)  # F2-2 = F2REF
        upper_bounds.append(_f2_ref)  # F2-2 = F2REF

        return jnp.array(lower_bounds), jnp.array(upper_bounds)
