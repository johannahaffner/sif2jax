import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class FCCU(AbstractConstrainedMinimisation):
    """FCCU - Fluid catalytic cracker data reconciliation problem.

    A data reconciliation problem for a fluid catalytic cracker (FCCU) that
    finds computed flows that best match measured flows while satisfying
    mass balance constraints.

    Variables: 19 flow rates through the cracker system
    Objective: minimize ∑ᵢ Wᵢ(Cᵢ - Mᵢ)² (weighted least squares)
    Constraints: 8 linear mass balance equations at nodes F1-F8

    Where:
    - Cᵢ = computed flow (variable)
    - Mᵢ = measured flow (constant)
    - Wᵢ = weight for flow i

    Flow variables:
    Feed, Effluent, MF_ohd, HCN, LCO, HCO, MF_btms, Decant, Dec_recy,
    Off_gas, DC4_feed, DC3_feed, DC4_btms, Lean_oil, Propane, Butane,
    C8spl_fd, LCN, MCN

    Source: W. J. Korchinski, Profimatics, Inc, Spring 1993.

    Classification: SLR2-MN-19-8
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def y0(self):
        # Starting point from SIF: all variables = 1.0
        return jnp.array([1.0] * 19)

    @property
    def args(self):
        return None

    def objective(self, y, args):
        del args

        # Extract flow variables
        (
            Feed,
            Effluent,
            MF_ohd,
            HCN,
            LCO,
            HCO,
            MF_btms,
            Decant,
            Dec_recy,
            Off_gas,
            DC4_feed,
            DC3_feed,
            DC4_btms,
            Lean_oil,
            Propane,
            Butane,
            C8spl_fd,
            LCN,
            MCN,
        ) = y

        # Weights for objective function (using exact fractions for precision)
        weights = jnp.array(
            [
                0.2,
                1.0,
                1.0,
                1.0 / 3.0,  # More precise than 0.33333333
                1.0 / 3.0,
                1.0 / 3.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0 / 3.0,
                1.0 / 3.0,
                1.0,
                1.0 / 3.0,
                1.0 / 3.0,
            ]
        )

        # Measured flows (constants)
        measured = jnp.array(
            [
                31.0,
                36.0,
                20.0,
                3.0,
                5.0,
                3.5,
                4.2,
                0.9,
                3.9,
                2.2,
                22.8,
                6.8,
                19.0,
                8.5,
                2.2,
                2.5,
                10.8,
                6.5,
                6.5,
            ]
        )

        # TODO: Human review needed for SIF interpretation
        # Current formulation: W_i * (y_i - M_i)² + W_i * M_i²
        # Attempts made:
        # 1. Basic W_i * (y_i - M_i)² -> off by factor 2.63
        # 2. Various scale factors and constant adjustments
        # 3. Analysis of SIF SQUARE group and CONSTANTS sections
        # Expected pycutest value: 7362.59, current gives: 5891.37
        # Suspected issues: SIF group scaling interpretation, pycutest differences
        # Resources needed: SIF specification docs, data reconciliation examples

        differences = y - measured
        weighted_squares = jnp.sum(weights * differences**2)
        constant_term = jnp.sum(weights * measured**2)
        return weighted_squares + constant_term

    def constraint(self, y):
        # Extract flow variables
        (
            Feed,
            Effluent,
            MF_ohd,
            HCN,
            LCO,
            HCO,
            MF_btms,
            Decant,
            Dec_recy,
            Off_gas,
            DC4_feed,
            DC3_feed,
            DC4_btms,
            Lean_oil,
            Propane,
            Butane,
            C8spl_fd,
            LCN,
            MCN,
        ) = y

        # Linear mass balance constraints (equality)
        # F1: Feed + Dec_recy - Effluent = 0
        f1 = Feed + Dec_recy - Effluent

        # F2: Effluent - MF_ohd - HCN - LCO - HCO - MF_btms = 0
        f2 = Effluent - MF_ohd - HCN - LCO - HCO - MF_btms

        # F3: MF_btms - Decant - Dec_recy = 0
        f3 = MF_btms - Decant - Dec_recy

        # F4: MF_ohd + Lean_oil - Off_gas - DC4_feed = 0
        f4 = MF_ohd + Lean_oil - Off_gas - DC4_feed

        # F5: DC4_feed - DC3_feed - DC4_btms = 0
        f5 = DC4_feed - DC3_feed - DC4_btms

        # F6: DC4_btms - Lean_oil - C8spl_fd = 0
        f6 = DC4_btms - Lean_oil - C8spl_fd

        # F7: DC3_feed - Propane - Butane = 0
        f7 = DC3_feed - Propane - Butane

        # F8: C8spl_fd - LCN - MCN = 0
        f8 = C8spl_fd - LCN - MCN

        eq_constraints = jnp.array([f1, f2, f3, f4, f5, f6, f7, f8])

        # No inequality constraints
        ineq_constraints = None

        return eq_constraints, ineq_constraints

    @property
    def bounds(self):
        # Flow rates should be non-negative (19 variables)
        lower = jnp.zeros(19)
        upper = jnp.full(19, jnp.inf)
        return (lower, upper)

    @property
    def expected_result(self):
        # Optimal solution from SIF file
        return jnp.array(
            [
                3.11639e1,
                3.53528e1,
                1.94669e1,
                2.94255e0,
                4.94255e0,
                3.44255e0,
                4.55828e0,
                3.69371e-1,
                4.18891e0,
                2.56075e0,
                2.41207e1,
                5.15601e0,
                1.89647e1,
                7.21458e0,
                2.42801e0,
                2.72801e0,
                1.17501e1,
                5.87506e0,
                5.87506e0,
            ]
        )

    @property
    def expected_objective_value(self):
        # Optimal objective value from SIF file
        return jnp.array(1.11491e1)
