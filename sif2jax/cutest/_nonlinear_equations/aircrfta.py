import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class AIRCRFTA(AbstractNonlinearEquations):
    """The aircraft stability problem by Rheinboldt.

    The aircraft stability problem by Rheinboldt, as a function
    of the elevator, aileron and rudder deflection controls.

    Source: Problem 9 in
    J.J. More',"A collection of nonlinear model problems"
    Proceedings of the AMS-SIAM Summer Seminar on the Computational
    Solution of Nonlinear Systems of Equations, Colorado, 1988.
    Argonne National Laboratory MCS-P60-0289, 1989.

    SIF input: Ph. Toint, Dec 1989.

    Classification: NOR2-RN-8-5
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Control values
    ELVVAL: float = 0.1  # Elevator
    AILVAL: float = 0.0  # Aileron
    RUDVAL: float = 0.0  # Rudder deflection

    @property
    def n(self):
        """Number of variables (including fixed controls)."""
        return 8

    @property
    def y0(self):
        """Initial guess."""
        y = jnp.zeros(8)
        # Set control values
        y = y.at[5].set(self.ELVVAL)  # ELEVATOR
        y = y.at[6].set(self.AILVAL)  # AILERON
        y = y.at[7].set(self.RUDVAL)  # RUDDERDF
        return y

    @property
    def args(self):
        """No additional arguments."""
        return None

    def constraint(self, y):
        """Compute the system of nonlinear equations.

        Variables are:
        0: ROLLRATE
        1: PITCHRAT
        2: YAWRATE
        3: ATTCKANG
        4: SSLIPANG
        5: ELEVATOR (fixed)
        6: AILERON (fixed)
        7: RUDDERDF (fixed)
        """
        # Extract variables
        rollrate = y[0]
        pitchrat = y[1]
        yawrate = y[2]
        attckang = y[3]
        sslipang = y[4]
        elevator = y[5]
        aileron = y[6]
        rudderdf = y[7]

        # Define 2PR elements (product of two variables)
        e1a = pitchrat * yawrate
        e1b = yawrate * attckang
        e1c = attckang * sslipang
        e1d = pitchrat * attckang
        e2a = rollrate * yawrate
        e2b = rollrate * sslipang
        e3a = rollrate * pitchrat
        e3b = rollrate * attckang

        # Define equations
        equations = jnp.zeros(5)

        # G1
        equations = equations.at[0].set(
            -3.933 * rollrate
            + 0.107 * pitchrat
            + 0.126 * yawrate
            - 9.99 * sslipang
            - 45.83 * aileron
            - 7.64 * rudderdf
            - 0.727 * e1a
            + 8.39 * e1b
            - 684.4 * e1c
            + 63.5 * e1d
        )

        # G2
        equations = equations.at[1].set(
            -0.987 * pitchrat
            - 22.95 * attckang
            - 28.37 * elevator
            + 0.949 * e2a
            + 0.173 * e2b
        )

        # G3
        equations = equations.at[2].set(
            0.002 * rollrate
            - 0.235 * yawrate
            + 5.67 * sslipang
            - 0.921 * aileron
            - 6.51 * rudderdf
            - 0.716 * e3a
            - 1.578 * e3b
            + 1.132 * e1d
        )

        # G4
        equations = equations.at[3].set(
            1.0 * pitchrat - 1.0 * attckang - 1.168 * elevator - 1.0 * e2b
        )

        # G5
        equations = equations.at[4].set(
            -1.0 * yawrate - 0.196 * sslipang - 0.0071 * aileron + 1.0 * e3b
        )

        # Also return constraints to fix control variables
        inequalities = None

        return equations, inequalities

    @property
    def bounds(self):
        """Bounds on variables."""
        # All variables are unbounded except the fixed controls
        lower = jnp.full(8, -jnp.inf)
        upper = jnp.full(8, jnp.inf)

        # Fix control variables
        lower = lower.at[5].set(self.ELVVAL)  # ELEVATOR
        upper = upper.at[5].set(self.ELVVAL)
        lower = lower.at[6].set(self.AILVAL)  # AILERON
        upper = upper.at[6].set(self.AILVAL)
        lower = lower.at[7].set(self.RUDVAL)  # RUDDERDF
        upper = upper.at[7].set(self.RUDVAL)

        return lower, upper

    @property
    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value is 0.0 for exact solution."""
        return jnp.array(0.0)
