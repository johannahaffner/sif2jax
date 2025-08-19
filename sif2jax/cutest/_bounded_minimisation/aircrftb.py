import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class AIRCRFTB(AbstractBoundedMinimisation):
    """The aircraft stability problem by Rheinboldt (variant B).

    The aircraft stability problem by Rheinboldt, as a function
    of the elevator, aileron and rudder deflection controls.
    This is formulated as an unconstrained minimization problem
    with fixed control values.

    Source: Problem 9 in
    J.J. More',"A collection of nonlinear model problems"
    Proceedings of the AMS-SIAM Summer Seminar on the Computational
    Solution of Nonlinear Systems of Equations, Colorado, 1988.
    Argonne National Laboratory MCS-P60-0289, 1989.

    SIF input: Ph. Toint, Dec 1989.

    Classification: SXR2-RN-8-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Control values (fixed in the problem)
    ELVVAL: float = -0.05  # Elevator
    AILVAL: float = 0.1  # Aileron
    RUDVAL: float = 0.0  # Rudder deflection

    def objective(self, y, args):
        """Compute the sum of squared residuals for the aircraft stability equations."""
        del args

        # Variables are:
        # 0: ROLLRATE
        # 1: PITCHRAT
        # 2: YAWRATE
        # 3: ATTCKANG
        # 4: SSLIPANG
        # 5: ELEVATOR (fixed at ELVVAL)
        # 6: AILERON (fixed at AILVAL)
        # 7: RUDDERDF (fixed at RUDVAL)

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

        # Define groups (equations to be minimized in L2 sense)
        # G1
        g1 = (
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
        g2 = (
            -0.987 * pitchrat
            - 22.95 * attckang
            - 28.37 * elevator
            + 0.949 * e2a
            + 0.173 * e2b
        )

        # G3
        g3 = (
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
        g4 = 1.0 * pitchrat - 1.0 * attckang - 1.168 * elevator - 1.0 * e2b

        # G5
        g5 = -1.0 * yawrate - 0.196 * sslipang - 0.0071 * aileron + 1.0 * e3b

        # L2 objective: sum of squares
        objective = g1**2 + g2**2 + g3**2 + g4**2 + g5**2

        return objective

    @property
    def y0(self):
        """Initial guess with fixed control values."""
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

    @property
    def bounds(self):
        """Bounds on variables.

        Variables 0-4 are free (rollrate, pitchrat, yawrate, attckang, sslipang).
        Variables 5-7 are fixed at their control values.
        """
        lower = jnp.full(8, -jnp.inf)
        upper = jnp.full(8, jnp.inf)

        # Fix control variables to their exact values
        lower = lower.at[5].set(self.ELVVAL)  # ELEVATOR
        upper = upper.at[5].set(self.ELVVAL)
        lower = lower.at[6].set(self.AILVAL)  # AILERON
        upper = upper.at[6].set(self.AILVAL)
        lower = lower.at[7].set(self.RUDVAL)  # RUDDERDF
        upper = upper.at[7].set(self.RUDVAL)

        return (lower, upper)

    @property
    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value from SIF file."""
        # From SIF: *LO SOLTN 6.4099D-02
        return jnp.array(0.064099)
