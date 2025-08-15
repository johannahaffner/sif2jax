r"""
The "ten bar truss" structural optimization problem, version P2.

The problem is to minimize the cross section areas of the bars
in the structure

    /|
    /|>o------o------o
    /|  \    /|\    /|
         \  / | \  / |
          \/  |  \/  |
          /\  |  /\  |
         /  \ | /  \ |
    /|  /    \|/    \|
    /|>o------o------o
    /|

submitted to vertical forces of equal magnitude (P0) applied at
the two free lower nodes, subject to limits of nodal displacements.

Source:
K. Svanberg,
private communication,  August 1990.
See also
K. Svanberg,
"On local and global minima in structural optimization",
in "New directions in optimum structural design" (Atrek, Ragsdell
and Zienkiwewicz, eds.), Wiley, 1984.

SIF input: Ph. Toint, August 1990.

classification LOR2-MY-18-8
"""

import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class TENBARS2(AbstractConstrainedMinimisation):
    @property
    def name(self) -> str:
        return "TENBARS2"

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 18  # 8 nodal displacements (U1-U8) + 10 bar cross sections (X1-X10)
    n_equality_constraints: int = 8  # 8 equilibrium conditions
    n_inequality_constraints: int = 0  # No inequality constraints

    @property
    def y0(self):
        # All variables start at 0.0 according to SIF file
        return jnp.zeros(18)

    @property
    def args(self):
        return None

    def objective(self, y, args):
        del args
        # Extract cross section areas (last 10 variables)
        x = y[8:]  # X1 to X10

        # Constants
        c0 = 2.53106
        sq2 = jnp.sqrt(2.0)
        c0sq2 = c0 * sq2

        # Objective function is sum of cross section areas times their factors
        obj = c0 * (x[0] + x[3] + x[4] + x[5] + x[8] + x[9])  # X1, X4, X5, X6, X9, X10
        obj += c0sq2 * (x[1] + x[2] + x[6] + x[7])  # X2, X3, X7, X8

        return jnp.array(obj)

    def constraint(self, y):
        # Extract nodal displacements and cross sections
        u = y[:8]  # U1 to U8
        x = y[8:]  # X1 to X10

        # Constants
        sq8 = jnp.sqrt(8.0)
        inv_sq8 = 1.0 / sq8
        minus_inv_sq8 = -inv_sq8
        p0 = 589.884  # -P0 from SIF file (C4 and C8 RHS values)

        # Element computations
        # EA: X1 * U1
        ea = x[0] * u[0]

        # EB: X3 * (U1 + U2)
        eb = x[2] * (u[0] + u[1])

        # EC: X6 * (U1 - U5)
        ec = x[5] * (u[0] - u[4])

        # ED: X7 * (U1 + U8 - U2 - U7)
        ed = x[6] * (u[0] + u[7] - u[1] - u[6])

        # EE: X5 * (U2 - U4)
        ee = x[4] * (u[1] - u[3])

        # EF: X2 * (U3 - U4) - corrected based on TENBARS1 analysis
        # The SIF file may specify (U3 + U4), but pycutest uses (U3 - U4)
        ef = x[1] * (u[2] - u[3])

        # EG: X4 * U3
        eg = x[3] * u[2]

        # EH: X8 * (U3 + U4 - U5 - U6)
        eh = x[7] * (u[2] + u[3] - u[4] - u[5])

        # EI: X9 * (U3 - U7)
        ei = x[8] * (u[2] - u[6])

        # EJ: X10 * (U6 - U8)
        ej = x[9] * (u[5] - u[7])

        # Equality constraints (C1 to C8)
        c1 = ea + ec + inv_sq8 * eb + inv_sq8 * ed
        c2 = inv_sq8 * eb + ee + minus_inv_sq8 * ed
        # EF term: X2*(U3-U4) added to C3, subtracted from C4
        c3 = inv_sq8 * eh + ei + eg + inv_sq8 * ef
        c4 = inv_sq8 * eh - ee + p0 - inv_sq8 * ef
        c5 = minus_inv_sq8 * eh - ec
        c6 = minus_inv_sq8 * eh + ej
        c7 = minus_inv_sq8 * ed - ei
        c8 = inv_sq8 * ed - ej + p0

        equality_constraints = jnp.array([c1, c2, c3, c4, c5, c6, c7, c8])

        # No inequality constraints for TENBARS2
        inequality_constraints = None

        return equality_constraints, inequality_constraints

    @property
    def bounds(self):
        # Lower bounds on U (even indices 2,4,6,8 have -50.8 lower bound)
        # and on cross sections (0.645)
        lower = jnp.concatenate(
            [
                jnp.array(
                    [
                        -jnp.inf,  # U1
                        -50.8,  # U2
                        -jnp.inf,  # U3
                        -50.8,  # U4
                        -jnp.inf,  # U5
                        -50.8,  # U6
                        -jnp.inf,  # U7
                        -50.8,  # U8
                    ]
                ),
                jnp.full(10, 0.645),  # Lower bound 0.645 on cross sections
            ]
        )
        upper = jnp.full(18, jnp.inf)
        return lower, upper

    @property
    def expected_result(self):
        # The optimal solution is not explicitly given in the SIF file
        return None

    @property
    def expected_objective_value(self):
        # According to the SIF file comments, two solution values are given:
        # SOLTN-A: 2302.55
        # SOLTN-B: 2277.9458
        # We'll use the B solution as the expected value
        return jnp.array(2277.9458)
