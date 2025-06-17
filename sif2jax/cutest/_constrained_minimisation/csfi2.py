import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class CSFI2(AbstractConstrainedMinimisation):
    """CSFI2 problem - continuous caster product dimensions optimization.

    Source: problem MINLEN in Vasko and Stott
    "Optimizing continuous caster product dimensions:
     an example of a nonlinear design problem in the steel industry"
    SIAM Review, Vol 37 No, 1 pp.82-84, 1995

    SIF input: A.R. Conn April 1995

    Classification: LOR2-RN-5-4
    """

    @property
    def n(self):
        """Number of variables."""
        return 5

    @property
    def m(self):
        """Number of constraints."""
        return 5

    def objective(self, y, args):
        """Compute the objective (LEN to minimize length)."""
        del args

        # Extract variables
        thick, wid, len_, tph, ipm = y

        # Objective is LEN
        return len_

    def constraint(self, y):
        """Compute the constraints."""
        # Extract variables
        thick, wid, len_, tph, ipm = y

        # Parameters
        maxaspr = 2.0
        minarea = 200.0
        maxarea = 250.0

        # CMPLQ element: 117.3708920187793427 * V1 / (V2 * V3)
        cmplq_const = 117.3708920187793427
        e1 = cmplq_const * tph / (wid * thick)

        # SQQUT element: V1 * (V1 * V2 / 48.0)
        e2 = thick * (thick * ipm / 48.0)

        # QUOTN element: V1 / V2
        e3 = wid / thick

        # PROD element: V1 * V2
        e4 = thick * wid

        # Constraints:
        # 1. CIPM: IPM - E1 = 0
        c1 = ipm - e1

        # 2. CLEN: LEN - E2 = 0
        c2 = len_ - e2

        # 3. WOT: E3 <= MAXASPR
        c3 = e3 - maxaspr

        # 4. TTW: MINAREA <= E4 <= MAXAREA (converted to two inequalities)
        # E4 <= MAXAREA
        c4 = e4 - maxarea

        # E4 >= MINAREA (which becomes -E4 + MINAREA <= 0)
        c5 = minarea - e4

        # Return all constraints
        return jnp.array([c1, c2, c3, c4, c5]), None

    def equality_constraints(self):
        """First two constraints are equalities, rest are inequalities."""
        return jnp.array([True, True, False, False, False])

    def y0(self):
        """Initial guess."""
        # From START POINT section with default 0.5
        return jnp.array([0.5, 0.5, 0.5, 0.5, 0.5])

    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    def bounds(self):
        """Variable bounds."""
        # Parameters
        minthick = 7.0
        mintph = 45.0

        # Bounds from SIF file
        lower = jnp.array([minthick, 0.0, 0.0, mintph, 0.0])
        upper = jnp.array([jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf])

        return lower, upper

    def expected_result(self):
        """Expected optimal solution (from commented start point)."""
        return jnp.array([10.01, 20.02, 60.0, 49.1, 28.75])

    def expected_objective_value(self):
        """Expected optimal objective value."""
        # From OBJECT BOUND section
        return jnp.array(55.0)
