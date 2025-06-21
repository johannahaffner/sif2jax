import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class CSFI1(AbstractConstrainedMinimisation):
    """CSFI1 problem - continuous caster product dimensions optimization.

    Source: problem MAXTPH in Vasko and Stott
    "Optimizing continuous caster product dimensions:
     an example of a nonlinear design problem in the steel industry"
    SIAM Review, Vol 37 No, 1 pp.82-84, 1995

    SIF input: A.R. Conn April 1995

    Classification: LOR2-RN-5-4
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return 5

    def objective(self, y, args):
        """Compute the objective (-TPH to maximize TPH)."""
        del args

        # Extract variables
        thick, wid, len_, tph, ipm = y

        # Objective is -TPH (minimizing -TPH = maximizing TPH)
        return -tph

    def constraint(self, y):
        """Compute the constraints separated into equalities and inequalities."""
        # Extract variables
        thick, wid, len_, tph, ipm = y

        # Parameters
        maxaspr = 2.0
        minarea = 200.0

        # CMPLQ element: 117.3708920187793427 * V1 / (V2 * V3)
        cmplq_const = 117.3708920187793427
        e1 = cmplq_const * tph / (wid * thick)

        # SQQUT element: V1 * (V1 * V2 / 48.0)
        e2 = thick * (thick * ipm / 48.0)

        # QUOTN element: V1 / V2
        e3 = wid / thick

        # PROD element: V1 * V2
        e4 = thick * wid

        # Equality constraints:
        # 1. CIPM: E1 - IPM = 0
        c1 = e1 - ipm

        # 2. CLEN: E2 - LEN = 0
        c2 = e2 - len_

        # Inequality constraints:
        # 3. WOT: E3 <= MAXASPR means E3 - MAXASPR <= 0
        c3 = e3 - maxaspr

        # 4. TTW: MINAREA <= E4 <= MAXAREA
        # pycutest returns E4 - MINAREA for the lower bound check
        c4 = e4 - minarea

        # Return equalities and inequalities separately
        equalities = jnp.array([c1, c2])
        inequalities = jnp.array([c3, c4])

        return equalities, inequalities

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
        lenmax = 60.0
        minthick = 7.0

        # Bounds from SIF file
        lower = jnp.array([minthick, 0.0, 0.0, 0.0, 0.0])
        upper = jnp.array([jnp.inf, jnp.inf, lenmax, jnp.inf, jnp.inf])

        return lower, upper

    def expected_result(self):
        """Expected optimal solution (from commented start point)."""
        return jnp.array([10.01, 20.02, 60.0, 49.1, 28.75])

    def expected_objective_value(self):
        """Expected optimal objective value."""
        # From OBJECT BOUND section
        return jnp.array(-49.1)
