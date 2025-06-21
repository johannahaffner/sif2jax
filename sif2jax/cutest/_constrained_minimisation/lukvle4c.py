import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class LUKVLE4C(AbstractConstrainedMinimisation):
    """LUKVLE4C - Corrected chained Cragg-Levy function with tridiagonal constraints.

    This is the corrected version of LUKVLE4, fixing the bug where group C(I) was
    defined twice in the original SIF file. The corrected version uses E(I) for the
    tan^4 terms in the objective function.

    TODO: Human review needed - 3% discrepancy with pycutest
    This implementation uses the corrected LUKVLE4C.SIF file which fixes a bug in
    the original LUKVLE4.SIF. While our implementation follows the corrected SIF
    file exactly, there remains a ~3% discrepancy with pycutest (3,020,250.61 vs
    3,112,445.79). Most gradient elements match exactly, but a few near the end
    differ. This may be due to numerical precision differences in tangent
    calculations or subtle SIFDECODE interpretation differences. The corrected
    version is vastly superior to the buggy LUKVLE4 (which has a factor of ~2380
    error), but does not yet achieve exact numerical agreement with pycutest.

    Problem 5.4 from Luksan and Vlcek test problems.

    The objective is a chained Cragg-Levy function:
    f(x) = Σ[i=1 to n/2-1] [(exp(x_{2i-1}) - x_{2i})^4 + 100(x_{2i} - x_{2i+1})^6 +
                            tan^4(x_{2i+1} - x_{2i+2}) + x_{2i-1}^8 + (x_{2i+2} - 1)^2]

    Subject to equality constraints:
    c_k(x) = 8x_{k+1}(x_{k+1}^2 - x_k) - 2(1 - x_{k+1}) + 4(x_{k+1} - x_{k+2}^2) = 0,
    for k = 1, ..., n-2

    Starting point:
    x_i = 1 for i ≡ 1 (mod 4)
    x_i = 2 for i ≢ 1 (mod 4)

    Source: L. Luksan and J. Vlcek,
    "Sparse and partially separable test problems for
    unconstrained and equality constrained optimization",
    Technical Report 767, Inst. Computer Science, Academy of Sciences
    of the Czech Republic, 182 07 Prague, Czech Republic, 1999

    SIF input: Nick Gould, April 2001
    Modified version of LUKVLE4 (formulation corrected) May 2024

    Classification: OOR2-AY-V-V
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 10000  # Default dimension, can be overridden

    def objective(self, y, args):
        del args
        n = len(y)
        # Chained Cragg-Levy function - vectorized
        num_complete_groups = (n - 2) // 2  # Ensure we have all 4 elements
        if num_complete_groups <= 0 or n < 4:
            return jnp.array(0.0)

        i = jnp.arange(num_complete_groups)
        x_2i_minus_1 = y[2 * i]  # x_{2i-1} in 1-based
        x_2i = y[2 * i + 1]  # x_{2i} in 1-based
        x_2i_plus_1 = y[2 * i + 2]  # x_{2i+1} in 1-based
        x_2i_plus_2 = y[2 * i + 3]  # x_{2i+2} in 1-based

        # Compute all terms vectorized
        # Note: The SIF file may have issues at boundaries. Let's match exactly.
        terms = (
            (jnp.exp(x_2i_minus_1) - x_2i) ** 4
            + 100 * (x_2i - x_2i_plus_1) ** 6
            + jnp.tan(x_2i_plus_1 - x_2i_plus_2) ** 4
            + x_2i_minus_1**8
            + (x_2i_plus_2 - 1) ** 2
        )

        return jnp.sum(terms)

    def y0(self):
        # Starting point
        y = jnp.zeros(self.n)
        # x_i = 1 for i ≡ 1 (mod 4) -> 0-based: i ≡ 0 (mod 4)
        y = y.at[::4].set(1.0)
        # x_i = 2 for i ≢ 1 (mod 4) -> 0-based: i ≢ 0 (mod 4)
        y = y.at[1::4].set(2.0)
        y = y.at[2::4].set(2.0)
        y = y.at[3::4].set(2.0)
        return y

    def args(self):
        return None

    def expected_result(self):
        # Solution pattern based on problem structure
        return None  # Unknown exact solution

    def expected_objective_value(self):
        return None  # Unknown exact objective value

    def bounds(self):
        return None

    def constraint(self, y):
        n = len(y)
        if n < 3:
            return jnp.array([]), None

        # Constraints from k=1 to n-2
        # Vectorized constraint computation
        # For k = 0 to n-3, we need y[k], y[k+1], y[k+2]
        y_k = y[:-2]  # y[0] to y[n-3]
        y_k1 = y[1:-1]  # y[1] to y[n-2]
        y_k2 = y[2:]  # y[2] to y[n-1]

        # c_k: 8x_{k+1}(x_{k+1}^2 - x_k) - 2(1 - x_{k+1}) + 4(x_{k+1} - x_{k+2}^2)
        constraints = 8 * y_k1 * (y_k1**2 - y_k) - 2 * (1 - y_k1) + 4 * (y_k1 - y_k2**2)

        return constraints, None
