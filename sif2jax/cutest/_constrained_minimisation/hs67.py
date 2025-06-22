import jax
import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


# TODO: Human review needed
# Attempts made:
# 1. Verified our implementation matches the SIF file exactly
# 2. Confirmed coefficient -0.063 for Y2*Y5 term in our SIF file
# Suspected issues:
# - pycutest may be using a different SIF file version with coefficient ~0.062
# - Our f* = -1162.036507 matches H&S book, pycutest f* = -868.636 doesn't
# Additional resources needed: "Verify which SIF file version pycutest uses"
class HS67(AbstractConstrainedMinimisation):
    """Problem 67 from the Hock-Schittkowski test collection (Colville No.8).

    A 3-variable nonlinear objective function with 14 inequality constraints and bounds.

    f(x) = 5.04*x₁ + 0.035*x₂ + 10*x₃ - 0.063*y₂(x)*y₅(x) + 3.36*y₃(x)

    where yⱼ(x) : cf. Appendix A

    Subject to:
        yᵢ₊₁(x) - aᵢ ≥ 0, i=1,...,7
        aᵢ₊₇ - yᵢ₊₁(x) ≥ 0, i=1,...,7
        1.E-5 ≤ x₁ ≤ 2.E3
        1.E-5 ≤ x₂ ≤ 1.6E4
        1.E-5 ≤ x₃ ≤ 1.2E2

    Source: problem 67 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    Also cited: Colville [20], Himmelblau [29]

    Classification: GGI-P1-1

    Note: This problem uses fixed point iterations to solve the Y-function system from
    Appendix A.
    The Y2 and Y4 functions are computed using jax.lax.scan for mathematically correct
    and differentiable convergence as specified in the original Hock-Schittkowski book.

    Note on pycutest discrepancy: Our implementation matches the SIF file and H&S book
    with f* = -1162.036507. However, pycutest reports f* = -868.636, which would require
    a coefficient of approximately 0.062 instead of 0.063 for the Y2*Y5 term.
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def _fixed_point_scan(self, update_fn, init_val, max_iter=50):
        """Solve fixed point equation using jax.lax.scan for differentiability."""

        def scan_body(carry, _):
            return update_fn(carry), None

        # Use scan for a fixed number of iterations
        final_val, _ = jax.lax.scan(scan_body, init_val, None, length=max_iter)
        return final_val

    def _compute_y_functions(self, x1, x2, x3):
        """Compute Y functions using fixed point iteration from Appendix A."""

        # Y2 update function
        def y2_update(y2):
            # Y3 = 1.22*Y2 - x1
            y3 = 1.22 * y2 - x1
            # Y6 = (x2 + Y3) / x1
            y6 = (x2 + y3) / x1
            # Y2c = 0.01*x1*(112 + 13.167*Y6 - 0.6667*Y6^2)
            return 0.01 * x1 * (112 + 13.167 * y6 - 0.6667 * y6**2)

        # Solve for Y2 using scan
        y2_init = 1.6 * x1
        y2 = self._fixed_point_scan(y2_update, y2_init)

        # Compute Y3 and Y6 from converged Y2
        y3 = 1.22 * y2 - x1
        y6 = (x2 + y3) / x1

        # Y4 update function (depends on converged y2)
        def y4_update(y4):
            # Y5 = 86.35 + 1.098*Y6 - 0.038*Y6^2 + 0.325*(Y4 - 89)
            y5 = 86.35 + 1.098 * y6 - 0.038 * y6**2 + 0.325 * (y4 - 89)
            # Y8 = 3*Y5 - 133
            y8 = 3 * y5 - 133
            # Y7 = 35.82 - 0.222*Y8
            y7 = 35.82 - 0.222 * y8
            # Y4c = 98000*x3 / (Y2*Y7 + 1000*x3)
            return 98000 * x3 / (y2 * y7 + 1000 * x3)

        # Solve for Y4 using scan
        y4_init = 93.0
        y4 = self._fixed_point_scan(y4_update, y4_init)

        # Compute final Y5, Y7, Y8 from converged Y4
        y5 = 86.35 + 1.098 * y6 - 0.038 * y6**2 + 0.325 * (y4 - 89)
        y8 = 3 * y5 - 133
        y7 = 35.82 - 0.222 * y8

        return y2, y3, y4, y5, y6, y7, y8

    def objective(self, y, args):
        x1, x2, x3 = y
        # Compute Y functions using Appendix A algorithm
        y2, y3, y4, y5, y6, y7, y8 = self._compute_y_functions(x1, x2, x3)

        # SIF file: f(x) = 5.04*x1 + 0.035*x2 + 10.0*x3 - 0.063*y2*y5 + 3.36*y3
        return 5.04 * x1 + 0.035 * x2 + 10.0 * x3 - 0.063 * y2 * y5 + 3.36 * y3

    def y0(self):
        return jnp.array([1745.0, 12000.0, 110.0])  # feasible according to the problem

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([1728.371286, 16000.00000, 98.14151402])

    def expected_objective_value(self):
        return jnp.array(-1162.036507)  # This matches H&S book and our SIF file

    def bounds(self):
        return (jnp.array([1.0e-5, 1.0e-5, 1.0e-5]), jnp.array([2.0e3, 1.6e4, 1.2e2]))

    def constraint(self, y):
        x1, x2, x3 = y
        # Compute Y functions using Appendix A algorithm
        y2, y3, y4, y5, y6, y7, y8 = self._compute_y_functions(x1, x2, x3)

        # A constants from Appendix A table
        A = jnp.array(
            [
                0.0,
                0.0,
                85.0,
                90.0,
                3.0,
                0.01,
                145.0,
                5000.0,
                2000.0,
                93.0,
                95.0,
                12.0,
                4.0,
                162.0,
            ]
        )

        Y = jnp.array([0.0, y2, y3, y4, y5, y6, y7, y8])  # Y(1) is not used

        # Constraints from SIF:
        # AG(i): Y(i+1) - A(i) ≥ 0 for i=1,...,7
        # AL(i): Y(i+1) - A(i+7) ≤ 0 for i=1,...,7 (converted to -Y(i+1) + A(i+7) ≥ 0)

        ineq_constraints = []
        # AG constraints: Y(i+1) - A(i) ≥ 0 for i=1,...,7
        for i in range(1, 8):  # i=1,2,...,7
            ineq_constraints.append(
                Y[i] - A[i - 1]
            )  # Y(i) - A(i-1) since both are 0-indexed and Y[1]=y2, Y[2]=y3, etc.

        # AL constraints: -Y(i+1) + A(i+7) ≥ 0 for i=1,...,7
        for i in range(1, 8):  # i=1,2,...,7
            ineq_constraints.append(
                A[i + 6] - Y[i]
            )  # A(i+6) - Y(i) since both are 0-indexed

        inequality_constraints = jnp.array(ineq_constraints)
        return None, inequality_constraints
