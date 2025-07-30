import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractConstrainedMinimisation


# TODO: Human review needed
# Attempts made: Verified implementation against SIF file
# Suspected issues: Large Jacobian differences (max 92.82) suggest fundamental issue
# Resources needed: Deep analysis of constraint formulation differences
class EXPFITC(AbstractConstrainedMinimisation):
    """One-sided rational approximation to the exponential function.

    The problem involves fitting a rational function to approximate the
    exponential function over the interval [0, 5] with 251 fitting points.

    Source: M.J.D. Powell,
    "A tolerant algorithm for linearly constrained optimization calculations",
    Mathematical Programming 45(3), pp.561-562, 1989.

    SIF input: Ph. Toint and N. Gould, May 1990.

    Classification: OLR2-AN-5-502
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self) -> int:
        """Number of variables: P0, P1, P2, Q1, Q2."""
        return 5

    @property
    def m(self) -> int:
        """Number of constraints: 502 linear constraints (251 × 2)."""
        return 502

    def starting_point(self) -> Array:
        """Return the starting point for the problem."""
        return jnp.array([1.0, 1.0, 6.0, 0.0, 0.0], dtype=jnp.float64)

    def objective(self, y: Array, args) -> Array:
        """Compute the sum of squared residuals."""
        del args
        P0, P1, P2, Q1, Q2 = y

        # Number of fitting points
        R = 251

        # Fitting points: T(i) = 5*(i-1)/(R-1) for i = 1, ..., R
        i_values = jnp.arange(1, R + 1, dtype=jnp.float64)  # 1 to 251
        T = 5.0 * (i_values - 1.0) / (R - 1.0)  # [0, 0.02, 0.04, ..., 5.0]

        # Target values: ET(i) = exp(T(i))
        ET = jnp.exp(T)

        # Numerator polynomial: PT(i) = P0 + P1*T(i) + P2*T(i)^2
        PT = P0 + P1 * T + P2 * T**2

        # Denominator polynomial: QT(i) = 1.0 + Q1*(T(i)-5) + Q2*(T(i)-5)^2
        T_shift = T - 5.0
        QT = 1.0 + Q1 * T_shift + Q2 * T_shift**2

        # ETQT(i) = exp(T(i)) * QT(i)
        ETQT = ET * QT

        # Residuals: F(i) = PT(i)/ETQT(i) - 1.0
        F = PT / ETQT - 1.0

        # Objective: sum of squared residuals
        objective = jnp.sum(F**2)

        return objective

    def constraint(self, y: Array):
        """Compute the linear constraints."""
        P0, P1, P2, Q1, Q2 = y

        # Number of fitting points
        R = 251

        # Fitting points
        i_values = jnp.arange(1, R + 1, dtype=jnp.float64)
        T = 5.0 * (i_values - 1.0) / (R - 1.0)

        # Target values: ET(i) = exp(T(i))
        ET = jnp.exp(T)

        # Compute QC1(i) = (T(i)-5) * exp(T(i)) and QC2(i) = (T(i)-5)^2 * exp(T(i))
        T_shift = T - 5.0
        QC1 = T_shift * ET
        QC2 = T_shift**2 * ET

        # C(i) constraints: P0 + P1*T(i) + P2*T(i)^2 - Q1*QC1(i) - Q2*QC2(i) >= ET(i)
        # Rearranged as: P0 + P1*T(i) + P2*T(i)^2 - Q1*QC1(i) - Q2*QC2(i) - ET(i) >= 0
        PT = P0 + P1 * T + P2 * T**2
        C_constraints = PT - Q1 * QC1 - Q2 * QC2 - ET

        # B(i) constraints: Q1*(T(i)-5) + Q2*(T(i)-5)^2 >= -0.99999
        # Rearranged as: Q1*(T(i)-5) + Q2*(T(i)-5)^2 + 0.99999 >= 0
        B_constraints = Q1 * T_shift + Q2 * T_shift**2 + 0.99999

        # Combine all constraints (502 total)
        inequality_constraints = jnp.concatenate([C_constraints, B_constraints])

        return None, inequality_constraints

    @property
    def bounds(self) -> tuple[Array, Array] | None:
        """All variables are free (unbounded)."""
        return None

    @property
    def y0(self) -> Array:
        """Initial guess for the optimization problem."""
        return self.starting_point()

    @property
    def args(self):
        """Additional arguments for the objective and constraint functions."""
        return None

    @property
    def expected_result(self):
        """Expected result of the optimization problem."""
        # No specific solution given in SIF file
        return jnp.zeros(self.n, dtype=jnp.float64)

    @property
    def expected_objective_value(self):
        """Expected value of the objective at the solution."""
        # Solution value given as 2.5196e-3
        return jnp.array(2.5196e-3, dtype=jnp.float64)
