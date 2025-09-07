"""DRUGDIS problem implementation."""

import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractConstrainedMinimisation


class DRUGDIS(AbstractConstrainedMinimisation):
    """Drug displacement control problem.

    A control problem based on the kinetic model of Aarons and Rowland for
    DRUG DISplacement, which simulates the interaction of the two drugs
    (warfarin and phenylbutazone) in a patient bloodstream.
    The state variables are the concentrations of unbound warfarin (w) and
    phenylbutazone (p). The problem is to control the rate of injection (u)
    of the pain-killing phenylbutazone so that both drugs reach a specified
    steady-state in minimum time and the concentration of warfarin does not
    rise above a given toxicity level.

    The problem is discretized using the trapezoidal rule. It is non-convex.

    Source:
    H. Maurer and M. Wiegand,
    "Numerical solution of a drug displacement problem with bounded state
    variables",
    Optimal Control Applications and Methods 13, pp. 43-55, 1992.

    SIF input: Ph. Toint, Nov 1993.
    correction by S. Gratton & Ph. Toint, May 2024

    Classification: LOR2-MN-V-V
    """

    # Problem parameters
    toxic: float = 0.026  # warfarin toxicity level
    wss: float = 0.02  # initial/final warfarin levels
    umax: float = 8.0  # maximal injection rate
    pstart: float = 0.0  # initial phenylbutazone level
    pfinal: float = 2.0  # final phenylbutazone level

    # Model parameters
    z: float = 46.4  # interaction coefficient
    dcoeff: float = 0.2  # diffusion coefficient
    abcnst: float = 232.0  # constant in dynamics

    # Discretization parameter
    ni: int = 2000  # number of interior points + 1 (default from SIF)

    # SIF scaling factors (pycutest applies these internally)
    tf_scale: float = 200.0  # TF variable scale
    tf_obj_scale: float = 100.0  # TF objective scale
    w_scale: float = 0.02  # W variable scale
    ew_scale: float = 0.02  # EW constraint scale

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n_var(self) -> int:
        """Number of variables."""
        return 1 + 3 * (self.ni + 1)  # TF + W(0..NI) + P(0..NI) + U(0..NI)

    @property
    def n_con(self) -> int:
        """Number of constraints."""
        return 2 * self.ni  # EW(0..NI-1) + EP(0..NI-1)

    def objective(self, y: Array, args) -> Array:
        """Objective function: minimize final time.

        PyCUTEst applies internal scaling but the API uses actual values.
        The objective has an internal scale factor of 100.
        """
        del args
        return y[0] / self.tf_obj_scale  # TF / 100

    def constraint(self, y: Array) -> tuple[Array, None]:
        """Constraint functions: state equations (vectorized).

        PyCUTEst applies internal scaling but the API uses actual values.
        """
        tf = y[0]  # TF actual value
        w = y[1 : self.ni + 2]  # W(0..NI) actual values
        p = y[self.ni + 2 : 2 * self.ni + 3]  # P(0..NI)
        u = y[2 * self.ni + 3 :]  # U(0..NI)

        # State equations using trapezoidal rule
        dt_ni = tf / (2.0 * self.ni)

        # Vectorized dynamics calculation for all points
        fw = self._warfarin_dynamics_vectorized(w, p, u)
        fp = self._phenylbutazone_dynamics_vectorized(w, p, u)

        # Compute constraints using slicing (vectorized)
        # EW(i): w[i+1] - w[i] - dt_ni * (fw[i] + fw[i+1])
        ew = w[1:] - w[:-1] - dt_ni * (fw[:-1] + fw[1:])
        # Apply EW inverse scaling: SIF has 'SCALE' 0.02, so divide by it
        ew = ew / self.ew_scale

        # EP(i): p[i+1] - p[i] - dt_ni * (fp[i] + fp[i+1])
        ep = p[1:] - p[:-1] - dt_ni * (fp[:-1] + fp[1:])
        # EP constraints have no scaling in SIF

        # Interleave constraints: EW(0), EP(0), EW(1), EP(1), ...
        constraints = jnp.empty(2 * self.ni)
        constraints = constraints.at[0::2].set(ew)
        constraints = constraints.at[1::2].set(ep)

        return constraints, None

    def _warfarin_dynamics_vectorized(self, w: Array, p: Array, u: Array) -> Array:
        """Compute warfarin concentration dynamics (vectorized)."""
        d = 1.0 + self.dcoeff * (w + p)
        dd = d * d

        a = dd + self.abcnst + self.z * w
        b = dd + self.abcnst + self.z * p
        c = a * b - self.z * self.z * w * p

        f = dd / c
        g = a * (self.wss - w) + self.z * w * (u - 2.0 * p)

        return f * g

    def _phenylbutazone_dynamics_vectorized(
        self, w: Array, p: Array, u: Array
    ) -> Array:
        """Compute phenylbutazone concentration dynamics (vectorized)."""
        d = 1.0 + self.dcoeff * (w + p)
        dd = d * d

        a = dd + self.abcnst + self.z * w
        b = dd + self.abcnst + self.z * p
        c = a * b - self.z * self.z * w * p

        f = dd / c
        g = b * (u - 2.0 * p) + self.z * p * (self.wss - w)

        return f * g

    @property
    def y0(self) -> Array:
        """Starting point for optimization (vectorized)."""
        x = jnp.zeros(self.n_var)

        # TF actual value
        x = x.at[0].set(240.0)

        # W(i) - all at steady state
        x = x.at[1 : self.ni + 2].set(self.wss)

        # P(i) - linear interpolation from pstart to pfinal
        p_values = jnp.linspace(self.pstart, self.pfinal, self.ni + 1)
        x = x.at[self.ni + 2 : 2 * self.ni + 3].set(p_values)

        # U(i) - at maximum rate for i=0..NI-1, U(NI) defaults to 0
        x = x.at[2 * self.ni + 3 : -1].set(self.umax)
        # U(NI) is not set in SIF START POINT, defaults to 0

        return x

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def expected_result(self) -> Array:
        """Expected result is not known analytically."""
        return self.y0  # Placeholder

    @property
    def expected_objective_value(self) -> Array:
        """Expected objective value from SIF file comments."""
        # From SIF file: SOLTN(10) = 3.82432, but we use NI=2000
        # The actual optimal value is not provided for NI=2000
        return jnp.array([])  # Not known for NI=2000

    @property
    def bounds(self) -> tuple[Array, Array]:
        """Variable bounds (vectorized)."""
        # All variables are non-negative (from SIF)
        lower = jnp.zeros(self.n_var)
        # Use infinity for no upper bound
        upper = jnp.full(self.n_var, jnp.inf)

        # TF >= 200.0 (overrides the 0 lower bound)
        lower = lower.at[0].set(200.0)

        # W(i) <= toxic for all i
        upper = upper.at[1 : self.ni + 2].set(self.toxic)

        # W(0) and W(NI) are fixed
        lower = lower.at[1].set(self.wss)
        upper = upper.at[1].set(self.wss)
        lower = lower.at[1 + self.ni].set(self.wss)
        upper = upper.at[1 + self.ni].set(self.wss)

        # P(0) and P(NI) are fixed
        lower = lower.at[self.ni + 2].set(self.pstart)
        upper = upper.at[self.ni + 2].set(self.pstart)
        lower = lower.at[2 * self.ni + 2].set(self.pfinal)
        upper = upper.at[2 * self.ni + 2].set(self.pfinal)

        # U(i) <= umax for i=0..NI-1 only (SIF specifies bounds for U(0..NI-1))
        upper = upper.at[2 * self.ni + 3 : -1].set(self.umax)
        # U(NI) has no explicit upper bound in SIF

        return lower, upper

    def num_constraints(self) -> tuple[int, int, int]:
        """Returns the number of constraints and bounds.

        Returns:
            (num_equality_constraints, num_inequality_constraints, num_finite_bounds)
        """
        # Count finite bounds (those that are not ±inf)
        lower, upper = self.bounds
        num_finite_lower = jnp.sum(jnp.isfinite(lower))
        num_finite_upper = jnp.sum(jnp.isfinite(upper))
        # pycutest counts fixed variables as 2 bounds (both lower and upper)
        num_finite_bounds = int(num_finite_lower + num_finite_upper)

        # All constraints are equality constraints
        return self.n_con, 0, num_finite_bounds
