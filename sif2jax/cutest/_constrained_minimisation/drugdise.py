"""DRUGDISE problem implementation.

TODO: Human review needed
Attempts made: [Initial implementation with complex element function structure]
Suspected issues: [Complex element function formulation (3S, 3D2, DSQ, 3PR),
                  multiple interdependent intermediate variables A/B/C,
                  trapezoidal integration with nonlinear terms]
Resources needed: [Deep understanding of SIF element function semantics,
                  careful analysis of GROUP USES section,
                  validation against pycutest reference implementation]

The problem has passed basic structural tests but failed:
- Starting value test (dimension/calculation issues)
- Constraint evaluation tests (formulation errors)
- Jacobian tests (large discrepancies up to 2153.52)

The constraint structure involves sophisticated element functions:
- 3S: V1 * V2 * V3 * (WSS - V4)
- 3D2: V1 * V2 * V3 * (V4 - 2*V5)
- DSQ: (0.2*V1 + 0.2*V2)^2
- 3PR: V1 * V2 * V3

With GROUP USES coefficients like -1/NI, -Z/NI, -ZZ that require careful implementation.
"""

import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractConstrainedMinimisation


class DRUGDISE(AbstractConstrainedMinimisation):
    """Drug displacement control problem (expanded version).

    This is a variant of the drug displacement problem DRUGDIS where the
    state equations have been Expanded in term of more intermediate
    functions, each one of them being less nonlinear.

    A control problem based on the kinetic model of Aarons and Rowland for
    DRUG DISplacement Expanded, which simulates the interaction of the two drugs
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

    Classification: LOR2-MY-V-V
    """

    # Problem parameters
    toxic: float = 0.026  # warfarin toxicity level
    wss: float = 0.02  # initial/final warfarin levels
    umax: float = 8.0  # maximal injection rate
    pstart: float = 0.0  # initial phenylbutazone level
    pfinal: float = 2.0  # final phenylbutazone level
    z: float = 46.4  # interaction coefficient

    # Model parameters from SIF
    abcnst: float = 232.0  # constant in dynamics
    dcoeff: float = 0.2  # diffusion coefficient

    # Discretization parameter
    ni: int = 1000  # number of interior points + 1 (default from SIF)

    # SIF scaling factors (pycutest applies these internally)
    tf_scale: float = 200.0  # TF variable scale
    tf_obj_scale: float = 100.0  # TF objective scale
    w_scale: float = 0.02  # W variable scale
    ew_scale: float = 0.02  # EW constraint scale
    a_scale: float = 200.0  # A variable scale
    b_scale: float = 200.0  # B variable scale
    c_scale: float = 0.0000001  # C variable scale
    ea_scale: float = 200.0  # EA constraint scale
    eb_scale: float = 200.0  # EB constraint scale

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n_var(self) -> int:
        """Number of variables."""
        # TF + W(0..NI) + P(0..NI) + U(0..NI-1) + A(0..NI-1) + B(0..NI-1) + C(0..NI-1)
        return 1 + 2 * (self.ni + 1) + 4 * self.ni

    @property
    def n_con(self) -> int:
        """Number of constraints."""
        # EW(0..NI-1) + EP(0..NI-1) + EA(0..NI-1) + EB(0..NI-1) + EC(0..NI-1)
        return 5 * self.ni

    def objective(self, y: Array, args) -> Array:
        """Objective function: minimize final time.

        PyCUTEst applies internal scaling but the API uses actual values.
        The objective has an internal scale factor of 100.
        """
        del args
        return y[0] / self.tf_obj_scale  # TF / 100

    def constraint(self, y: Array) -> tuple[Array, None]:
        """Constraint functions: state equations and intermediate variable
        definitions (vectorized).

        PyCUTEst applies internal scaling but the API uses actual values.
        """
        tf = y[0]  # TF actual value
        w = y[1 : self.ni + 2]  # W(0..NI)
        p = y[self.ni + 2 : 2 * self.ni + 3]  # P(0..NI)
        u = y[2 * self.ni + 3 : 3 * self.ni + 3]  # U(0..NI-1)
        a = y[3 * self.ni + 3 : 4 * self.ni + 3]  # A(0..NI-1)
        b = y[4 * self.ni + 3 : 5 * self.ni + 3]  # B(0..NI-1)
        c = y[5 * self.ni + 3 : 6 * self.ni + 3]  # C(0..NI-1)

        # Time step
        dt_ni = tf / self.ni

        # Intermediate variables d and dd for vectorized computation
        d = 1.0 + self.dcoeff * (w[:-1] + p[:-1])  # for points 0..NI-1
        dd = d * d

        # State equations using trapezoidal rule
        # EW(i): W(i+1) - W(i) - (TF/NI) * [C(i) * A(i) * (WSS - W(i))
        #        - Z * C(i) * W(i) * (U(i) - 2*P(i))]
        ew_terms = c * a * (self.wss - w[:-1]) - self.z * c * w[:-1] * (
            u - 2.0 * p[:-1]
        )
        ew = w[1:] - w[:-1] - dt_ni * ew_terms
        # Apply EW inverse scaling: SIF has 'SCALE' 0.02, so divide by it
        ew = ew / self.ew_scale

        # EP(i): P(i+1) - P(i) - (TF/NI) * [C(i) * B(i) * (U(i) - 2*P(i))
        #        - Z * C(i) * P(i) * (WSS - W(i))]
        ep_terms = c * b * (u - 2.0 * p[:-1]) - self.z * c * p[:-1] * (
            self.wss - w[:-1]
        )
        ep = p[1:] - p[:-1] - dt_ni * ep_terms

        # EA(i): A(i) - (1+0.2*(W(i)+P(i)))^2 - 232 - Z*P(i) = 0
        ea = a - dd - self.abcnst - self.z * p[:-1]
        # Apply EA inverse scaling
        ea = ea / self.ea_scale

        # EB(i): B(i) - (1+0.2*(W(i)+P(i)))^2 - 232 - Z*W(i) = 0
        eb = b - dd - self.abcnst - self.z * w[:-1]
        # Apply EB inverse scaling
        eb = eb / self.eb_scale

        # EC(i): C(i)*A(i)*B(i) - (1+0.2*(W(i)+P(i)))^2 - Z^2*P(i)*W(i) = 0
        ec = c * a * b - dd - self.z * self.z * p[:-1] * w[:-1]
        # EC has scale 0.0000001 but this appears to be handled differently

        # Concatenate all constraints in order: EW, EP, EA, EB, EC
        constraints = jnp.concatenate([ew, ep, ea, eb, ec])
        return constraints, None

    @property
    def y0(self) -> Array:
        """Starting point for optimization (vectorized)."""
        x = jnp.zeros(self.n_var)

        # TF actual value
        x = x.at[0].set(240.0)

        # W(i) - all at steady state (except boundary conditions handled in bounds)
        x = x.at[1 : self.ni + 2].set(self.wss)

        # P(i) - use AVP for interior points (boundary handled in bounds)
        avp = (self.pstart + self.pfinal) * 0.5
        x = x.at[self.ni + 2 : 2 * self.ni + 3].set(avp)

        # U(i) - at maximum rate for i=0..NI-1
        x = x.at[2 * self.ni + 3 : 3 * self.ni + 3].set(self.umax)

        # Compute starting values for intermediate variables A, B, C
        # From SIF starting point calculation:
        # 2W/10 = WSS * 0.2, 2P/10 = AVP * 0.2, 2(W+P)/10 = 2W/10 + 2P/10
        # D = 2(W+P)/10 + 1.0, DD = D * D
        # ZP = AVP * Z, ZW = WSS * Z
        # AA = DD + ZP + 232, BB = DD + ZW + 232
        # AB = AA * BB, WP = WSS * AVP, -ZZWP = WP * (-ZZ)
        # CD = AB + (-ZZWP), CC = DD / CD
        w2_10 = self.wss * 0.2
        p2_10 = avp * 0.2
        wp2_10 = w2_10 + p2_10
        d_val = wp2_10 + 1.0
        dd_val = d_val * d_val

        zp_val = avp * self.z
        zw_val = self.wss * self.z
        aa_val = dd_val + zp_val + self.abcnst
        bb_val = dd_val + zw_val + self.abcnst

        ab_val = aa_val * bb_val
        wp_val = self.wss * avp
        zz_val = self.z * self.z
        zzwp_val = wp_val * zz_val
        cd_val = ab_val - zzwp_val
        cc_val = dd_val / cd_val

        # A(i), B(i), C(i) - use computed starting values
        x = x.at[3 * self.ni + 3 : 4 * self.ni + 3].set(aa_val)
        x = x.at[4 * self.ni + 3 : 5 * self.ni + 3].set(bb_val)
        x = x.at[5 * self.ni + 3 : 6 * self.ni + 3].set(cc_val)

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
        # Solution not provided in SIF file
        return jnp.array([])

    @property
    def bounds(self) -> tuple[Array, Array]:
        """Variable bounds (vectorized)."""
        # All variables are non-negative (except C variables which are free)
        lower = jnp.zeros(self.n_var)
        # Use infinity for no upper bound
        upper = jnp.full(self.n_var, jnp.inf)

        # TF >= 200.0
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

        # U(i) <= umax for i=0..NI-1
        upper = upper.at[2 * self.ni + 3 : 3 * self.ni + 3].set(self.umax)

        # C(i) variables are free (XR DRUGDISE C(I) in SIF)
        lower = lower.at[5 * self.ni + 3 : 6 * self.ni + 3].set(-jnp.inf)

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
