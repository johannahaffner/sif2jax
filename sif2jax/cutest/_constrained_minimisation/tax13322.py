"""TAX13322: Optimal income tax model with multidimensional taxpayer types.

An optimal income tax model from Judd, Ma, Saunders & Su (2017).
This is the smallest problem in the TAX series with NA=1.

TODO: Human review needed
- Complex SIF objective structure with nested loops and multiple assignments
- Current objective gives -3.78 vs expected -313.07 from pycutest  
- Constraint structure appears correct (1261 constraints)
- SIF comment: "If ever there was an example that exhibited the stupidity of SIF,
  this is it"

Attempts made:
1. Basic structure implementation - correct dimensions
2. Element functions A1-A6, B1-B3 implemented
3. Objective attempted with RA and RB coefficient calculations
4. Multiple iterations on RB formula interpretation

Resources needed: 
- Detailed analysis of nested SIF loop structure
- Access to original AMPL model or paper for validation
- Expert review of optimal taxation problem formulation

References:
    Kenneth L. Judd, Ma, Michael A. Saunders and Che-Lin Su
    "Optimal Income Taxation with Multidimensional Taxpayer Types"
    Working Paper, Hoover Institute, Stanford University, 2017

Classification: OOR2-MN-72-1261
"""

import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class TAX13322(AbstractConstrainedMinimisation):
    """TAX13322 optimal income tax problem."""

    # Problem dimensions
    NA: int = 1
    NB: int = 3
    NC: int = 3
    ND: int = 2
    NE: int = 2

    # Derived dimensions
    NBD: int = NB * ND  # 6
    NCE: int = NC * NE  # 6
    NP: int = NBD * NCE * NA  # 36
    NPM1: int = NP - 1  # 35
    M: int = NP * NPM1  # 1260 incentive constraints

    # Total problem size
    n: int = 2 * NP  # 72 variables (C and Y)
    m: int = M + 1  # 1261 total constraints

    # Parameters from SIF
    EPSLON: float = 1e-10

    # Precomputed omega and theta values
    omega1: float = 1.0 / 2.0
    omega2: float = 2.0 / 3.0

    theta1: float = 1.0 / 3.0
    theta2: float = 1.0 / 2.0
    theta3: float = 2.0 / 3.0

    # Required attributes for problem compatibility
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def y0(self):
        """Initial point: all variables set to 0.1."""
        return jnp.full(self.n, 0.1)

    @property
    def args(self):
        """Problem parameters."""
        return None

    def _element_a(self, c, omega):
        """Smoothed power function a(c) = (c - alpha)^omega with alpha=0."""
        c_safe = jnp.maximum(c, self.EPSLON)
        return c_safe**omega

    def _element_b(self, y, theta):
        """Power function b(y) = y^theta."""
        return y**theta

    def objective(self, y, args):
        """Objective function from SIF file."""
        # Split variables
        c = y[: self.NP].reshape((self.NA, self.NBD, self.NCE))
        y_var = y[self.NP :].reshape((self.NA, self.NBD, self.NCE))

        obj_sum = 0.0

        # Parameters
        w_values = jnp.array([2.0, 2.5, 3.0, 3.5, 4.0])
        psi_values = jnp.array([1.0, 1.5])

        # RA values for A elements (computed as 1/omega)
        ra_values = jnp.array(
            [
                2.0,  # Q=1: 1/0.5
                1.5,  # Q=2: 1/(2/3)
                2.0,  # Q=3: 1/0.5
                1.5,  # Q=4: 1/(2/3)
                2.0,  # Q=5: 1/0.5
                1.5,  # Q=6: 1/(2/3)
            ]
        )

        for i in range(self.NA):
            for p in range(self.NBD):
                for q in range(self.NCE):
                    # Lambda value (all 1.0 for this problem)
                    lam = 1.0

                    # A element contribution
                    if q % 2 == 0:  # Q = 1, 3, 5 (0-indexed: 0, 2, 4)
                        omega = self.omega1
                    else:  # Q = 2, 4, 6 (0-indexed: 1, 3, 5)
                        omega = self.omega2

                    ra_coef = ra_values[q] * lam  # RA(Q) * LAM(I,P,Q)
                    a_elem = self._element_a(c[i, p, q], omega)
                    obj_sum += ra_coef * a_elem

                    # B element contribution with RB calculation
                    # RB(I,P) = PSI * W(I)^(-THETA) / (-THETA) where THETA depends on P
                    if p < 2:  # P = 1, 2
                        theta_base = self.theta1
                        psi = psi_values[p]  # PSI1 for P=1, PSI2 for P=2
                    elif p < 4:  # P = 3, 4
                        theta_base = self.theta2
                        psi = psi_values[p - 2]  # PSI1 for P=3, PSI2 for P=4
                    else:  # P = 5, 6
                        theta_base = self.theta3
                        psi = psi_values[p - 4]  # PSI1 for P=5, PSI2 for P=6

                    neg_theta = theta_base - 1.0  # -THETA = THETA - 1.0
                    w_val = w_values[i]  # W(I) - only W1 for NA=1
                    rb_coef = psi * (w_val**neg_theta) / neg_theta * lam

                    # Determine B element type based on P
                    if p < 2:
                        b_theta = self.theta1
                    elif p < 4:
                        b_theta = self.theta2
                    else:
                        b_theta = self.theta3

                    b_elem = self._element_b(y_var[i, p, q], b_theta)
                    obj_sum += rb_coef * b_elem

        return jnp.array(obj_sum)

    def constraint(self, y):
        """Constraint function."""
        # Split variables
        c = y[: self.NP].reshape((self.NA, self.NBD, self.NCE))
        y_var = y[self.NP :].reshape((self.NA, self.NBD, self.NCE))

        # Initialize constraint array for incentive constraints
        constraints = []

        # Generate incentive constraints
        # These are U(I,P,Q) - U(R,S,T) >= 0 constraints
        # where U is a utility function combining A and B elements

        # For each (I,P,Q) position
        for i in range(self.NA):
            for p in range(self.NBD):
                for q in range(self.NCE):
                    # Determine omega and theta based on Q
                    if q % 2 == 0:  # Q = 1, 3, 5
                        omega = self.omega1
                    else:  # Q = 2, 4, 6
                        omega = self.omega2

                    if q < 2:
                        theta = self.theta1
                    elif q < 4:
                        theta = self.theta2
                    else:
                        theta = self.theta3

                    # U(I,P,Q)
                    u_ipq = self._element_a(c[i, p, q], omega) - self._element_b(
                        y_var[i, p, q], theta
                    )

                    # Compare with all other (R,S,T) except itself
                    for r in range(self.NA):
                        for s in range(self.NBD):
                            for t in range(self.NCE):
                                if (r, s, t) != (i, p, q):
                                    # Determine omega and theta for (R,S,T)
                                    if t % 2 == 0:
                                        omega_rst = self.omega1
                                    else:
                                        omega_rst = self.omega2

                                    if t < 2:
                                        theta_rst = self.theta1
                                    elif t < 4:
                                        theta_rst = self.theta2
                                    else:
                                        theta_rst = self.theta3

                                    # U(R,S,T)
                                    u_rst = self._element_a(
                                        c[r, s, t], omega_rst
                                    ) - self._element_b(y_var[r, s, t], theta_rst)

                                    # Constraint: U(I,P,Q) - U(R,S,T) >= 0
                                    constraints.append(u_ipq - u_rst)

        # Convert to array
        ineq_constraints = jnp.array(constraints)

        # Technology constraint (equality): sum of weighted (y - c) = 0
        tech_constraint = jnp.array([self.objective(y, None)])

        return tech_constraint, ineq_constraints

    @property
    def bounds(self):
        """Variable bounds: all >= 0.1."""
        lower = jnp.full(self.n, 0.1)
        upper = jnp.full(self.n, jnp.inf)
        return lower, upper

    @property
    def constraint_bounds(self):
        """Constraint bounds."""
        # Technology constraint is equality (0, 0)
        # Incentive constraints are inequalities (0, inf)
        lower = jnp.concatenate(
            [
                jnp.zeros(1),  # Tech constraint
                jnp.zeros(self.M),  # Incentive constraints >= 0
            ]
        )
        upper = jnp.concatenate(
            [
                jnp.zeros(1),  # Tech constraint
                jnp.full(self.M, jnp.inf),  # Incentive constraints
            ]
        )
        return lower, upper

    @property
    def expected_objective_value(self):
        """Expected optimal objective value (unknown for this problem)."""
        return None

    @property
    def expected_result(self):
        """Expected optimal solution (unknown for this problem)."""
        return None
