"""
A simple Origin/Destination matrix fit using a minimum entropy approach.

The objective is a combination of different aims, namely to be close to an
a priori matrix for some entries, to be consistent with some traffic counts
(for some entries) and to be small (for entries where nothing else is known).

The objective function is of the form:

    SUM   m T [ ln( T / a ) - 1 ] + E   SUM  T [ ln ( T  ) - 1 ]
   i in I  i i       i   i            i in J  i        i

           +  g   SUM   q  F [ ln( F / c ) - 1 ]
                i in K   i  i       i   i

with the constraints that all Ti and Fi be positive and that

                    F  =  SUM p   T
                     i     j   ij  j

where the pij represent path weights from an a priori assignment.

Source: a modification of an example in
L.G. Willumsen,
"Origin-Destination Matrix: static estimation"
in "Concise Encyclopedia of Traffic and Transportation Systems"
(M. Papageorgiou, ed.), Pergamon Press, 1991.

M. Bierlaire, private communication, 1991.

SIF input: Ph Toint, Dec 1991.

classification OLR2-MN-10-6
"""

import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class ODFITS(AbstractConstrainedMinimisation):
    @property
    def name(self) -> str:
        return "ODFITS"

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 10  # 4 T variables + 6 F variables
    m: int = 6  # 6 equality constraints

    @property
    def y0(self):
        # Start from the a priori known values where available
        # Variables: T13, T14, T23, T24, F1, F2, F3, F4, F5, F6
        return jnp.array(
            [
                90.0,  # T13 (APV13)
                450.0,  # T14 (APV14)
                360.0,  # T23 (APV23)
                1.0,  # T24 (no prior info)
                100.0,  # F1 (TC1)
                500.0,  # F2 (TC2)
                400.0,  # F3 (TC3)
                1100.0,  # F4 (TC4)
                600.0,  # F5 (TC5)
                700.0,  # F6 (TC6)
            ]
        )

    @property
    def args(self):
        # Store all the problem parameters
        # A priori values
        apv = jnp.array([90.0, 450.0, 360.0])  # APV13, APV14, APV23

        # Traffic counts
        tc = jnp.array([100.0, 500.0, 400.0, 1100.0, 600.0, 700.0])

        # Quality of traffic counts
        qlt = jnp.ones(6)

        # mu coefficients for entries with a priori values
        mu = jnp.array([0.5, 0.5, 0.5])  # MU13, MU14, MU23

        # Path weights matrix (6x4)
        # Row i is for arc i, columns are for T13, T14, T23, T24
        p = jnp.array(
            [
                [1.0, 0.0, 0.0, 0.0],  # Arc 1: P131, P141, P231, P241
                [0.0, 1.0, 0.0, 0.0],  # Arc 2: P132, P142, P232, P242
                [0.0, 0.0, 1.0, 0.0],  # Arc 3: P133, P143, P233, P243
                [0.0, 1.0, 1.0, 1.0],  # Arc 4: P134, P144, P234, P244
                [0.0, 0.0, 1.0, 1.0],  # Arc 5: P135, P145, P235, P245
                [0.0, 0.0, 0.0, 1.0],  # Arc 6: P136, P146, P236, P246
            ]
        )

        # Other parameters
        gamma = 1.5
        entrop = 0.2

        return (apv, tc, qlt, mu, p, gamma, entrop)

    def objective(self, y, args):
        apv, tc, qlt, mu, p, gamma, entrop = args

        # Extract variables
        t13, t14, t23, t24 = y[:4]
        f = y[4:]

        # Based on AMPL formulation:
        # minimize f:
        # sum {j in KNOWN} MU[j]*((T[j]*log(T[j]/APV[j]))-T[j])
        # + sum {j in UNKNOWN} ENTROP*((T[j]*log(T[j]))-T[j])
        # + sum {i in 1..ARCS} (QLT[i]/GAMMA)*((F[i]*log(F[i]/TC[i]))-F[i])

        # First term: entries with a priori known values (I = {13, 14, 23})
        term1 = (
            mu[0] * (t13 * jnp.log(t13 / apv[0]) - t13)
            + mu[1] * (t14 * jnp.log(t14 / apv[1]) - t14)
            + mu[2] * (t23 * jnp.log(t23 / apv[2]) - t23)
        )

        # Second term: entries without information (J = {24})
        term2 = entrop * (t24 * jnp.log(t24) - t24)

        # Third term: arc flows (K = {1,2,3,4,5,6})
        term3 = jnp.sum((qlt / gamma) * (f * jnp.log(f / tc) - f))

        return jnp.array(term1 + term2 + term3)

    def constraint(self, y):
        apv, tc, qlt, mu, p, gamma, entrop = self.args

        # Extract variables
        t = y[:4]  # T13, T14, T23, T24
        f = y[4:]  # F1, F2, F3, F4, F5, F6

        # Constraints: F_i = sum_j p_ij * T_j
        # This is: f = p @ t
        # Pycutest expects the opposite sign
        equality_constraints = p @ t - f

        return equality_constraints, None

    @property
    def bounds(self):
        # All variables have lower bound 0.1
        lower = jnp.full(self.n, 0.1)
        upper = jnp.full(self.n, jnp.inf)
        return lower, upper

    @property
    def expected_result(self):
        # The optimal solution is not explicitly given in the SIF file
        return None

    @property
    def expected_objective_value(self):
        # According to the SIF file comment (line 266),
        # the optimal objective value is -2380.026775
        return jnp.array(-2380.026775)
