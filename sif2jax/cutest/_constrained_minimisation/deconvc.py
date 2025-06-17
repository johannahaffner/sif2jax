import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class DECONVC(AbstractConstrainedMinimisation):
    """DECONVC problem - deconvolution analysis (constrained version).

    A problem arising in deconvolution analysis.

    Source: J.P. Rasson, Private communication, 1996.

    SIF input: Ph. Toint, Nov 1996.

    Classification: SQR2-MN-61-1
    """

    @property
    def n(self):
        """Number of variables."""
        # Variables are C(-LGSG:LGTR) and SG(1:LGSG)
        # Total: (LGTR - (-LGSG) + 1) + LGSG = LGTR + 2*LGSG + 1
        lgtr = 40
        lgsg = 11
        return lgtr + 2 * lgsg + 1  # 40 + 22 + 1 = 63

    @property
    def m(self):
        """Number of constraints."""
        return 1  # Energy constraint

    def objective(self, y, args):
        """Compute the sum of squares objective."""
        del args

        lgtr = 40
        lgsg = 11

        # Extract variables
        # C goes from index -LGSG to LGTR, so indices 0 to LGTR+LGSG
        c = y[: lgtr + lgsg + 1]  # 52 values
        sg = y[lgtr + lgsg + 1 :]  # 11 values

        # Data values TR
        tr = jnp.array(
            [
                0.0,
                0.0,
                1.6e-3,
                5.4e-3,
                7.02e-2,
                0.1876,
                0.332,
                0.764,
                0.932,
                0.812,
                0.3464,
                0.2064,
                8.3e-2,
                3.4e-2,
                6.18e-2,
                1.2,
                1.8,
                2.4,
                9.0,
                2.4,
                1.801,
                1.325,
                7.62e-2,
                0.2104,
                0.268,
                0.552,
                0.996,
                0.36,
                0.24,
                0.151,
                2.48e-2,
                0.2432,
                0.3602,
                0.48,
                1.8,
                0.48,
                0.36,
                0.264,
                6e-3,
                6e-3,
            ]
        )

        # Compute residuals R(K) for K = 1 to LGTR
        obj = 0.0
        for k in range(lgtr):  # k = 0 to 39 (represents K = 1 to 40)
            # R(K) = (sum of SG(I) * C(K-I+1) for I = 1 to LGSG - TR(K))^2
            rk = -tr[k]
            for i in range(lgsg):  # i = 0 to 10 (represents I = 1 to 11)
                k_minus_i_plus_1 = k - i  # This is K-I+1 in 0-indexed
                # C index needs adjustment: C goes from -LGSG to LGTR
                # In our array, index 0 corresponds to C(-LGSG)
                # So C(K-I+1) is at index K-I+1+LGSG
                c_idx = k_minus_i_plus_1 + lgsg
                if 0 <= c_idx < len(c):
                    # Only include if IDX > 0 (i.e., K-I+1 > 0)
                    if k_minus_i_plus_1 >= 0:
                        rk += sg[i] * c[c_idx]

            obj += rk * rk

        return jnp.array(obj)

    def constraint(self, y):
        """Compute the energy constraint."""
        lgtr = 40
        lgsg = 11

        # Extract SG variables
        sg = y[lgtr + lgsg + 1 :]

        # Energy constraint: sum(SG(I)^2) = PIC
        pic = 12.35
        energy = jnp.sum(sg * sg) - pic

        return jnp.array([energy]), None

    def equality_constraints(self):
        """Energy constraint is an equality."""
        return jnp.ones(1, dtype=bool)

    def y0(self):
        """Initial guess."""
        lgtr = 40
        lgsg = 11

        # Initial C values (all zeros as given)
        c_init = jnp.zeros(lgtr + lgsg + 1)

        # Initial SG values
        sg_init = jnp.array(
            [1e-2, 2e-2, 0.4, 0.6, 0.8, 3.0, 0.8, 0.6, 0.44, 1e-2, 1e-2]
        )

        return jnp.concatenate([c_init, sg_init])

    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    def bounds(self):
        """Variable bounds."""
        lgtr = 40
        lgsg = 11
        n_total = lgtr + 2 * lgsg + 1

        # Default bounds
        lower = jnp.full(n_total, -jnp.inf)
        upper = jnp.full(n_total, jnp.inf)

        # C(K) for K = -LGSG to 0 are fixed at 0
        # These are indices 0 to LGSG in our array
        for i in range(lgsg + 1):
            lower = lower.at[i].set(0.0)
            upper = upper.at[i].set(0.0)

        return lower, upper

    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return None

    def expected_objective_value(self):
        """Expected optimal objective value (not provided in SIF)."""
        return None
