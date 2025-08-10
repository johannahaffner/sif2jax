import jax.numpy as jnp

from ..._problem import AbstractConstrainedQuadraticProblem


class GOULDQP1(AbstractConstrainedQuadraticProblem):
    """GOULDQP1 problem - problem 118 from Hock and Schittkowski modified by Gould.

    Source: problem 118 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981, as modified by N.I.M. Gould in "An algorithm
    for large-scale quadratic programming", IMA J. Num. Anal (1991),
    11, 299-324, problem class 1.

    SIF input: B Baudson, Jan 1990 modified by Nick Gould, Jan, 2011

    Classification: QLR2-AN-32-17
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return 32  # 15 X variables + 4 AS + 4 CS + 4 BS + 5 DS = 32 total

    @property
    def m(self):
        """Number of constraints."""
        return 17  # 4*3 + 5 = 17 constraints

    def objective(self, y, args):
        """Compute the objective."""
        del args

        x = y[:15]  # X(1) to X(15)

        # Linear terms: sum over k=0 to 4 of coefficients for X(3k+1), X(3k+2), X(3k+3)
        linear_obj = (
            2.3 * (x[0] + x[3] + x[6] + x[9] + x[12])  # X(1,4,7,10,13)
            + 1.7 * (x[1] + x[4] + x[7] + x[10] + x[13])  # X(2,5,8,11,14)
            + 2.2 * (x[2] + x[5] + x[8] + x[11] + x[14])  # X(3,6,9,12,15)
        )

        # Quadratic terms with specific coefficients from GROUP USES section
        quadratic_coeffs = jnp.array(
            [
                -1.0,  # E1 (X1^2)
                0.0001,  # E2 (X2^2)
                0.00015,  # E3 (X3^2)
                -0.0001,  # E4 (X4^2)
                0.0001,  # E5 (X5^2)
                10.0,  # E6 (X6^2)
                -0.0001,  # E7 (X7^2)
                0.0001,  # E8 (X8^2)
                25.0,  # E9 (X9^2)
                -2.5,  # E10 (X10^2)
                0.0001,  # E11 (X11^2)
                0.00015,  # E12 (X12^2)
                -0.0001,  # E13 (X13^2)
                0.0001,  # E14 (X14^2)
                0.00015,  # E15 (X15^2)
            ]
        )

        quadratic_obj = jnp.sum(quadratic_coeffs * x * x)

        return linear_obj + quadratic_obj

    def constraint(self, y):
        """Compute the constraints."""
        x = y[:15]  # X(1) to X(15)
        as_vars = y[15:19]  # AS(1) to AS(4)
        cs_vars = y[19:23]  # CS(1) to CS(4)
        bs_vars = y[23:27]  # BS(1) to BS(4)
        ds_vars = y[27:32]  # DS(1) to DS(5)

        constraints = []

        # A(K), B(K), C(K) constraints for K=1 to 4
        # From SIF: A(K): X(3K+1) - X(3K-2) - AS(K) = -7.0
        # From SIF: B(K): X(3K+3) - X(3K) - BS(K) = -7.0
        # From SIF: C(K): X(3K+2) - X(3K-1) - CS(K) = -7.0
        for k in range(1, 5):  # k = 1,2,3,4
            # Convert to 0-based indexing
            # A(K): X(3K+1) - X(3K-2) - AS(K) = -7.0
            i1 = 3 * k + 1 - 1  # X(3K+1) → 0-based
            i2 = 3 * k - 2 - 1  # X(3K-2) → 0-based
            ak = x[i1] - x[i2] - as_vars[k - 1] + 7.0
            constraints.append(ak)

            # B(K): X(3K+3) - X(3K) - BS(K) = -7.0
            i1 = 3 * k + 3 - 1  # X(3K+3) → 0-based
            i2 = 3 * k - 1  # X(3K) → 0-based
            bk = x[i1] - x[i2] - bs_vars[k - 1] + 7.0
            constraints.append(bk)

            # C(K): X(3K+2) - X(3K-1) - CS(K) = -7.0
            i1 = 3 * k + 2 - 1  # X(3K+2) → 0-based
            i2 = 3 * k - 1 - 1  # X(3K-1) → 0-based
            ck = x[i1] - x[i2] - cs_vars[k - 1] + 7.0
            constraints.append(ck)

        # D constraints
        d_constraints = [
            x[0] + x[1] + x[2] - ds_vars[0] - 60.0,  # D1
            x[3] + x[4] + x[5] - ds_vars[1] - 50.0,  # D2
            x[6] + x[7] + x[8] - ds_vars[2] - 70.0,  # D3
            x[9] + x[10] + x[11] - ds_vars[3] - 85.0,  # D4
            x[12] + x[13] + x[14] - ds_vars[4] - 100.0,  # D5
        ]
        constraints.extend(d_constraints)

        return jnp.array(constraints), None

    def equality_constraints(self):
        """All constraints are equalities."""
        return jnp.ones(self.m, dtype=bool)

    @property
    def y0(self):
        """Initial guess."""
        # Specific initial values from SIF file
        y0_vals = jnp.array(
            [
                20.0,  # X1 (default)
                55.0,  # X2
                15.0,  # X3
                20.0,  # X4 (default)
                60.0,  # X5
                20.0,  # X6 (default)
                20.0,  # X7 (default)
                60.0,  # X8
                20.0,  # X9 (default)
                20.0,  # X10 (default)
                60.0,  # X11
                20.0,  # X12 (default)
                20.0,  # X13 (default)
                60.0,  # X14
                20.0,  # X15 (default)
                7.0,  # AS1
                7.0,  # AS2
                7.0,  # AS3
                7.0,  # AS4
                12.0,  # CS1
                7.0,  # CS2
                7.0,  # CS3
                7.0,  # CS4
                12.0,  # BS1
                7.0,  # BS2
                7.0,  # BS3
                7.0,  # BS4
                30.0,  # DS1
                50.0,  # DS2
                30.0,  # DS3
                15.0,  # DS4
                0.0,  # DS5
            ]
        )

        return y0_vals

    @property
    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    @property
    def bounds(self):
        """Variable bounds."""
        # In SIF format, variables default to lower bound 0.0 if not specified
        lower = jnp.full(self.n, 0.0)
        upper = jnp.full(self.n, jnp.inf)

        # X variable bounds with explicit lower bounds
        lower = lower.at[0].set(8.0)  # X1 >= 8.0
        upper = upper.at[0].set(21.0)  # X1 <= 21.0

        lower = lower.at[1].set(43.0)  # X2 >= 43.0
        upper = upper.at[1].set(57.0)  # X2 <= 57.0

        lower = lower.at[2].set(3.0)  # X3 >= 3.0
        upper = upper.at[2].set(16.0)  # X3 <= 16.0

        # X4-X15 bounds: only upper bounds specified, lower defaults to 0.0
        for k in range(1, 5):  # k=1,2,3,4
            upper = upper.at[3 * k].set(90.0)  # X(3K+1) <= 90.0  (X4,X7,X10,X13)
            upper = upper.at[3 * k + 1].set(120.0)  # X(3K+2) <= 120.0 (X5,X8,X11,X14)
            upper = upper.at[3 * k + 2].set(60.0)  # X(3K+3) <= 60.0  (X6,X9,X12,X15)

        # AS, BS, CS bounds
        for k in range(4):
            lower = lower.at[15 + k].set(0.0)  # AS(K) >= 0.0
            upper = upper.at[15 + k].set(13.0)  # AS(K) <= 13.0

            lower = lower.at[19 + k].set(0.0)  # CS(K) >= 0.0
            upper = upper.at[19 + k].set(14.0)  # CS(K) <= 14.0

            lower = lower.at[23 + k].set(0.0)  # BS(K) >= 0.0
            upper = upper.at[23 + k].set(13.0)  # BS(K) <= 13.0

        # DS bounds
        ds_bounds = [60.0, 50.0, 70.0, 85.0, 100.0]
        for i, bound in enumerate(ds_bounds):
            lower = lower.at[27 + i].set(0.0)
            upper = upper.at[27 + i].set(bound)

        return lower, upper

    @property
    def expected_result(self):
        """Expected optimal solution."""
        # From SOLUTION section in SIF file
        return jnp.array(
            [
                2.1000e01,
                4.3000e01,
                3.0000e00,
                2.7000e01,
                3.6000e01,
                4.5469e-07,
                3.3000e01,
                3.7000e01,
                6.7853e-07,
                3.9000e01,
                4.4000e01,
                2.0000e00,
                4.1000e01,
                5.1000e01,
                8.0000e00,
                6.0000e00,
                6.0000e00,
                6.0000e00,
                2.0000e00,
                -7.0000e00,
                1.0000e00,
                7.0000e00,
                7.0000e00,
                -3.0000e00,
                2.2384e-07,
                2.0000e00,
                6.0000e00,
                6.7000e01,
                6.3000e01,
                7.0000e01,
                8.5000e01,
                1.0000e02,
            ]
        )

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        # From comment: Solution approximately -3.485333E+3
        return jnp.array(-3485.333)
