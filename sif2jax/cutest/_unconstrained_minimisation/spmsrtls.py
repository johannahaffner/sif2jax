import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class SPMSRTLS(AbstractUnconstrainedMinimisation):
    """Liu and Nocedal tridiagonal matrix square root least-squares problem.

    This is a least-squares variant of problem SPMSQRT. We seek a tridiagonal
    matrix X such that X^T * X approximates a given tridiagonal matrix A in
    the least-squares sense.

    Source: problem 151 (p. 93) in
    A.R. Buckley,
    "Test functions for unconstrained minimization",
    TR 1989CS-3, Mathematics, statistics and computing centre,
    Dalhousie University, Halifax (CDN), 1989.

    SIF input: Ph. Toint, Dec 1989.

    Classification: SUR2-AN-V-V
    """

    _m: int = 1667  # Default dimension
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables: 3*M-2."""
        return 3 * self._m - 2

    @property
    def args(self):
        """No additional arguments."""
        return None

    def _compute_b_matrix(self):
        """Compute the entries of the tridiagonal matrix B."""
        m = self._m

        # Initialize B matrix entries
        b = {}

        # Row 1
        b[(1, 1)] = jnp.sin(1.0)
        b[(1, 2)] = jnp.sin(4.0)

        # Rows 2 to M-1
        k = 2
        for i in range(2, m):
            for j in [i - 1, i, i + 1]:
                k += 1
                rk = float(k)
                b[(i, j)] = jnp.sin(rk * rk)

        # Row M
        k += 1
        rk = float(k)
        b[(m, m - 1)] = jnp.sin(rk * rk)
        k += 1
        rk = float(k)
        b[(m, m)] = jnp.sin(rk * rk)

        return b

    def _compute_a_matrix(self, b):
        """Compute the pentadiagonal matrix A from B."""
        m = self._m
        a = {}

        # Diagonal entries
        a[(1, 1)] = b[(1, 1)] * b[(1, 1)] + b[(1, 2)] * b[(2, 1)]

        for i in range(2, m):
            a[(i, i)] = (
                b[(i, i)] * b[(i, i)]
                + b[(i - 1, i)] * b[(i, i - 1)]
                + b[(i + 1, i)] * b[(i, i + 1)]
            )

        a[(m, m)] = b[(m, m)] * b[(m, m)] + b[(m - 1, m)] * b[(m, m - 1)]

        # Sub-diagonal entries
        for i in range(1, m):
            a[(i + 1, i)] = (
                b[(i + 1, i)] * b[(i, i)] + b[(i + 1, i + 1)] * b[(i + 1, i)]
            )

        # Super-diagonal entries
        for i in range(2, m + 1):
            a[(i - 1, i)] = (
                b[(i - 1, i)] * b[(i, i)] + b[(i - 1, i - 1)] * b[(i - 1, i)]
            )

        # Sub-sub-diagonal entries
        for i in range(2, m):
            a[(i + 1, i - 1)] = b[(i + 1, i)] * b[(i, i - 1)]

        # Super-super-diagonal entries
        for i in range(2, m):
            a[(i - 1, i + 1)] = b[(i - 1, i)] * b[(i, i + 1)]

        return a

    @property
    def y0(self):
        """Initial guess: 0.2 * B matrix entries."""
        m = self._m
        b = self._compute_b_matrix()
        y0 = jnp.zeros(self.n)

        # X(1,1) and X(1,2)
        y0 = y0.at[0].set(0.2 * b[(1, 1)])
        y0 = y0.at[1].set(0.2 * b[(1, 2)])

        # X(i,i-1), X(i,i), X(i,i+1) for i = 2 to M-1
        idx = 2
        for i in range(2, m):
            y0 = y0.at[idx].set(0.2 * b[(i, i - 1)])
            y0 = y0.at[idx + 1].set(0.2 * b[(i, i)])
            y0 = y0.at[idx + 2].set(0.2 * b[(i, i + 1)])
            idx += 3

        # X(M,M-1) and X(M,M)
        y0 = y0.at[idx].set(0.2 * b[(m, m - 1)])
        y0 = y0.at[idx + 1].set(0.2 * b[(m, m)])

        return y0

    def objective(self, y, args):
        """Compute the least-squares objective function.

        The objective is the sum of squared differences between entries of
        A and X^T * X, where X is the tridiagonal matrix formed from y.
        """
        del args  # Not used

        m = self._m

        # Compute target matrix A
        b = self._compute_b_matrix()
        a = self._compute_a_matrix(b)

        # Extract X matrix entries from y
        x11 = y[0]
        x12 = y[1]

        # Initialize objective
        obj = 0.0

        # Line 1 of X^T * X
        # E(1,1): D(1) + G(1) where D(1) = x11^2, G(1) = x12 * x21
        # Since X is tridiagonal, x21 = x12
        e11 = x11**2 + x12 * x12
        obj += (e11 - a[(1, 1)]) ** 2

        # E(1,2): H(1) + R(1) where H(1) = x11 * x12, R(1) = x12 * x22
        idx = 2
        x22 = y[idx + 1] if m > 2 else y[idx]  # X(2,2)
        e12 = x11 * x12 + x12 * x22
        obj += (e12 - a.get((1, 2), 0.0)) ** 2

        # E(1,3): S(1) where S(1) = x12 * x23
        if m > 2:
            x23 = y[idx + 2]  # X(2,3)
            e13 = x12 * x23
            obj += (e13 - a.get((1, 3), 0.0)) ** 2

        # Lines 2 to M-1 of X^T * X
        idx = 2
        for i in range(2, m):
            # Extract X entries for row i
            xi_im1 = y[idx]  # X(i,i-1)
            xi_i = y[idx + 1]  # X(i,i)
            xi_ip1 = y[idx + 2] if i < m - 1 else 0.0  # X(i,i+1)

            # Also need entries from adjacent rows
            if i > 2:
                xim1_im2 = y[idx - 3]  # X(i-1,i-2)
                xim1_im1 = y[idx - 2]  # X(i-1,i-1)
                xim1_i = y[idx - 1]  # X(i-1,i)
            else:
                xim1_im2 = 0.0
                xim1_im1 = x11
                xim1_i = x12

            if i < m - 1:
                xip1_i = xi_ip1  # X(i+1,i) = X(i,i+1) by symmetry
                xip1_ip1 = y[idx + 4] if i < m - 2 else y[idx + 3]  # X(i+1,i+1)
                xip1_ip2 = y[idx + 5] if i < m - 2 else 0.0  # X(i+1,i+2)
            else:
                xip1_i = y[idx + 2]  # X(M,M-1)
                xip1_ip1 = y[idx + 3]  # X(M,M)
                xip1_ip2 = 0.0

            # E(i,i-2): A(i) = xi_im1 * xim1_im2
            if i > 2:
                e_iim2 = xi_im1 * xim1_im2
                obj += (e_iim2 - a.get((i, i - 2), 0.0)) ** 2

            # E(i,i-1): B(i) + C(i) where B(i) = xi_im1 * xim1_im1, C(i) = xi_i * xi_im1
            e_iim1 = xi_im1 * xim1_im1 + xi_i * xi_im1
            obj += (e_iim1 - a.get((i, i - 1), 0.0)) ** 2

            # E(i,i): F(i) + D(i) + G(i)
            # F(i) = xi_im1 * xim1_i, D(i) = xi_i^2, G(i) = xi_ip1 * xip1_i
            f_i = xi_im1 * xim1_i
            d_i = xi_i**2
            g_i = xi_ip1 * xip1_i if i < m else 0.0
            e_ii = f_i + d_i + g_i
            obj += (e_ii - a.get((i, i), 0.0)) ** 2

            # E(i,i+1): H(i) + R(i)
            # H(i) = xi_i * xi_ip1, R(i) = xi_ip1 * xip1_ip1
            if i < m:
                h_i = xi_i * xi_ip1
                r_i = xi_ip1 * xip1_ip1
                e_iip1 = h_i + r_i
                obj += (e_iip1 - a.get((i, i + 1), 0.0)) ** 2

            # E(i,i+2): S(i) = xi_ip1 * xip1_ip2
            if i < m - 1:
                s_i = xi_ip1 * xip1_ip2
                e_iip2 = s_i
                obj += (e_iip2 - a.get((i, i + 2), 0.0)) ** 2

            idx += 3

        # Line M of X^T * X
        xm_mm1 = y[-2]  # X(M,M-1)
        xm_m = y[-1]  # X(M,M)

        # Need X(M-1,M-2), X(M-1,M-1), X(M-1,M)
        if m > 2:
            xm1_mm2 = y[-5]  # X(M-1,M-2)
            xm1_mm1 = y[-4]  # X(M-1,M-1)
            xm1_m = y[-3]  # X(M-1,M)
        else:
            xm1_mm2 = 0.0
            xm1_mm1 = x11
            xm1_m = x12

        # E(M,M-2): A(M) = xm_mm1 * xm1_mm2
        if m > 2:
            em_mm2 = xm_mm1 * xm1_mm2
            obj += (em_mm2 - a.get((m, m - 2), 0.0)) ** 2

        # E(M,M-1): B(M) + C(M)
        em_mm1 = xm_mm1 * xm1_mm1 + xm_m * xm_mm1
        obj += (em_mm1 - a.get((m, m - 1), 0.0)) ** 2

        # E(M,M): F(M) + D(M)
        fm = xm_mm1 * xm1_m
        dm = xm_m**2
        em_m = fm + dm
        obj += (em_m - a.get((m, m), 0.0)) ** 2

        return obj

    @property
    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value (not provided in SIF)."""
        return None
