import jax.numpy as jnp

from ..._problem import AbstractConstrainedQuadraticProblem


class HS90(AbstractConstrainedQuadraticProblem):
    """Hock and Schittkowski problem 90.

    A time-optimal heat conduction problem.

    Source: problem 90 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    SIF input: Nick Gould, September 1991.

    classification QOR2-MN-4-1
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return 4

    @property
    def y0(self):
        """Initial guess."""
        # From SIF: x1 = 0.5, x2 = -0.5, x3 = 0.5, x4 = -0.5
        return jnp.array([0.5, -0.5, 0.5, -0.5], dtype=jnp.float64)

    @property
    def args(self):
        return None

    def objective(self, y, args):
        """Quadratic objective function."""
        del args
        x1, x2, x3, x4 = y[0], y[1], y[2], y[3]

        # Simple quadratic objective: sum of squares
        return x1**2 + x2**2 + x3**2 + x4**2

    @property
    def bounds(self):
        """No bounds on variables."""
        return None

    def constraint(self, y):
        """Nonlinear constraint involving complex heat conduction function."""
        x1, x2, x3, x4 = y[0], y[1], y[2], y[3]

        # This implements the EVAL90 function similar to EVAL88/89
        # MU values from the Fortran code (30 integration points)
        mu = jnp.array(
            [
                8.6033358901938017e-01,
                3.4256184594817283e00,
                6.4372981791719468e00,
                9.5293344053619631e00,
                1.2645287223856643e01,
                1.5771284874815882e01,
                1.8902409956860023e01,
                2.2036496727938566e01,
                2.5172446326646664e01,
                2.8309642854452012e01,
                3.1447714637546234e01,
                3.4586424215288922e01,
                3.7725612827776501e01,
                4.0865170330488070e01,
                4.4005017920830845e01,
                4.7145097736761031e01,
                5.0285366337773652e01,
                5.3425790477394663e01,
                5.6566344279821521e01,
                5.9707007305335459e01,
                6.2847763194454451e01,
                6.5988598698490392e01,
                6.9129502973895256e01,
                7.2270467060308960e01,
                7.5411483488848148e01,
                7.8552545984242926e01,
                8.1693649235601683e01,
                8.4834788718042290e01,
                8.7975960552493220e01,
                9.1117161394464745e01,
            ],
            dtype=y.dtype,
        )

        # Calculate integration constants
        t = 2.0 / 15.0

        # Calculate A and S arrays (integration weights)
        sin_mu = jnp.sin(mu)
        cos_mu = jnp.cos(mu)
        a = 2.0 * sin_mu / (mu + sin_mu * cos_mu)
        s = 2.0 * a * (cos_mu - sin_mu / mu)

        # Build R matrix efficiently using JAX operations
        mu_i = mu[:, None]
        mu_j = mu[None, :]
        a_i = a[:, None]
        a_j = a[None, :]

        # Compute off-diagonal elements
        mu_sum = mu_i + mu_j
        mu_diff = mu_i - mu_j
        aimui2 = a_i * mu_i**2
        ajmuj2 = a_j * mu_j**2

        # Off-diagonal elements
        r_off = (
            0.5
            * (
                jnp.sin(mu_sum) / mu_sum
                + jnp.sin(mu_diff) / jnp.where(mu_diff != 0, mu_diff, 1.0)
            )
            * aimui2
            * ajmuj2
        )

        # Diagonal elements
        aimui2_diag = a * mu**2
        r_diag = 0.5 * (1.0 + 0.5 * jnp.sin(2 * mu) / mu) * aimui2_diag**2

        # Combine to form R matrix
        r = r_off.at[jnp.diag_indices(30)].set(r_diag)

        # Calculate p functions (partial sums of squares)
        # For N=4: p[0] = sum(x_i^2), p[1] = sum(x_i^2, i=2..4),
        # p[2] = sum(x_i^2, i=3..4), p[3] = x4^2, p[4] = 0
        p = jnp.array(
            [
                x1**2 + x2**2 + x3**2 + x4**2,
                x2**2 + x3**2 + x4**2,
                x3**2 + x4**2,
                x4**2,
                0.0,
            ],
            dtype=y.dtype,
        )

        # Calculate rho functions following Fortran logic
        muj2 = mu**2

        # Initial contribution from i=1
        u = jnp.exp(-muj2 * p[0])

        # Alternating contributions for i=2,3,4
        # alpha starts at -2 and alternates
        u = u - 2.0 * jnp.exp(-muj2 * p[1])  # i=2, alpha=-2
        u = u + 2.0 * jnp.exp(-muj2 * p[2])  # i=3, alpha=2
        u = u - 2.0 * jnp.exp(-muj2 * p[3])  # i=4, alpha=-2

        # Final constant term (alpha = 2 for i=5, p[4]=0 so exp(0)=1)
        # From line 265 in Fortran: u = u + 0.5 * alpha
        # where alpha = 2 after N=4 iterations
        u = u + 1.0

        # Compute rho
        rho = -u / muj2

        # Calculate the function value H(x)
        h_val = t

        # Add linear term (S'*rho)
        h_val = h_val + jnp.dot(s, rho)

        # Add quadratic term (rho'*R*rho)
        h_val = h_val + jnp.dot(rho, jnp.dot(r, rho))

        # The constraint from SIF is: CON = -eps^2 - H(x) >= 0
        # Based on HS88/89 experience, we need the adjustment
        eps_sqr = 0.0001

        # Return the constraint value in pycutest format
        inequalities = jnp.array([eps_sqr - h_val])

        return None, inequalities

    @property
    def expected_result(self):
        """Expected result from SIF file."""
        # HS90SOL: X1=0.821443, X2=7.74077e-68≈0, X3=-0.690910, X4=-0.456045
        return jnp.array([0.821443, 0.0, -0.690910, -0.456045], dtype=jnp.float64)

    @property
    def expected_objective_value(self):
        """Expected objective value from SIF file."""
        # XL HS90SOL = 1.36009 (not explicitly given, but similar to HS88/89)
        return None  # Not provided in SIF file
