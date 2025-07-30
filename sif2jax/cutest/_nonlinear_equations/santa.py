import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class SANTA(AbstractNonlinearEquations):
    """
    The Santa problem as suggested in a Christmas competition by Jens Jensen
    (Scientific Computing, STFC).

    Santa is flying around the world, presently presenting presents. The Earth is
    modeled as a perfect sphere with radius precisely 6,371,000 metres. Santa sets
    off from the North Pole along 2°6'57.6" E bearing south and visits various
    locations, leaving an elf behind at each
    location to help unwrap presents.

    The problem is to find Santa's route given the distances traveled between locations.
    The constraints are given by the spherical law of cosines:
    sin phi_1 sin phi_2 + cos phi_1 cos phi_2 cos(lam_1 - lam_2) = cos(d/r)

    The problem has many local minimizers of the sum of squares of infeasibility,
    but only the solution with zero residuals is of interest.

    Source:
    Jens Jensen, SCD Christmas programming challenge 2016

    SIF input: Nick Gould, Dec 2016.

    classification NOR2-AN-21-23
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 21  # Number of variables
    m: int = 23  # Number of equations

    @property
    def y0(self):
        # Starting point from SIF file
        return jnp.array(
            [
                0.7223835215,  # PHI1
                0.8069093428,  # PHI2
                -0.031657133,  # LAM2
                0.9310164154,  # PHI3
                0.1199353230,  # LAM3
                6.6067392710,  # PHI4
                -1.214314477,  # LAM4
                -3.530946794,  # PHI5
                2.5329493980,  # LAM5
                -9.798251905,  # PHI6
                4.3021328700,  # LAM6
                14.632267534,  # PHI7
                -12.96253311,  # LAM7
                2.0349445303,  # PHI8
                -4.050000443,  # LAM8
                -28.45607804,  # PHI9
                22.430117198,  # LAM9
                16.034035489,  # PHI10
                -17.28050167,  # LAM10
                0.8717052037,  # PHI11
                -0.833052840,  # LAM11
            ]
        )

    @property
    def args(self):
        # Constants from the problem
        pi = jnp.pi

        # Initial and final positions (converted to radians)
        phi0 = 90.0 * pi / 180.0  # North Pole
        lam0 = 0.0 * pi / 180.0
        phi12 = -31.77844444 * pi / 180.0
        lam12 = -144.77025 * pi / 180.0

        # Santa's initial direction
        lam1 = 2.116 * pi / 180.0

        # Radius of Earth in metres
        radius = 6371000.0

        # Path lengths between nodes (in metres)
        distances = {
            (0, 1): 5405238.0,
            (0, 2): 4866724.0,
            (1, 2): 623852.0,
            (1, 3): 1375828.0,
            (2, 3): 1005461.0,
            (2, 4): 6833740.0,
            (3, 4): 7470967.0,
            (3, 5): 4917407.0,
            (4, 5): 3632559.0,
            (4, 6): 13489586.0,
            (5, 6): 10206818.0,
            (5, 7): 10617953.0,
            (6, 7): 7967212.0,
            (6, 8): 9195575.0,
            (7, 8): 5896361.0,
            (7, 9): 10996150.0,
            (8, 9): 8337266.0,
            (8, 10): 9704793.0,
            (9, 10): 13019505.0,
            (9, 11): 7901038.0,
            (10, 11): 8690818.0,
            (10, 12): 12498127.0,
            (11, 12): 8971302.0,
        }

        return phi0, lam0, phi12, lam12, lam1, radius, distances

    def residual(self, y, args):
        """Compute residuals for the spherical law of cosines constraints."""
        phi0, lam0, phi12, lam12, lam1, radius, distances = args

        # Extract variables - note the structure:
        # y[0] = PHI1 (location 1 has only phi, no lambda since lambda1 is fixed)
        # y[1] = PHI2, y[2] = LAM2 (location 2)
        # y[3] = PHI3, y[4] = LAM3 (location 3)
        # ...
        # y[19] = PHI11, y[20] = LAM11 (location 11)
        phi1 = y[0]
        # phi[i] = phi_{i+2} for i=0..9
        phi = jnp.array(
            [y[1], y[3], y[5], y[7], y[9], y[11], y[13], y[15], y[17], y[19]]
        )
        # lam[i] = lam_{i+2} for i=0..9
        lam = jnp.array(
            [y[2], y[4], y[6], y[8], y[10], y[12], y[14], y[16], y[18], y[20]]
        )

        # Helper function to compute spherical law of cosines residual
        def spherical_residual(phi_i, lam_i, phi_j, lam_j, dist_ij):
            d_rad = dist_ij / radius
            cos_d = jnp.cos(d_rad)
            return (
                jnp.sin(phi_i) * jnp.sin(phi_j)
                + jnp.cos(phi_i) * jnp.cos(phi_j) * jnp.cos(lam_j - lam_i)
                - cos_d
            )

        residuals = []

        # Constraint R0,1: between North Pole and location 1
        res = spherical_residual(phi0, lam0, phi1, lam1, distances[(0, 1)])
        residuals.append(res)

        # Constraint R0,2: between North Pole and location 2
        res = spherical_residual(phi0, lam0, phi[0], lam[0], distances[(0, 2)])
        residuals.append(res)

        # Constraint R1,2: between location 1 and location 2
        res = spherical_residual(phi1, lam1, phi[0], lam[0], distances[(1, 2)])
        residuals.append(res)

        # Constraint R1,3: between location 1 and location 3
        res = spherical_residual(phi1, lam1, phi[1], lam[1], distances[(1, 3)])
        residuals.append(res)

        # Constraint R2,3: between location 2 and location 3
        res = spherical_residual(phi[0], lam[0], phi[1], lam[1], distances[(2, 3)])
        residuals.append(res)

        # Constraints for other edges - following the pattern in the SIF file
        # Note: phi[i-2] corresponds to location i, lam[i-2] corresponds to location i
        for i in range(2, 10):
            # Constraint R(i,i+2)
            if (i, i + 2) in distances:
                phi_i = phi[i - 2]
                lam_i = lam[i - 2]
                phi_j = phi[i]  # i+2-2 = i
                lam_j = lam[i]
                res = spherical_residual(
                    phi_i, lam_i, phi_j, lam_j, distances[(i, i + 2)]
                )
                residuals.append(res)

            # Constraint R(i+1,i+2)
            if (i + 1, i + 2) in distances:
                phi_i = phi[i - 1]  # (i+1)-2 = i-1
                lam_i = lam[i - 1]
                phi_j = phi[i]  # (i+2)-2 = i
                lam_j = lam[i]
                res = spherical_residual(
                    phi_i, lam_i, phi_j, lam_j, distances[(i + 1, i + 2)]
                )
                residuals.append(res)

        # Constraints R10,12 and R11,12: connections to final location
        res = spherical_residual(phi[8], lam[8], phi12, lam12, distances[(10, 12)])
        residuals.append(res)

        res = spherical_residual(phi[9], lam[9], phi12, lam12, distances[(11, 12)])
        residuals.append(res)

        return jnp.array(residuals)

    def constraint(self, y):
        """Returns the residuals as equality constraints."""
        return self.residual(y, self.args), None

    @property
    def bounds(self):
        # Bounds from -1000 to 1000 as specified in the SIF file
        lower = jnp.full(self.n, -1000.0)
        upper = jnp.full(self.n, 1000.0)
        return lower, upper

    @property
    def expected_result(self):
        # The optimal solution is not explicitly given in the SIF file
        return None

    @property
    def expected_objective_value(self):
        # Zero residuals expected for the correct Santa route
        return jnp.array(0.0)
