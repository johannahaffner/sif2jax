import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractBoundedMinimisation


class SANTALS(AbstractBoundedMinimisation):
    """SANTALS problem.

    TODO: Human review needed
    Attempts made: [bounded minimization, variable structure from SANTA, L2 residuals]
    Suspected issues: [Small gradient discrepancies (~1e-9) with pycutest, likely
                      numerical precision]
    Resources needed: [Review SIF element definitions and L2 group handling]

    The Santa problem as suggested in a Christmas competition
    by Jens Jensen (Scientific Computing, STFC). This is the least-squares
    version of the SANTA problem.

    Santa is flying around the world, presently presenting presents. The Earth is
    modeled as a perfect sphere with radius precisely 6,371,000 metres. Santa sets
    off from the North Pole along 2°6'57.6" E bearing south and visits various
    locations, leaving an elf behind at each location to help unwrap presents.

    The problem is to find Santa's route given the distances traveled between locations.
    The constraints are given by the spherical law of cosines:
    sin phi_1 sin phi_2 + cos phi_1 cos phi_2 cos(lam_1 - lam_2) = cos(d/r)

    This is the least-squares version where each constraint becomes a squared residual.

    Source:
    Jens Jensen, SCD Christmas programming challenge 2016

    SIF input: Nick Gould, Dec 2016.

    Classification: SBR2-AN-21-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return 21

    def objective(self, y, args):
        """Compute the objective function.

        This is the least-squares formulation where each spherical law of cosines
        constraint becomes a squared residual in the objective.
        """
        del args

        # Constants from SANTA implementation
        pi = jnp.pi
        phi0 = 90.0 * pi / 180.0  # North Pole
        lam0 = 0.0 * pi / 180.0
        phi12 = -31.77844444 * pi / 180.0
        lam12 = -144.77025 * pi / 180.0
        lam1 = 2.116 * pi / 180.0  # Santa's initial direction
        radius = 6371000.0

        # Path lengths between nodes (in metres) - same as SANTA
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

        # Extract variables - same structure as SANTA
        # y = [PHI1, PHI2, LAM2, PHI3, LAM3, ..., PHI11, LAM11]
        phi1 = y[0]
        phi_lam_pairs = y[1:].reshape(-1, 2)
        phis = jnp.concatenate([jnp.array([phi1]), phi_lam_pairs[:, 0]])
        lams = jnp.concatenate([jnp.array([lam1]), phi_lam_pairs[:, 1]])

        # Create coordinate arrays including fixed points
        all_phis = jnp.concatenate([jnp.array([phi0]), phis, jnp.array([phi12])])
        all_lams = jnp.concatenate([jnp.array([lam0]), lams, jnp.array([lam12])])

        # Compute sum of squared residuals (L2 formulation)
        total = inexact_asarray(jnp.array(0.0))

        for (i, j), distance in distances.items():
            phi_i, phi_j = all_phis[i], all_phis[j]
            lam_i, lam_j = all_lams[i], all_lams[j]

            # Spherical law of cosines
            expected_cos = jnp.cos(distance / radius)
            actual_cos = jnp.sin(phi_i) * jnp.sin(phi_j) + jnp.cos(phi_i) * jnp.cos(
                phi_j
            ) * jnp.cos(lam_i - lam_j)

            # Squared residual
            residual = actual_cos - expected_cos
            total += residual * residual

        return total

    @property
    def bounds(self):
        """Variable bounds from SIF file."""
        # From SIF: LO SANTA 'DEFAULT' -1000.0, UP SANTA 'DEFAULT' 1000.0
        n = self.n
        lower = inexact_asarray(jnp.full(n, -1000.0))
        upper = inexact_asarray(jnp.full(n, 1000.0))
        return lower, upper

    @property
    def y0(self):
        """Initial guess from SIF file."""
        return inexact_asarray(
            jnp.array(
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
        )

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def expected_result(self):
        """Expected result not provided in SIF file."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value from SIF file."""
        # From SIF comment: LO SANTA 0.0 (commented out)
        # This suggests the optimal value should be close to 0
        return inexact_asarray(jnp.array(0.0))
