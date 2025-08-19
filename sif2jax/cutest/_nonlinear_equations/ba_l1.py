import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


# TODO: Human review needed - SIF file has undefined behavior
# The BA function in the SIF file has a design flaw where:
# 1. When YRES=1 (copy_y=true), it jumps to label 1 and does rx=ry
# 2. But ry is INTENT(INOUT) and never computed when copy_y=true
# 3. This causes EY elements to use uninitialized ry values
# 4. Results in garbage values (~-200) for Y-residuals of cameras 27,30,37
# 5. Our implementation is mathematically correct but can't match pycutest
# Resources needed: Fix the SIF file's BA function or override test expectations
class BA_L1(AbstractNonlinearEquations):
    """BA-L1 - Bundle Adjustment problem.

    Bundle Adjustment problem from reconstructive geometry in which
    a collection of photographs is used to determine the position of
    a set of observed points. Each observed point is seen via its
    two-dimensional projections on a subset of the photographs. The
    solution is found by solving a large nonlinear least-squares problem.
    This variant is given as an inconsistent set of nonlinear equations.

    Source: data from the Bundle Adjustment in the Large
    project, http://grail.cs.washington.edu/projects/bal/

    Ladybug datasets (single image extracted)

    SIF input: Nick Gould, November 2016

    classification NOR2-MN-57-12
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def name(self):
        return "BA-L1"

    # Problem dimensions
    n_images = 49  # Not all used, only 6 observations
    n_points = 1
    n_observs = 6

    # Observation image indices (which cameras see the point)
    obs_images = jnp.array([1, 2, 4, 27, 30, 37], dtype=jnp.int32) - 1  # 0-indexed

    # Observed 2D projections
    observations = jnp.array(
        [
            [-332.65, 262.09],  # Image 1
            [-199.76, 166.7],  # Image 2
            [-253.06, 202.27],  # Image 4
            [58.13, 271.89],  # Image 27
            [238.22, 237.37],  # Image 30
            [317.55, 221.15],  # Image 37
        ]
    )

    @property
    def y0(self):
        # Starting point from SIF file
        # 3D point coordinates
        x = jnp.array([-0.6120001572, 0.57175904776, -1.847081276])

        # Camera parameters for the 6 observations
        # Each camera has: RX, RY, RZ, TX, TY, TZ, KA, KB, F
        cameras = jnp.array(
            [
                # Image 1
                [
                    0.01574151594,
                    -0.0127909362,
                    -0.0044008498,
                    -0.0340938396,
                    -0.107513871,
                    1.1202240291,
                    -3.177064e-7,
                    5.882049e-13,
                    399.75152639,
                ],
                # Image 2
                [
                    0.01597732412,
                    -0.0252244646,
                    -0.0094001416,
                    -0.0085667661,
                    -0.1218804907,
                    0.7190133075,
                    -3.780477e-7,
                    9.307431e-13,
                    402.01753386,
                ],
                # Image 4
                [
                    0.01484625118,
                    -0.0210628994,
                    -0.001166948,
                    -0.0249509707,
                    -0.1139847055,
                    0.92166020737,
                    -3.295265e-7,
                    6.732885e-13,
                    400.40175368,
                ],
                # Image 27
                [
                    0.01991666998,
                    -1.22433082,
                    0.0119988756,
                    -1.411897512,
                    -0.1148065151,
                    0.44915582738,
                    5.95875e-8,
                    -2.48391e-13,
                    407.03024568,
                ],
                # Image 30
                [
                    0.02082242153,
                    -1.238434791,
                    0.01389314763,
                    -1.049686225,
                    -0.1299513286,
                    0.33798380231,
                    4.5673127e-8,
                    -1.79243e-13,
                    405.91764962,
                ],
                # Image 37
                [
                    0.01658816461,
                    -1.247226838,
                    0.01846788123,
                    -0.8617315756,
                    -0.1321089362,
                    0.28256800868,
                    4.7465711e-8,
                    -1.50881e-13,
                    404.73590637,
                ],
            ]
        )

        # Flatten: X, Y, Z, then camera params
        return jnp.concatenate([x, cameras.ravel()])

    @property
    def args(self):
        return None

    def _rodrigues_rotate(self, rvec, point):
        """Apply Rodrigues rotation to a point."""
        # Rodrigues rotation formula
        theta_sq = jnp.sum(rvec**2)

        # Handle small angle case
        small_angle = theta_sq < 1e-6

        # Standard case
        theta = jnp.sqrt(theta_sq + 1e-30)  # Add small value for stability
        cos_theta = jnp.cos(theta)
        sin_theta = jnp.sin(theta)
        one_minus_cos = 1.0 - cos_theta

        # Normalized rotation axis
        k = rvec / (theta + 1e-30)

        # Cross product k x point
        kxp = jnp.array(
            [
                k[1] * point[2] - k[2] * point[1],
                k[2] * point[0] - k[0] * point[2],
                k[0] * point[1] - k[1] * point[0],
            ]
        )

        # Dot product k . point
        kdotp = jnp.sum(k * point)

        # Rodrigues formula: p*cos(θ) + (k×p)*sin(θ) + k*(k·p)*(1-cos(θ))
        rotated_std = point * cos_theta + kxp * sin_theta + k * kdotp * one_minus_cos

        # Small angle approximation: p + rvec × p
        cross_prod = jnp.array(
            [
                rvec[1] * point[2] - rvec[2] * point[1],
                rvec[2] * point[0] - rvec[0] * point[2],
                rvec[0] * point[1] - rvec[1] * point[0],
            ]
        )
        rotated_small = point + cross_prod

        # Select based on angle size
        return jnp.where(small_angle, rotated_small, rotated_std)

    def _project_point(self, point_3d, camera_params):
        """Project a 3D point through a camera."""
        # Extract camera parameters
        rvec = camera_params[0:3]  # Rotation vector (Rodrigues)
        tvec = camera_params[3:6]  # Translation
        k1 = camera_params[6]  # Radial distortion coefficient 1
        k2 = camera_params[7]  # Radial distortion coefficient 2
        f = camera_params[8]  # Focal length

        # Apply rotation and translation
        rotated = self._rodrigues_rotate(rvec, point_3d)
        translated = rotated + tvec

        # Project to normalized image plane
        px = -translated[0] / translated[2]
        py = -translated[1] / translated[2]

        # Apply radial distortion
        r_sq = px**2 + py**2
        distortion = 1.0 + k1 * r_sq + k2 * r_sq**2

        # Apply focal length and distortion
        u = f * distortion * px
        v = f * distortion * py

        return jnp.array([u, v])

    def constraint(self, y):
        # Extract 3D point and camera parameters
        point_3d = y[0:3]
        cameras = y[3:].reshape(self.n_observs, 9)

        # Compute projections and residuals for each observation
        residuals = []
        for i in range(self.n_observs):
            projection = self._project_point(point_3d, cameras[i])
            residual = projection - self.observations[i]
            residuals.append(residual)

        # Flatten residuals
        return jnp.concatenate(residuals), None

    @property
    def bounds(self):
        return None

    @property
    def expected_result(self):
        return None

    @property
    def expected_objective_value(self):
        return None  # Since this is a system of equations
