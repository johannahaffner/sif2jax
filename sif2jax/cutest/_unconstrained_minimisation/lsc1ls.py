import jax.numpy as jnp

from ._lsc_base import _AbstractLSC


# TODO: Human review needed to verify the implementation matches the problem definition
class LSC1LS(_AbstractLSC):
    """Fit a circle to a set of 2D points: case 1, data points surround the circle.

    Source: Problem from the SciPy cookbook
    http://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html

    SIF input: Nick Gould, Nov 2016
    Classification: SUR2-MN-3-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def _data(self):
        """Data points from the SIF file."""
        # Data points from the SIF file (lines 33-44)
        x = jnp.array([9.0, 35.0, -13.0, 10.0, 23.0, 0.0])
        y = jnp.array([34.0, 10.0, 6.0, -14.0, 27.0, -10.0])
        return x, y

    def y0(self):
        """Get the starting point based on the y0_id."""
        if self.y0_id == 0:
            # START1 (lines 64-66)
            return jnp.array([105.0, 96.0, 230.0])
        elif self.y0_id == 1:
            # START2 (lines 70-72)
            return jnp.array([10.5, 9.6, 23.0])
        else:
            assert False, f"y0_id must be 0 or 1, got {self.y0_id}"
