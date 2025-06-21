import jax.numpy as jnp

from ._lsc_base import _AbstractLSC


# TODO: Human review needed to verify the implementation matches the problem definition
class LSC2LS(_AbstractLSC):
    """Fit a circle to a set of 2D points: case 2, data points in a small arc.

    Source: Problem from the SciPy cookbook
    http://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html

    SIF input: Nick Gould, Nov 2016
    Classification: SUR2-MN-3-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def _data(self):
        # Data points from the SIF file
        x = jnp.array([36.0, 36.0, 19.0, 18.0, 33.0, 26.0])
        y = jnp.array([14.0, 10.0, 28.0, 31.0, 18.0, 26.0])
        return x, y

    def y0(self):
        """Get the starting point based on the y0_id."""
        if self.y0_id == 0:
            # START1 (lines 64-66)
            return jnp.array([98.0, 36.0, 270.0])
        elif self.y0_id == 1:
            # START2 (lines 70-72)
            return jnp.array([9.8, 3.6, 27.0])
        else:
            assert False, f"y0_id must be 0 or 1, got {self.y0_id}"
