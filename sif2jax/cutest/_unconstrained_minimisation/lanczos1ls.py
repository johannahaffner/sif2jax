import jax
import jax.numpy as jnp

from ._lanczos_base import _AbstractLanczos


# TODO: Human review needed to verify the implementation matches the problem definition
class LANCZOS1LS(_AbstractLanczos):
    """NIST Data fitting problem LANCZOS1.

    In LANCZOS1, the y values are artificially created using the exact model
    with known parameter values.
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def _data(self):
        """Initialize y_values based on the exact model."""
        # The y values are artificially created with known parameter values
        exact_params = jnp.array([0.0951, 1.0, 0.8607, 3.0, 1.5576, 5.0])
        x_values = jnp.array(
            [
                0.00,
                0.05,
                0.10,
                0.15,
                0.20,
                0.25,
                0.30,
                0.35,
                0.40,
                0.45,
                0.50,
                0.55,
                0.60,
                0.65,
                0.70,
                0.75,
                0.80,
                0.85,
                0.90,
                0.95,
                1.00,
                1.05,
                1.10,
                1.15,
            ]
        )
        y = jax.vmap(lambda x: self.model(x, exact_params))(x_values)
        return y

    def expected_result(self):
        """The exact solution for LANCZOS1."""
        return jnp.array([0.0951, 1.0, 0.8607, 3.0, 1.5576, 5.0])
