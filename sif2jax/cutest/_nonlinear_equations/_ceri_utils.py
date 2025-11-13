"""Utilities for CERI651 problems."""

from jax import numpy as jnp
from jax.scipy.special import erfc
from jaxtyping import Array


def erfc_scaled(z: Array) -> Array:
    """Scaled complementary error function: exp(z^2) * erfc(z)

    Uses more accurate asymptotic expansion for large z.
    """
    # For large positive z, use more terms in asymptotic expansion
    # erfcx(z) ≈ 1/(sqrt(pi)*z) * (1 - 0.5/z^2 + 0.75/z^4 - ...)
    sqrt_pi = jnp.sqrt(jnp.pi)

    # Use more terms for better accuracy
    def asymptotic(z):
        z2 = z * z
        return 1.0 / (sqrt_pi * z) * (1.0 - 0.5 / z2 + 0.75 / (z2 * z2))

    result = jnp.where(
        z > 5.0,
        asymptotic(z),
        jnp.exp(z * z) * erfc(z),
    )

    # Ensure result is an Array, not a tuple
    return jnp.asarray(result)
