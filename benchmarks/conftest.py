"""Configuration for benchmark tests."""

import jax

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)
