"""Configuration for benchmark tests.

This conftest.py ensures that benchmark tests are isolated from the main test suite
configuration, preventing any interference from parametrization or fixtures defined in
the main tests/conftest.py.
"""

import jax


# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)
