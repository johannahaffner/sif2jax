"""Miscellaneous utility functions for sif2jax.

This module contains utilities copied from the Optimistix library.
Source: https://github.com/patrick-kidger/optimistix/blob/9927984fb8cbec77f9514fad7af076dce64e3993/optimistix/_misc.py#L144
"""

import jax
import jax.numpy as jnp


def _asarray(dtype, x):
    return jnp.asarray(x, dtype=dtype)


# Work around JAX issue #15676
_asarray = jax.custom_jvp(_asarray, nondiff_argnums=(0,))


@_asarray.defjvp
def _asarray_jvp(dtype, x, tx):
    (x,) = x
    (tx,) = tx
    return _asarray(dtype, x), _asarray(dtype, tx)


def asarray(x):
    """Convert x to array with appropriate dtype."""
    dtype = jnp.result_type(x)
    return _asarray(dtype, x)


def inexact_asarray(x):
    """Convert x to array with inexact (floating) dtype."""
    dtype = jnp.result_type(x)
    if not jnp.issubdtype(jnp.result_type(x), jnp.inexact):
        dtype = jnp.float64
    return _asarray(dtype, x)
