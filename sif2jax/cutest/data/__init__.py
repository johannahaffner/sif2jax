"""Data loading utilities for CUTEst problems."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np


_DATA_DIR = Path(__file__).parent


def load_fbrain_data(variant="fbrain"):
    """Load FBRAIN data from .npz files.

    Args:
        variant: One of "fbrain", "fbrainls", "fbrainne"

    Returns:
        tuple: (A_coeffs, A_lambdas, B_coeffs, B_lambdas, R_values) as JAX arrays
    """
    npz_path = _DATA_DIR / f"{variant}.npz"

    if not npz_path.exists():
        raise FileNotFoundError(f"FBRAIN data file not found: {npz_path}")

    with np.load(npz_path) as data:
        # Load as default dtype (respects JAX's X64 configuration)
        return (
            jnp.asarray(data["A_coeffs"]),
            jnp.asarray(data["A_lambdas"]),
            jnp.asarray(data["B_coeffs"]),
            jnp.asarray(data["B_lambdas"]),
            jnp.asarray(data["R_values"]),
        )


__all__ = ["load_fbrain_data"]
