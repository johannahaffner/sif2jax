import jax


jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_numpy_rank_promotion", "raise")
jax.config.update("jax_numpy_dtype_promotion", "standard")
