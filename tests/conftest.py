import jax


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_numpy_rank_promotion", "raise")
jax.config.update("jax_numpy_dtype_promotion", "strict")


def pytest_addoption(parser):
    parser.addoption(
        "--test-case",
        action="store",
        default=None,
        help=(
            "Run only specified test case(s). "
            "Separate multiple test cases with a comma, don't add white space."
        ),
    )
    parser.addoption(
        "--runtime-threshold",
        action="store",
        default=None,
        help="Runtime ratio threshold for JAX/pycutest comparison (default: 5.0)",
    )
