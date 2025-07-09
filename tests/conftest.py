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
    parser.addoption(
        "--start-at-letter",
        action="store",
        default=None,
        help=(
            "Skip problems starting with letters before the specified letter "
            "(e.g., --start-at-letter=M skips A-L)"
        ),
    )


def pytest_generate_tests(metafunc):
    """Generate test cases for parametrization."""
    if "problem" in metafunc.fixturenames:
        import sif2jax

        requested = metafunc.config.getoption("--test-case")
        start_at_letter = metafunc.config.getoption("--start-at-letter")

        if requested is not None:
            # Split by comma and strip whitespace
            test_case_names = [name.strip() for name in requested.split(",")]
            test_cases = []

            for name in test_case_names:
                try:
                    test_case = sif2jax.cutest.get_problem(name)
                    assert (
                        test_case is not None
                    ), f"Test case '{name}' not found in sif2jax.cutest problems."
                    test_cases.append(test_case)
                except Exception as e:
                    raise RuntimeError(
                        f"Test case '{name}' not found in sif2jax.cutest problems."
                    ) from e
            test_cases = tuple(test_cases)
        else:
            # Get all available problems
            all_problems = list(sif2jax.problems)

            # Apply letter filter if specified
            if start_at_letter:
                start_letter = start_at_letter.upper()
                filtered_problems = []
                for problem in all_problems:
                    if problem.name[0].upper() >= start_letter:
                        filtered_problems.append(problem)
                test_cases = tuple(filtered_problems)
            else:
                test_cases = tuple(all_problems)

        metafunc.parametrize("problem", test_cases, scope="class")
