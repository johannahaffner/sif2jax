"""Runtime benchmarks comparing JAX implementations against pycutest."""

import time
from collections.abc import Callable

import jax
import jax.numpy as jnp
import pycutest  # pyright: ignore[reportMissingImports]
import pytest  # pyright: ignore[reportMissingImports]
import sif2jax


# Default runtime ratio threshold (JAX time / pycutest time)
DEFAULT_THRESHOLD = 5.0


def pytest_generate_tests(metafunc):
    if "problem" in metafunc.fixturenames:
        requested = metafunc.config.getoption("--test-case")
        test_cases = get_test_cases(requested)
        metafunc.parametrize("problem", test_cases, scope="class")


def get_test_cases(requested):
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
        return tuple(test_cases)
    else:
        # For runtime benchmarks, use a representative subset by default
        # Include small, medium, and large problems of different types
        default_problems = [
            "ROSENBR",  # Small unconstrained
            "ARWHEAD",  # Medium unconstrained
            "HS10",  # Small constrained
            "HS76",  # Medium constrained
            "BOX",  # Small unconstrained
            "BROWNDEN",  # Medium unconstrained
        ]
        test_cases = []
        for name in default_problems:
            try:
                test_case = sif2jax.cutest.get_problem(name)
                if test_case is not None:
                    test_cases.append(test_case)
            except Exception:
                pass  # Skip if problem not available
        return tuple(test_cases)


def benchmark_function(
    func: Callable, *args, warmup: int = 3, iterations: int = 10
) -> float:
    """Benchmark a function with warmup runs.

    Note: While JAX only needs 1 warmup for JIT compilation, we use 3 for consistency
    between JAX and pycutest benchmarks. This also helps ensure stable timings.
    """
    # Warmup runs
    for _ in range(warmup):
        _ = func(*args)

    # Timed runs
    start = time.perf_counter()
    for _ in range(iterations):
        _ = func(*args)
    end = time.perf_counter()

    return (end - start) / iterations


class TestRuntime:
    """Runtime benchmarks comparing JAX to pycutest."""

    @pytest.fixture(scope="class")
    def pycutest_problem(self, problem):
        """Load pycutest problem once per problem per class."""
        return pycutest.import_problem(problem.name)

    @pytest.fixture(autouse=True)
    def clear_jax_cache(self):
        """Clear JAX cache before each test to ensure fair comparison."""
        jax.clear_caches()
        yield
        jax.clear_caches()

    @pytest.fixture
    def threshold(self, request):
        """Get runtime ratio threshold from command line or use default."""
        threshold_value = request.config.getoption("--runtime-threshold", default=None)
        return float(threshold_value) if threshold_value else DEFAULT_THRESHOLD

    def test_objective_runtime(self, problem, pycutest_problem, threshold):
        """Compare objective function runtime."""
        # Get starting point
        x0 = problem.y0()
        args = problem.args()

        # Compile JAX function
        jax_obj = jax.jit(problem.objective)

        # Benchmark pycutest
        pycutest_time = benchmark_function(pycutest_problem.obj, x0)

        # Benchmark JAX
        jax_time = benchmark_function(jax_obj, x0, args)

        # Calculate ratio
        ratio = jax_time / pycutest_time if pycutest_time > 0 else float("inf")

        # Print results (visible with -s flag)
        print(f"\nObjective runtime for {problem.name}:")
        print(f"  pycutest: {pycutest_time * 1000:.3f} ms")
        print(f"  JAX:      {jax_time * 1000:.3f} ms")
        print(f"  Ratio:    {ratio:.2f}x")

        # Assert within threshold
        assert ratio < threshold, (
            f"JAX objective is {ratio:.2f}x slower than pycutest "
            f"(threshold: {threshold})"
        )

    def test_gradient_runtime(self, problem, pycutest_problem, threshold):
        """Compare gradient computation runtime."""
        # Get starting point
        x0 = problem.y0()
        args = problem.args()

        # Compile JAX gradient
        jax_grad = jax.jit(jax.grad(problem.objective))

        # Benchmark pycutest
        pycutest_time = benchmark_function(pycutest_problem.grad, x0)

        # Benchmark JAX
        jax_time = benchmark_function(jax_grad, x0, args)

        # Calculate ratio
        ratio = jax_time / pycutest_time if pycutest_time > 0 else float("inf")

        # Print results (visible with -s flag)
        print(f"\nGradient runtime for {problem.name}:")
        print(f"  pycutest: {pycutest_time * 1000:.3f} ms")
        print(f"  JAX:      {jax_time * 1000:.3f} ms")
        print(f"  Ratio:    {ratio:.2f}x")

        # Assert within threshold
        assert ratio < threshold, (
            f"JAX gradient is {ratio:.2f}x slower than pycutest "
            f"(threshold: {threshold})"
        )

    def test_combined_runtime(self, problem, pycutest_problem, threshold):
        """Compare combined objective and gradient runtime."""

        # Get starting point
        x0 = problem.y0()
        args = problem.args()

        # Create combined JAX function using value_and_grad
        jax_obj_and_grad = jax.jit(jax.value_and_grad(problem.objective))

        # For pycutest, we'll time obj and grad separately since objgrad
        # might not be available
        def pycutest_obj_and_grad(x):
            obj = pycutest_problem.obj(x)
            grad = pycutest_problem.grad(x)
            return obj, grad

        # Benchmark pycutest
        pycutest_time = benchmark_function(pycutest_obj_and_grad, x0)

        # Benchmark JAX - note that value_and_grad returns (value, grad)
        # so we need a wrapper
        def jax_wrapper(x):
            return jax_obj_and_grad(x, args)

        jax_time = benchmark_function(jax_wrapper, x0)

        # Calculate ratio
        ratio = jax_time / pycutest_time if pycutest_time > 0 else float("inf")

        # Print results (visible with -s flag)
        print(f"\nCombined obj+grad runtime for {problem.name}:")
        print(f"  pycutest: {pycutest_time * 1000:.3f} ms")
        print(f"  JAX:      {jax_time * 1000:.3f} ms")
        print(f"  Ratio:    {ratio:.2f}x")

        # Assert within threshold
        assert ratio < threshold, (
            f"JAX combined is {ratio:.2f}x slower than pycutest "
            f"(threshold: {threshold})"
        )

    def test_constraint_runtime(self, problem, pycutest_problem, threshold):
        """Compare constraint function runtime."""
        # Skip if problem is unconstrained
        if not isinstance(problem, sif2jax.AbstractConstrainedMinimisation):
            pytest.skip("Problem has no constraints")

        # Get starting point
        x0 = problem.y0()

        # Compile JAX constraint function
        jax_cons = jax.jit(problem.constraint)

        # Benchmark pycutest
        pycutest_time = benchmark_function(pycutest_problem.cons, x0)

        # Benchmark JAX
        jax_time = benchmark_function(jax_cons, x0)

        # Calculate ratio
        ratio = jax_time / pycutest_time if pycutest_time > 0 else float("inf")

        # Print results (visible with -s flag)
        print(f"\nConstraint runtime for {problem.name}:")
        print(f"  pycutest: {pycutest_time * 1000:.3f} ms")
        print(f"  JAX:      {jax_time * 1000:.3f} ms")
        print(f"  Ratio:    {ratio:.2f}x")

        # Assert within threshold
        assert ratio < threshold, (
            f"JAX constraint is {ratio:.2f}x slower than pycutest "
            f"(threshold: {threshold})"
        )

    def test_constraint_jacobian_runtime(self, problem, pycutest_problem, threshold):
        """Compare constraint Jacobian computation runtime."""
        # Skip if problem is unconstrained
        if not isinstance(problem, sif2jax.AbstractConstrainedMinimisation):
            pytest.skip("Problem has no constraints")

        # Get starting point
        x0 = problem.y0()

        # Create JAX Jacobian function
        # For constrained problems, we want the Jacobian of all constraints
        def constraint_wrapper(x):
            eq_cons, ineq_cons = problem.constraint(x)
            # Concatenate all constraints
            all_cons = []
            if eq_cons is not None:
                all_cons.append(eq_cons)
            if ineq_cons is not None:
                all_cons.append(ineq_cons)
            return jnp.concatenate(all_cons) if all_cons else jnp.array([])

        jax_jac = jax.jit(jax.jacfwd(constraint_wrapper))

        # Benchmark pycutest - use dense Jacobian
        # For constrained problems, cjac returns (gradient, Jacobian)
        if hasattr(pycutest_problem, "cjac"):

            def pycutest_jac_func(x):
                _, J = pycutest_problem.cjac(x)
                return J
        else:
            # Skip if no Jacobian method available
            pytest.skip("No Jacobian method available in pycutest")

        pycutest_time = benchmark_function(pycutest_jac_func, x0)

        # Benchmark JAX
        jax_time = benchmark_function(jax_jac, x0)

        # Calculate ratio
        ratio = jax_time / pycutest_time if pycutest_time > 0 else float("inf")

        # Print results (visible with -s flag)
        print(f"\nConstraint Jacobian runtime for {problem.name}:")
        print(f"  pycutest: {pycutest_time * 1000:.3f} ms")
        print(f"  JAX:      {jax_time * 1000:.3f} ms")
        print(f"  Ratio:    {ratio:.2f}x")

        # Assert within threshold
        assert ratio < threshold, (
            f"JAX constraint Jacobian is {ratio:.2f}x slower than pycutest "
            f"(threshold: {threshold})"
        )
